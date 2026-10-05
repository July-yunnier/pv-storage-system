"""Model adapters. Names never imply equivalence to an unverified paper."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from experiments.protocol import ROOT


class Sunset(nn.Module):
    """Architecture port of official SUNSET (Conv-ReLU-BN-Pool order).

    Unified protocol changes frames/size and output horizons. This is an
    adaptation, not the original 16-frame 64px/10-fold 15min benchmark.
    """
    def __init__(self, frames=30, size=128, outputs=3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3 * frames, 24, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(24), nn.MaxPool2d(2),
                                  nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(48), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Linear(48 * (size // 4)**2 + frames, 1024), nn.ReLU(), nn.Dropout(.4),
                                  nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(.4), nn.Linear(1024, outputs))

    def forward(self, image, power):
        b, t, c, h, w = image.shape
        feat = self.conv(image.reshape(b, t * c, h, w)).flatten(1)
        return self.head(torch.cat([feat, power], dim=1))


class ConvLSTMCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels * 2, channels * 4, 3, padding=1)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c


def make_backbone(name):
    if name == "mobilenetv3":
        # Match the original architecture including final AvgPool to preserve
        # checkpoint behavior. Separate spatial version is an explicit ablation.
        net = models.mobilenet_v3_small(weights=None)
        return nn.Sequential(net.features, net.avgpool), 576, False
    if name == "mobilenetv3_spatial":
        return models.mobilenet_v3_small(weights=None).features, 576, False
    if name == "convnext":
        return models.convnext_tiny(weights=None).features, 768, False
    if name == "swin":
        return models.swin_t(weights=None).features, 768, True
    if name in ("vmamba", "spatial_mamba"):
        # Only load an explicitly supplied official adapter. A GRU/cumsum or
        # generic Mamba substitute would not reproduce selective 2D scanning.
        path = ROOT / "configs" / f"{name}_adapter.py"
        if not path.is_file():
            raise RuntimeError(f"Official {name} adapter absent: {path}; no substitute is labelled {name}")
        spec = importlib.util.spec_from_file_location(f"official_{name}_adapter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        net, channels = module.build_backbone(pretrained=False)
        return net, channels, False
    raise ValueError(name)


class SkyFusion(nn.Module):
    """MobileNetV3/ConvLSTM/PV-guided attention/gate, with controlled variants."""
    def __init__(self, backbone="mobilenetv3", channels=256, attention=True, fusion="gate", outputs=3):
        super().__init__()
        self.backbone, backbone_channels, self.channels_last = make_backbone(backbone)
        self.proj = nn.Conv2d(backbone_channels, channels, 1)
        self.cell = ConvLSTMCell(channels)
        self.query = nn.Linear(1, channels)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.power_lstm = nn.LSTM(1, 32, batch_first=True)
        self.image_proj = nn.Linear(channels, 64)
        self.power_proj = nn.Linear(32, 64)
        self.gate = nn.Linear(128, 2)
        self.attention = attention
        self.fusion = fusion
        self.head = nn.Sequential(nn.Linear(64 if fusion == "gate" else channels + 32, 128), nn.ReLU(), nn.Dropout(.3),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, outputs))

    def forward(self, image, power):
        b, t, c, size, _ = image.shape
        # Chunk frames for bounded memory; do not detach backbone gradients.
        features = self.backbone(image.reshape(b*t, c, size, size))
        if self.channels_last:
            features = features.permute(0, 3, 1, 2).contiguous()
        features = F.adaptive_avg_pool2d(self.proj(features), (8, 8))
        features = features.reshape(b, t, -1, 8, 8)
        h = features.new_zeros(b, self.cell.channels, 8, 8)
        c = torch.zeros_like(h)
        for frame in features.unbind(1):
            h, c = self.cell(frame, h, c)
        if self.attention:
            q = self.query(power.mean(dim=1, keepdim=True)).unsqueeze(1)
            weights = (q @ self.key(h).flatten(2) / math.sqrt(self.cell.channels)).softmax(dim=-1)
            v = (self.value(h).flatten(2) @ weights.transpose(1, 2)).reshape(b, -1, 1, 1)
            h = h * v
        image_vec = h.mean(dim=(2, 3))
        _, (hidden, _) = self.power_lstm(power.unsqueeze(-1))
        power_vec = hidden[0]
        if self.fusion == "gate":
            i = F.relu(self.image_proj(image_vec))
            p = F.relu(self.power_proj(power_vec))
            gate = self.gate(torch.cat([i, p], dim=1)).softmax(1)
            fused = gate[:, :1] * i + gate[:, 1:] * p
        else:
            fused = torch.cat([image_vec, power_vec], dim=1)
        return self.head(fused)


class FusionSFStyle(nn.Module):
    """Explicitly FusionSF-inspired cross-modal attention, not paper reproduction.

    NWP token construction happens upstream using issue-safe forecast vintages.
    """
    def __init__(self, sky_encoder, nwp_features=2, hidden=64):
        super().__init__()
        self.sky_encoder = sky_encoder
        self.sky_proj = nn.Linear(3, hidden)
        self.power_encoder = nn.GRU(1, hidden, batch_first=True)
        self.weather_encoder = nn.GRU(nwp_features, hidden, batch_first=True)
        self.cross = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.gate = nn.Linear(hidden * 3, 3)
        self.head = nn.Linear(hidden, 3)

    def forward(self, image, power, nwp):
        s = self.sky_proj(self.sky_encoder(image, power))
        p = self.power_encoder(power.unsqueeze(-1))[1][0]
        w = self.weather_encoder(nwp)[1][0]
        tokens = torch.stack([s, p, w], dim=1)
        attended = self.cross(tokens, tokens, tokens)[0]
        gate = self.gate(attended.flatten(1)).softmax(1)
        return self.head((attended * gate.unsqueeze(-1)).sum(1))


def official_ampdnet(outputs=16, ablation=False):
    path = ROOT / "research/official-ampdnet"
    sys.path.insert(0, str(path))
    try:
        import ConvPVNet
        cls = ConvPVNet.ConvPV7 if ablation else ConvPVNet.ConvPVNet
        return cls(M=4, L=64, T=outputs, D=64, P=4, S=2, r=2, dp=0., num_layers=2)
    finally:
        sys.path.remove(str(path))
