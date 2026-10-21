"""1711.11053v2 MQ-RNN 电价适配：直接多步分位数、全局/局部 MLP。

工程扩展：日/周滞后、训练集固定尺度、单调分位数头。无递归采样。
"""
from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

from app.domain.forecast.price_data import FEATURE_NAMES, calendar_features, validate_series
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = nn = F = None
from app.domain.forecast.result import ForecastModel, ForecastPoint, ForecastProvenance, TargetForecast

QUANTILES = [0.1, 0.5, 0.9]
FORMAT = "price_mqrnn_v1"
TARGET = "energy_price_excluding_transmission_and_fund"


def require_torch():
    if torch is None:
        raise RuntimeError("MQ-RNN 电价模型需要 PyTorch，请安装 requirements-ai.txt")


if nn is not None:
    class PriceMQRNN(nn.Module):
        def __init__(self, horizon: int, num_series: int = 1, hidden_size: int = 30,
                     context_size: int = 8, decoder_size: int = 64, embedding_size: int = 4):
            super().__init__()
            self.horizon = horizon
            self.context_size = context_size
            self.embedding = nn.Embedding(num_series, embedding_size)
            # 当前已观测价格、日滞后、周滞后 + 已知日历 + 静态序列嵌入
            self.encoder = nn.LSTM(3 + len(FEATURE_NAMES) + embedding_size,
                                   hidden_size, batch_first=True)
            self.global_mlp = nn.Sequential(
                nn.Linear(hidden_size + horizon * len(FEATURE_NAMES), decoder_size), nn.ReLU(),
                nn.Linear(decoder_size, (horizon + 1) * context_size))
            self.local_mlp = nn.Sequential(
                nn.Linear(2 * context_size + len(FEATURE_NAMES), decoder_size), nn.ReLU(),
                nn.Linear(decoder_size, len(QUANTILES)))

        def encode(self, observed, series_ids):
            embedding = self.embedding(series_ids)[:, None].expand(-1, observed.shape[1], -1)
            return self.encoder(torch.cat([observed, embedding], dim=-1))[0]

        def decode(self, states, future_features):
            """支持 [B,T,H,F] 多个 fork，也支持 [B,H,F] 单个预测原点。"""
            contexts = self.global_mlp(torch.cat([states, future_features.flatten(-2)], -1))
            contexts = contexts.reshape(*states.shape[:-1], self.horizon + 1, self.context_size)
            specific = contexts[..., :-1, :]
            shared = contexts[..., -1:, :].expand_as(specific)
            raw = self.local_mlp(torch.cat([specific, shared, future_features], -1))
            # 附加约束，非论文原始头：median ± softplus(delta)，避免分位数交叉。
            median = raw[..., 1]
            return torch.stack([median - F.softplus(raw[..., 0]), median,
                                median + F.softplus(raw[..., 2])], -1)

        def forward(self, observed, future_features, series_ids):
            # 一次编码，所有时刻共享解码器，一次反向传播聚合各 fork 的误差。
            return self.decode(self.encode(observed, series_ids), future_features)
else:
    class PriceMQRNN:
        def __init__(self, *args, **kwargs):
            require_torch()


def pinball_loss(prediction, target, mask=None):
    if mask is not None:
        target = torch.where(mask, target, torch.zeros_like(target))
    error = target[..., None] - prediction
    q = prediction.new_tensor(QUANTILES)
    losses = torch.maximum(q * error, (q - 1) * error)
    if mask is None:
        return losses.mean()
    # 用 where 而非乘零，避免缺失标签的 NaN 污染 loss。
    losses = torch.where(mask[..., None], losses, torch.zeros_like(losses))
    return losses.sum() / (mask.sum().clamp_min(1) * len(QUANTILES))


def observed_features(history, resolution, scale):
    """只构造已观测时刻的输入；滞后按原生频率对齐。"""
    day = 1440 // resolution
    lag = day * 7
    return [[value / scale, history[i - day][1] / scale, history[i - lag][1] / scale,
             *calendar_features(ts)] for i, (ts, value) in enumerate(history) if i >= lag]


def direct_quantiles(model, history, *, resolution, context, scale, series_index=0):
    required = context + 7 * 1440 // resolution
    if len(history) < required:
        raise ValueError(f"MQ-RNN price needs at least {required} native history points")
    history = history[-required:]
    validate_series(history, resolution)
    first = history[-1][0] + timedelta(minutes=resolution)
    future = [first + timedelta(minutes=resolution * k) for k in range(model.horizon)]
    x = torch.tensor([observed_features(history, resolution, scale)], dtype=torch.float32)
    f = torch.tensor([[calendar_features(ts) for ts in future]], dtype=torch.float32)
    with torch.inference_mode():
        states = model.encode(x, torch.tensor([series_index]))
        q = model.decode(states[:, -1], f)[0] * scale
    if not torch.isfinite(q).all():
        raise ValueError("MQ-RNN produced non-finite quantiles")
    return q


def load_artifact(model_path=None):
    require_torch()
    path = Path(model_path or os.getenv("PRICE_MODEL_PATH", ""))
    if not path.is_file():
        raise ValueError("未配置有效的 PRICE_MODEL_PATH；请先离线训练 MQ-RNN 电价模型")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    m = artifact["metadata"]
    if (m.get("format") != FORMAT or m.get("feature_names") != FEATURE_NAMES
            or m.get("quantiles") != QUANTILES or m.get("target") != TARGET):
        raise ValueError("unsupported MQ-RNN artifact format, features, quantiles or target")
    if m["resolution_minutes"] not in (15, 60) or m["context_length"] < 2:
        raise ValueError("invalid MQ-RNN resolution or context length")
    if len(m["series_ids"]) != m["architecture"]["num_series"]:
        raise ValueError("invalid MQ-RNN series metadata")
    if any(not math.isfinite(s) or s <= 0 for s in m["scales"].values()):
        raise ValueError("invalid MQ-RNN scale")
    model = PriceMQRNN(**m["architecture"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, m, hashlib.sha256(path.read_bytes()).hexdigest()


def predict_price_mqrnn(history, start: datetime, periods: int, issued_at: datetime, *,
                        timezone_name: str, series_id: str, transmission_and_fund=0,
                        include_uncertainty=True, model_path=None) -> TargetForecast:
    model, m, digest = load_artifact(model_path)
    if m["unit"] != "CNY/kWh":
        raise ValueError("在线电价模型必须使用 CNY/kWh；禁止把外币/MWh 权重当作人民币价格")
    if m["timezone"] != timezone_name:
        raise ValueError("price model timezone does not match site timezone")
    if series_id not in m["series_ids"]:
        raise ValueError(f"price series_id {series_id!r} is absent from model metadata")
    if datetime.fromisoformat(m["selection_end"]) >= issued_at:
        raise ValueError("price model training/validation cutoff must precede issued_at")
    if not 0 < periods <= 288 or start < issued_at:
        raise ValueError("MQ-RNN price forecast requires a future start and at most 72 hours")
    resolution = m["resolution_minutes"]
    if start.minute % resolution or start.second or start.microsecond:
        raise ValueError("forecast start must align with model native resolution")
    required = m["context_length"] + 7 * 1440 // resolution
    history = [(ts.astimezone(start.tzinfo), v) for ts, v in history if ts < issued_at and ts < start][-required:]
    if len(history) < required:
        raise ValueError(f"MQ-RNN price needs at least {required} native history points")
    validate_series(history, resolution)
    first = history[-1][0] + timedelta(minutes=resolution)
    if (issued_at - first).total_seconds() > resolution * 60:
        raise ValueError("price history is stale at issued_at")
    timestamps = [start + timedelta(minutes=15 * i) for i in range(periods)]
    indices = [int((ts - first).total_seconds() // (resolution * 60)) for ts in timestamps]
    if indices[0] < 0 or indices[-1] >= model.horizon:
        raise ValueError("MQ-RNN trained horizon is too short for requested start/bridge; train a 4-day horizon")
    q = direct_quantiles(model, history, resolution=resolution, context=m["context_length"],
                         scale=m["scales"][series_id], series_index=m["series_ids"].index(series_id))
    curves = [[round(float(q[i, j]) + transmission_and_fund, 6) for i in indices] for j in range(3)]
    note = ("MQ-RNN：LSTM + 全局/局部MLP，直接预测p10/p50/p90，点预测为p50；"
            "允许负电价；电能量价仅加一次输配及基金附加；区间尚未经本站覆盖率校准")
    if resolution == 60:
        note += "；原生小时模型，按小时结算价展开为15分钟接口点"
    return TargetForecast(
        target="price", unit="CNY/kWh", model=ForecastModel(name="price_mqrnn", version="v1"),
        points=[ForecastPoint(ts=ts, value=v) for ts, v in zip(timestamps, curves[1])],
        p10=curves[0] if include_uncertainty else None, p90=curves[2] if include_uncertainty else None,
        components={"energy_price_p50": [round(v - transmission_and_fund, 6) for v in curves[1]],
                    "transmission_and_fund": [transmission_and_fund] * periods},
        provenance=ForecastProvenance(provider="local_price_history", checkpoint_sha256=digest,
            training_status="offline_trained", window=f"context={history[0][0].isoformat()}..{history[-1][0].isoformat()}; "
            f"native={resolution}min; direct_horizon={model.horizon}; selection_end={m['selection_end']}; series={series_id}"),
        notes=note)
