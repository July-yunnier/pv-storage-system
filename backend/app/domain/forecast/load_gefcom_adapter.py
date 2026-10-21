"""Hourly GEFCom adapter of the original MES-attention load network.

This is not the original 96-slot factory contract. MES is unknown and the
unlabelled workshop head is frozen; no synthetic MES or workshop labels exist.
"""
from __future__ import annotations

import numpy as np
import torch

from app.domain.forecast.load_advanced import EliteEnergyTaskNet


class GEFComEliteEnergyTaskNet(EliteEnergyTaskNet):
    """Reuse the original modules, replacing only the fixed-slot forward contract."""

    def __init__(self, history_feature_dim: int, target_feature_dim: int, hidden_dim: int = 128):
        super().__init__(history_feature_dim, target_feature_dim, num_mes_plans=1,
                         mes_embed_dim=8, hidden_dim=hidden_dim)
        # Unknown MES has no learned production semantics. No auxiliary targets.
        with torch.no_grad():
            self.mes_embedding.weight.zero_()
        self.mes_embedding.weight.requires_grad_(False)
        for parameter in self.workshop_load_head.parameters():
            parameter.requires_grad_(False)

    def forward(self, history_features, history_mes_labels, target_features, target_mes_labels):
        self._check_mes(history_mes_labels)
        self._check_mes(target_mes_labels)
        if history_features.shape[:2] != history_mes_labels.shape:
            raise ValueError("history feature/MES dimensions differ")
        if target_features.shape[:2] != target_mes_labels.shape or target_features.shape[1] < 1:
            raise ValueError("target feature/MES dimensions differ")
        encoded, _ = self.history_lstm(torch.cat(
            [history_features, self.mes_embedding(history_mes_labels)], dim=-1))
        context, weights = self.attention(encoded)
        context = context.unsqueeze(1).expand(-1, target_features.shape[1], -1)
        decoded = self.target_decoder(torch.cat(
            [context, target_features, self.mes_embedding(target_mes_labels)], dim=-1))
        return (self.total_load_head(decoded).squeeze(-1),
                self.workshop_load_head(decoded).squeeze(-1), weights)


def adapter_stats(data):
    ids = data.indices("train")
    if not len(ids):
        raise ValueError("empty training split")
    history = np.concatenate([data.seq[ids, :, None], data.exog[ids],
                              data.calendar[ids, :data.history_size]], axis=-1)
    target = data.calendar[ids, data.history_size:]
    labels = data.y[ids, -1]
    return {
        "fit_split": "train",
        "history_mean": history.mean(axis=(0, 1)).tolist(),
        "history_std": np.maximum(history.std(axis=(0, 1)), 1e-6).tolist(),
        "target_mean": target.mean(axis=(0, 1)).tolist(),
        "target_std": np.maximum(target.std(axis=(0, 1)), 1e-6).tolist(),
        "load_mean": float(labels.mean()), "load_std": max(float(labels.std()), 1e-6),
    }


def adapter_batch(data, stats, ids):
    history = np.concatenate([data.seq[ids, :, None], data.exog[ids],
                              data.calendar[ids, :data.history_size]], axis=-1)
    target = data.calendar[ids, data.history_size:]
    history = (history - np.asarray(stats["history_mean"])) / np.asarray(stats["history_std"])
    target = (target - np.asarray(stats["target_mean"])) / np.asarray(stats["target_std"])
    return (torch.as_tensor(history, dtype=torch.float32),
            torch.zeros(history.shape[:2], dtype=torch.long),
            torch.as_tensor(target, dtype=torch.float32),
            torch.zeros(target.shape[:2], dtype=torch.long))


def adapter_predict(model, data, stats, ids, batch_size=512):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = adapter_batch(data, stats, ids[start:start + batch_size])
            values = model(*batch)[0][:, -1].numpy()
            chunks.append(values * stats["load_std"] + stats["load_mean"])
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite adapter prediction")
    return result
