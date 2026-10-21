"""Native probabilistic sequence models for the GEFCom pre-experiment.

The heads emit P10/P50/P90 directly and optimize pinball loss.  This module
does not import or train the frozen photovoltaic model.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

QUANTILES = (0.10, 0.50, 0.90)


def pinball_loss(pred, target):
    import torch
    q = torch.as_tensor(QUANTILES, dtype=pred.dtype, device=pred.device).reshape(1, 1, -1)
    error = target.unsqueeze(-1) - pred
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def pinball_numpy(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    if not len(y) or len(y) != len(pred) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("pinball requires nonempty matching finite arrays")
    q = np.asarray(QUANTILES).reshape(1, -1)
    e = y - pred
    return float(np.maximum(q * e, (q - 1.0) * e).mean())


def interval_coverage(y, p10, p90) -> float:
    return float(np.mean((np.asarray(y) >= np.asarray(p10)) & (np.asarray(y) <= np.asarray(p90))))


def rolling_stats(seq: np.ndarray, window: int = 24) -> np.ndarray:
    """Causal rolling mean/std for each historical sequence."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or window < 1:
        raise ValueError("rolling stats require a matrix and positive window")
    c = np.pad(np.cumsum(seq, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
    c2 = np.pad(np.cumsum(seq.astype(np.float64)**2, axis=1), ((0, 0), (1, 0)))
    end = np.arange(1, seq.shape[1] + 1)
    start = np.maximum(0, end - window)
    n = end - start
    mean = (c[:, end] - c[:, start]) / n
    var = np.maximum((c2[:, end] - c2[:, start]) / n - mean**2, 0)
    return np.stack([mean, np.sqrt(var)], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class QuantileSettings:
    seed: int = 42
    max_epochs: int = 120
    patience: int = 10
    min_delta: float = 1e-3
    learning_rate: float = 1e-3
    batch_size: int = 256
    hidden: int = 64
    dropout: float = 0.10
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    noise_std: float = 0.0
    use_rolling: bool = False

    def __post_init__(self):
        if self.max_epochs < 10 or self.patience < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("full training requires >=10 epochs budget and valid settings")


class QuantileNet:
    def __new__(cls, kind: str, input_dim: int, hidden: int, horizon: int, dropout: float):
        import torch
        import torch.nn as nn

        class Chomp(nn.Module):
            def __init__(self, size):
                super().__init__(); self.size = size
            def forward(self, x):
                return x[:, :, :-self.size] if self.size else x

        class TCNBlock(nn.Module):
            def __init__(self, cin, cout, dilation):
                super().__init__(); pad = 2 * dilation
                self.net = nn.Sequential(nn.Conv1d(cin, cout, 3, padding=pad, dilation=dilation), Chomp(pad), nn.GELU(), nn.Dropout(dropout),
                                         nn.Conv1d(cout, cout, 3, padding=pad, dilation=dilation), Chomp(pad), nn.GELU(), nn.Dropout(dropout))
                self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
            def forward(self, x): return self.net(x) + self.skip(x)

        class Net(nn.Module):
            def __init__(self):
                super().__init__(); self.kind = kind; self.horizon = horizon
                if kind == "LSTM":
                    self.encoder = nn.LSTM(input_dim, hidden, num_layers=2, dropout=dropout, batch_first=True)
                    head_input = hidden
                elif kind == "TCN":
                    self.encoder = nn.Sequential(TCNBlock(input_dim, hidden, 1), TCNBlock(hidden, hidden, 2), TCNBlock(hidden, hidden, 4))
                    head_input = hidden
                else: raise ValueError(kind)
                self.head = nn.Sequential(nn.LayerNorm(head_input), nn.Linear(head_input, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, horizon * 3))
            def forward(self, batch):
                values, hist_exog, calendar, rolling = batch
                x = torch.cat([values, hist_exog, calendar[:, :values.shape[1]], rolling], dim=-1)
                if self.kind == "LSTM": context = self.encoder(x)[1][0][-1]
                else: context = self.encoder(x.transpose(1, 2))[:, :, -1]
                raw = self.head(context).reshape(len(values), horizon, 3)
                # Native ordered quantile parameterization, not post-hoc sorting.
                median = raw[..., 1]
                return torch.stack([median - torch.nn.functional.softplus(raw[..., 0]),
                                    median,
                                    median + torch.nn.functional.softplus(raw[..., 2])], dim=-1)
        return Net()


def set_seed(seed: int = 42):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


def make_stats(data, train_ids, use_rolling=True):
    seq = data.seq[train_ids]; exog = data.exog[train_ids]; roll = rolling_stats(seq)
    return {"use_rolling": use_rolling, "fit_split": "train", "target_mean": float(data.rows.iloc[train_ids].origin_value.mean()),
            "target_std": max(float(data.rows.iloc[train_ids].origin_value.std(ddof=0)), 1e-6),
            "exog_mean": exog.reshape(-1, exog.shape[-1]).mean(0).tolist() if exog.shape[-1] else [],
            "exog_std": np.maximum(exog.reshape(-1, exog.shape[-1]).std(0), 1e-6).tolist() if exog.shape[-1] else [],
            "rolling_mean": roll.reshape(-1, 2).mean(0).tolist(),
            "rolling_std": np.maximum(roll.reshape(-1, 2).std(0), 1e-6).tolist()}


def batch_from(data, stats, ids):
    import torch
    seq, exog, cal = data.seq[ids], data.exog[ids], data.calendar[ids]
    roll = rolling_stats(seq) if stats.get("use_rolling", True) else np.empty((*seq.shape, 0), dtype=np.float32)
    seq = (seq - stats["target_mean"]) / stats["target_std"]
    if exog.shape[-1]: exog = (exog - np.asarray(stats["exog_mean"])) / np.asarray(stats["exog_std"])
    if roll.shape[-1]: roll = (roll - np.asarray(stats["rolling_mean"])) / np.asarray(stats["rolling_std"])
    return tuple(torch.as_tensor(v, dtype=torch.float32) for v in (seq[..., None], exog, cal, roll))


def predict(model, data, stats, ids, batch_size):
    import torch
    model.eval(); chunks = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            pred = model(batch_from(data, stats, ids[i:i + batch_size])).cpu().numpy()
            chunks.append(pred)
    result = np.concatenate(chunks) * stats["target_std"] + stats["target_mean"]
    if not np.isfinite(result).all(): raise ValueError("nonfinite quantile predictions")
    return result


def fit_quantile(name: str, data, settings: QuantileSettings, exp: Path, loss_mode: str = "pinball"):
    import torch
    set_seed(settings.seed)
    train_ids, val_ids = data.indices("train"), data.indices("validation")
    if not len(train_ids) or not len(val_ids): raise ValueError("empty train/validation")
    if loss_mode != "pinball": raise ValueError("native head must train all quantiles with pinball")
    exp.mkdir(parents=True, exist_ok=True)
    stats = make_stats(data, train_ids, settings.use_rolling)
    input_dim = 1 + data.exog.shape[-1] + data.calendar.shape[-1] + (2 if settings.use_rolling else 0)
    model = QuantileNet(name, input_dim, settings.hidden, data.horizon, settings.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    best, wait, best_state, log, best_epoch = float("inf"), 0, None, [], 0
    stop_reason = "max_budget_not_converged"
    target = data.y
    # Cache only train/validation inputs. Test is never visited during fitting.
    train_batch = batch_from(data, stats, train_ids)
    val_batch = batch_from(data, stats, val_ids)
    target_tensor = torch.as_tensor((target[train_ids] - stats["target_mean"]) / stats["target_std"], dtype=torch.float32)
    (exp / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
    for epoch in range(1, settings.max_epochs + 1):
        model.train(); total = 0.0
        for start in range(0, len(train_ids), settings.batch_size):
            ids = train_ids[start:start + settings.batch_size]; optimizer.zero_grad()
            batch = [value[start:start + settings.batch_size] for value in train_batch]
            if settings.noise_std: batch[0] = batch[0] + torch.randn_like(batch[0]) * settings.noise_std
            pred = model(tuple(batch)); y = target_tensor[start:start + settings.batch_size]
            # Match the selected forecast horizon in training, validation, metrics.
            loss = pinball_loss(pred[:, -1:, :], y[:, -1:])
            if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip); optimizer.step(); total += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            val_pred = np.concatenate([model(tuple(v[i:i+settings.batch_size] for v in val_batch)).numpy()
                                       for i in range(0, len(val_ids), settings.batch_size)])
        val_pred = val_pred * stats["target_std"] + stats["target_mean"]
        val_loss = pinball_numpy(target[val_ids, -1], val_pred[:, -1, :])
        log.append({"epoch": epoch, "train_loss": total / len(train_ids), "validation_loss": val_loss})
        if val_loss < best - settings.min_delta:
            best, wait, best_epoch = val_loss, 0, epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, exp / "model.pt")
        else:
            wait += 1
        (exp / "training_log.json").write_text(json.dumps(log, indent=2))
        if epoch % 10 == 0:
            print(f"  {name} epoch={epoch} val_pinball={val_loss:.6f} best={best:.6f} stale={wait}", flush=True)
        if wait >= settings.patience:
            stop_reason = "validation_early_stop"
            break
    if best_state is None: raise RuntimeError("no best checkpoint")
    model.load_state_dict(best_state); exp.mkdir(parents=True, exist_ok=True); torch.save(best_state, exp / "model.pt")
    (exp / "training_log.json").write_text(json.dumps(log, indent=2))
    meta = {"kind": "native_quantile", "name": name, "loss_mode": loss_mode, "settings": asdict(settings),
            "best_validation_loss": best, "best_epoch": best_epoch, "epochs_completed": len(log),
            "stop_reason": stop_reason, "validation_objective": "final_horizon_pinball",
            "ordered_quantiles": True, "input_dim": input_dim, "normalization": stats}
    (exp / "artifact.json").write_text(json.dumps(meta, indent=2))
    return model, stats, meta
