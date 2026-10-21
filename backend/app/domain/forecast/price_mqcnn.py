"""MQ-CNN：多步分位数卷积预测器（Wen et al., NIPS 2017, arXiv:1711.11053）。

原论文的三个要素，这里逐条落地：

1. **序列编码器**（temporal nature）：因果卷积吃过去 168 小时的价格与协变量，
   等价于论文里的 CNN 版本 MQ-CNN（论文指出电价任务上 CNN 优于 RNN）。
2. **分位数回归**（nonparametric）：直接优化 pinball loss，一次输出 P10/P50/P90；
   用 median ± softplus(·) 参数化保证分位数单调，不做事后排序。
3. **直接多步**（direct multi-horizon）：一次前向输出 1..H 全部步长，
   不递归，避免误差随步长累积——这是电价这种高噪声序列的关键。

与论文的差异（须在报告中披露）：未使用 embedding 处理跨序列的静态协变量
（本项目是单序列），未做 forking-sequences（直接多步已覆盖其收益）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.forecast.price_calendar import is_holiday, is_workday
from app.domain.forecast.price_metrics import pinball
from app.domain.forecast.price_model import PriceForecastOutput, QUANTILES

ENC_CHANNELS_DEFAULT = 4


def _calendar_channels(ts: pd.DatetimeIndex) -> np.ndarray:
    hour = np.asarray([t.hour for t in ts], dtype="float64")
    dow = np.asarray([t.dayofweek for t in ts], dtype="float64")
    holiday = np.asarray([1.0 if is_holiday(t.date()) else 0.0 for t in ts])
    workday = np.asarray([1.0 if is_workday(t.date()) else 0.0 for t in ts])
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            holiday,
            workday,
        ]
    )


def build_sequences(
    ts: pd.DatetimeIndex,
    price: np.ndarray,
    load: np.ndarray | None,
    lookback: int = 168,
    horizon: int = 24,
    origins: np.ndarray | None = None,
    min_origin: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """构造 (B, lookback, C) 编码器输入与 (B, horizon, F) 未来协变量。

    Returns:
        enc, fut, y, origin_idx, target_start_ts
    """
    n = len(ts)
    min_origin = lookback - 1 if min_origin is None else min_origin
    if origins is None:
        origins = np.arange(n)
    origins = np.asarray(origins, dtype=int)
    origins = origins[(origins >= min_origin) & (origins + horizon < n)]

    price_filled = pd.Series(price, dtype="float64").ffill(limit=2).to_numpy()
    cal_all = _calendar_channels(ts)
    load_filled = None
    if load is not None:
        load_filled = pd.Series(load, dtype="float64").ffill(limit=2).bfill(limit=2).to_numpy()

    enc_list, fut_list, y_list, keep, starts = [], [], [], [], []
    for origin in origins:
        window = price_filled[origin - lookback + 1 : origin + 1]
        if not np.isfinite(window).all():
            continue
        targets = price[origin + 1 : origin + 1 + horizon]
        if not np.isfinite(targets).all():
            continue

        cal_window = cal_all[origin - lookback + 1 : origin + 1]
        channels = [window, cal_window]
        if load_filled is not None:
            load_window = load_filled[origin - lookback + 1 : origin + 1]
            channels.insert(1, load_window)

        fut_idx = np.arange(origin + 1, origin + 1 + horizon)
        fut_parts = [cal_all[fut_idx]]
        if load_filled is not None:
            fut_parts.insert(0, load_filled[fut_idx].reshape(-1, 1))

        enc_list.append(np.column_stack(channels))
        fut_list.append(np.column_stack(fut_parts))
        y_list.append(targets)
        keep.append(origin)
        starts.append(ts[origin + 1])

    if not enc_list:
        raise ValueError("no usable sequences")
    return (
        np.stack(enc_list).astype("float32"),
        np.stack(fut_list).astype("float32"),
        np.stack(y_list).astype("float64"),
        np.asarray(keep),
        pd.DatetimeIndex(starts),
    )


class MqCnn:
    """单序列 MQ-CNN。训练/推理都只依赖 torch，不碰业务代码。"""

    name = "price_mqcnn"
    version = "v1"

    def __init__(
        self,
        enc_channels: int,
        future_features: int,
        horizon: int = 24,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        self.horizon = horizon
        self.price_mean = 0.0
        self.price_std = 1.0
        self.load_mean = 0.0
        self.load_std = 1.0
        self.has_load = False
        self._build(enc_channels, future_features, horizon, hidden, dropout)

    def _build(
        self,
        enc_channels: int,
        future_features: int,
        horizon: int = 24,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.horizon = horizon
                self.conv1 = nn.Conv1d(enc_channels, hidden, kernel_size=7, padding=0)
                self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=0)
                self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=0)
                self.act = nn.GELU()
                self.drop = nn.Dropout(dropout)
                self.head = nn.Sequential(
                    nn.Linear(hidden + future_features, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                )
                self.out = nn.Linear(hidden, 3)

            def forward(self, enc, fut):
                # enc: (B, L, C) -> (B, C, L)；因果卷积靠"只保留末尾"实现
                x = enc.transpose(1, 2)
                x = self.act(self.conv1(x))
                x = self.act(self.conv2(x))
                x = self.drop(self.act(self.conv3(x)))
                context = x[:, :, -1]
                repeated = context.unsqueeze(1).expand(-1, fut.shape[1], -1)
                merged = self.act(self.head(torch.cat([repeated, fut], dim=-1)))
                raw = self.out(merged)
                median = raw[..., 1]
                lower = median - nn.functional.softplus(raw[..., 0])
                upper = median + nn.functional.softplus(raw[..., 2])
                return torch.stack([lower, median, upper], dim=-1)

        self.net = Net()
        self.horizon = horizon

    # ---- 训练 ----
    def fit(
        self,
        ts: pd.DatetimeIndex,
        price: np.ndarray,
        load: np.ndarray | None,
        horizon: int = 24,
        lookback: int = 168,
        max_epochs: int = 40,
        batch_size: int = 128,
        learning_rate: float = 2e-3,
        patience: int = 6,
        val_days: int = 60,
        seed: int = 42,
        origin_stride: int = 1,
        verbose: bool = False,
    ) -> "MqCnn":
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(2)

        n = len(ts)
        split = n - val_days * 24
        train_origins = np.arange(0, split, max(1, origin_stride))
        val_origins = np.arange(split, n, max(1, origin_stride))

        enc_tr, fut_tr, y_tr, _, _ = build_sequences(ts, price, load, lookback, horizon, train_origins)
        enc_va, fut_va, y_va, _, _ = build_sequences(ts, price, load, lookback, horizon, val_origins)

        self.price_mean = float(y_tr.mean())
        self.price_std = float(max(y_tr.std(), 1e-6))
        self.has_load = load is not None
        if self.has_load:
            finite_load = load[np.isfinite(load)]
            self.load_mean = float(finite_load.mean())
            self.load_std = float(max(finite_load.std(), 1e-6))

        enc_tr, fut_tr = self._scale(enc_tr, fut_tr)
        enc_va, fut_va = self._scale(enc_va, fut_va)
        y_tr_n = (y_tr - self.price_mean) / self.price_std
        y_va_n = (y_va - self.price_mean) / self.price_std

        self.horizon = horizon
        self._build(
            enc_channels=enc_tr.shape[-1],
            future_features=fut_tr.shape[-1],
            horizon=horizon,
        )
        torch.manual_seed(seed)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=1e-4)
        enc_tr_t = torch.as_tensor(enc_tr)
        fut_tr_t = torch.as_tensor(fut_tr)
        y_tr_t = torch.as_tensor(y_tr_n, dtype=torch.float32)
        enc_va_t = torch.as_tensor(enc_va)
        fut_va_t = torch.as_tensor(fut_va)

        levels = torch.as_tensor(QUANTILES, dtype=torch.float32).reshape(1, 1, -1)
        best, wait, best_state = float("inf"), 0, None
        for epoch in range(1, max_epochs + 1):
            self.net.train()
            order = torch.randperm(len(enc_tr_t))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                pred = self.net(enc_tr_t[idx], fut_tr_t[idx])
                error = y_tr_t[idx].unsqueeze(-1) - pred
                loss = torch.maximum(levels * error, (levels - 1.0) * error).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(enc_va_t, fut_va_t).numpy() * self.price_std + self.price_mean
            val_loss = pinball(y_va.reshape(-1), val_pred.reshape(-1, 3))
            if verbose and epoch % 5 == 0:
                print(f"    mqcnn epoch={epoch} val_pinball={val_loss:.4f}", flush=True)
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_pinball = float(best)
        return self

    def _scale(self, enc: np.ndarray, fut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        enc = enc.copy()
        fut = fut.copy()
        enc[:, :, 0] = (enc[:, :, 0] - self.price_mean) / self.price_std
        if self.has_load and enc.shape[-1] > 1:
            enc[:, :, 1] = (enc[:, :, 1] - self.load_mean) / self.load_std
            fut[:, :, 0] = (fut[:, :, 0] - self.load_mean) / self.load_std
        return enc, fut

    # ---- 推理 ----
    def predict(self, ts: pd.DatetimeIndex, price: np.ndarray, load: np.ndarray | None, origin: int) -> PriceForecastOutput:
        import torch

        enc, fut, _, _, starts = build_sequences(
            ts, price, load, lookback=168, horizon=self.horizon, origins=np.asarray([origin])
        )
        enc, fut = self._scale(enc, fut)
        self.net.eval()
        with torch.no_grad():
            pred = self.net(torch.as_tensor(enc), torch.as_tensor(fut)).numpy()[0]
        pred = pred * self.price_std + self.price_mean
        target_ts = pd.date_range(ts[origin] + pd.Timedelta("1h"), periods=self.horizon, freq="h")
        return PriceForecastOutput(
            ts=target_ts,
            p10=pred[:, 0],
            p50=pred[:, 1],
            p90=pred[:, 2],
            method=self.name,
            version=self.version,
            notes="MQ-CNN（arXiv:1711.11053）：因果卷积编码 168h + 直接输出 24 步分位数",
        )
