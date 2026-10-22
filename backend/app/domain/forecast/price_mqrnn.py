"""MQ-RNN：多步分位数循环预测器（Wen et al., arXiv:1711.11053 的 RNN 变体）。

移植自 July 在 product/ 旧树的论文级复现（feat/product-mvp-impl 分支
`price_mqrnn.py` + `train_price_mqrnn.py`），按主线 MqCnn 的接口约定落地，
供 GEFCom 15 任务日同协议对决与后续站点重训使用。

论文要素的落地口径（与她的实现逐项对齐）：

1. **LSTM 序列编码器**：每个位置输入 [当前价、日滞后、周滞后]（按训练段尺度
   缩放）+ 7 个日历特征 + 序列嵌入；context=168（一周）为一块。
2. **forking-sequences**：块内每个位置共享同一全局/局部 MLP 解码器，
   一次反向传播聚合全部 fork 的误差；块间重置状态（截断 BPTT）。
3. **直接多步分位数**：一次输出 1..H 步的 P10/P50/P90，pinball 损失直接优化；
   median ± softplus(·) 参数化保证分位数单调（她的工程扩展，非论文原始头）。
4. **边界遮罩**：块尾不足 H 步的标签不参与损失，训练数据由调用方截断在
   origin 之前——本模块不存在看到未来的路径。

与她的 product/ 版差异（须披露）：只保留单序列离线训练/推理路径
（去掉了 TargetForecast 在线封装与工件校验，那部分待本站重训后按注册表契约
另接）；calendar-only，不使用负荷协变量（与她的 GEFCom 适配口径一致）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.domain.forecast.price_metrics import pinball
from app.domain.forecast.price_model import PriceForecastOutput, QUANTILES

FEATURE_NAMES = ["day_sin", "day_cos", "week_sin", "week_cos", "year_sin", "year_cos", "weekend"]
LOOKBACK = 168  # 编码器块长：一周（小时粒度）
WEEK_LAG = 168


def calendar_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """7 个日历特征（day/week/year 正余弦 + 周末），与 product/ 版逐式对齐。"""
    day = (ts.hour * 60 + ts.minute) / 1440.0
    week = (ts.dayofweek + day) / 7.0
    year = (ts.dayofyear - 1 + day) / 365.25
    return np.column_stack(
        [
            np.sin(2 * np.pi * day),
            np.cos(2 * np.pi * day),
            np.sin(2 * np.pi * week),
            np.cos(2 * np.pi * week),
            np.sin(2 * np.pi * year),
            np.cos(2 * np.pi * year),
            (ts.dayofweek >= 5).astype("float64"),
        ]
    )


def observed_features(ts: pd.DatetimeIndex, price_scaled: np.ndarray) -> np.ndarray:
    """每个已观测位置的编码器输入：[v, v-24h, v-168h, *日历]（前 168 点无周滞后，丢弃）。"""
    n = len(price_scaled)
    cal = calendar_features(ts)
    out = np.zeros((n, 3 + len(FEATURE_NAMES)), dtype="float64")
    out[:, 0] = price_scaled
    out[WEEK_LAG:, 1] = price_scaled[WEEK_LAG - 24 : n - 24]
    out[WEEK_LAG:, 2] = price_scaled[: n - WEEK_LAG]
    out[:, 3:] = cal
    return out[WEEK_LAG:]


class MqRnn:
    """单序列 MQ-RNN。fit/predict 接口与 MqCnn 对齐；只依赖 torch，不碰业务代码。"""

    name = "price_mqrnn"
    version = "v1"

    def __init__(
        self,
        horizon: int = 24,
        hidden_size: int = 30,
        context_size: int = 8,
        decoder_size: int = 64,
        embedding_size: int = 4,
    ) -> None:
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.decoder_size = decoder_size
        self.embedding_size = embedding_size
        self.scale = 1.0
        self.best_val_pinball = math.inf
        self._build()

    def _build(self) -> None:
        import torch
        from torch import nn
        from torch.nn import functional as F

        horizon = self.horizon
        hidden = self.hidden_size
        context_size = self.context_size
        decoder_size = self.decoder_size
        embedding_size = self.embedding_size

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(1, embedding_size)
                self.encoder = nn.LSTM(
                    3 + len(FEATURE_NAMES) + embedding_size, hidden, batch_first=True
                )
                self.global_mlp = nn.Sequential(
                    nn.Linear(hidden + horizon * len(FEATURE_NAMES), decoder_size),
                    nn.ReLU(),
                    nn.Linear(decoder_size, (horizon + 1) * context_size),
                )
                self.local_mlp = nn.Sequential(
                    nn.Linear(2 * context_size + len(FEATURE_NAMES), decoder_size),
                    nn.ReLU(),
                    nn.Linear(decoder_size, len(QUANTILES)),
                )

            def encode(self, observed):
                emb = self.embedding(torch.zeros(observed.shape[0], dtype=torch.long))
                emb = emb[:, None].expand(-1, observed.shape[1], -1)
                return self.encoder(torch.cat([observed, emb], dim=-1))[0]

            def decode(self, states, future):
                contexts = self.global_mlp(torch.cat([states, future.flatten(-2)], dim=-1))
                contexts = contexts.reshape(*states.shape[:-1], horizon + 1, context_size)
                specific = contexts[..., :-1, :]
                shared = contexts[..., -1:, :].expand_as(specific)
                raw = self.local_mlp(torch.cat([specific, shared, future], dim=-1))
                median = raw[..., 1]
                return torch.stack(
                    [median - F.softplus(raw[..., 0]), median, median + F.softplus(raw[..., 2])],
                    dim=-1,
                )

            def forward(self, observed, future):
                return self.decode(self.encode(observed), future)

        self.net = Net()

    # ---- 数据构造 ----
    def _blocks(
        self,
        ts: pd.DatetimeIndex,
        price: np.ndarray,
        block_stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """forking-sequences 块：enc (B,168,14)、fut (B,168,H,7)、y (B,168,H)、mask 同形。

        训练截止由调用方保证（传入的 ts/price 不含 origin 之后的数据）；
        块尾越界标签以 mask 剔除，与 product/ 版 boundary_policy 一致。
        """
        if not np.isfinite(price).all():
            raise ValueError("price series contains non-finite values")
        price_scaled = price / self.scale
        feats = observed_features(ts, price_scaled)  # 对齐到 ts[WEEK_LAG:]
        y_all = price_scaled[WEEK_LAG:]
        cal_all = calendar_features(ts)
        n = len(y_all)
        horizon = self.horizon

        enc_list, fut_list, y_list, mask_list = [], [], [], []
        for start in range(0, n - 1, LOOKBACK * max(1, block_stride)):
            positions = np.arange(start, start + LOOKBACK)
            valid_pos = positions < n - 1
            safe_pos = np.minimum(positions, n - 1)
            enc_list.append(feats[safe_pos])
            target_idx = positions[:, None] + np.arange(1, horizon + 1)[None, :]
            mask = valid_pos[:, None] & (target_idx < n)
            y_list.append(y_all[np.minimum(target_idx, n - 1)])
            # 日历特征可越界外推（未来已知量），索引按 ts 绝对位置取
            fut_ts_idx = np.minimum(target_idx + WEEK_LAG, len(ts) - 1)
            fut_list.append(cal_all[fut_ts_idx])
            mask_list.append(mask)
        if not enc_list:
            raise ValueError("not enough history for MQ-RNN training blocks")
        return (
            np.stack(enc_list).astype("float32"),
            np.stack(fut_list).astype("float32"),
            np.stack(y_list).astype("float32"),
            np.stack(mask_list),
        )

    def _pinball_masked(self, pred, target, mask):
        import torch

        levels = pred.new_tensor(QUANTILES)
        error = target.unsqueeze(-1) - pred
        losses = torch.maximum(levels * error, (levels - 1.0) * error)
        losses = torch.where(mask.unsqueeze(-1), losses, torch.zeros_like(losses))
        return losses.sum() / (mask.sum().clamp_min(1) * len(QUANTILES))

    def _direct_quantiles(self, ts: pd.DatetimeIndex, price: np.ndarray, origin: int) -> np.ndarray:
        """从 origin（最后一个已知点索引）一次前向输出 1..H 步分位数（已还原尺度）。"""
        import torch

        need = LOOKBACK + WEEK_LAG
        if origin + 1 < need:
            raise ValueError(f"MQ-RNN 需要 origin 前至少 {need} 个小时历史")
        window_ts = ts[origin - need + 1 : origin + 1]
        window_price = price[origin - need + 1 : origin + 1] / self.scale
        if not np.isfinite(window_price).all():
            raise ValueError("MQ-RNN 历史窗口含非有限值")
        enc = torch.as_tensor(observed_features(window_ts, window_price)[None], dtype=torch.float32)
        fut_ts = pd.date_range(ts[origin] + pd.Timedelta("1h"), periods=self.horizon, freq="h")
        fut = torch.as_tensor(calendar_features(fut_ts)[None], dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            states = self.net.encode(enc)
            q = self.net.decode(states[:, -1], fut)[0].numpy()
        q = q * self.scale
        if not np.isfinite(q).all():
            raise RuntimeError("MQ-RNN 输出了非有限分位数")
        return q

    # ---- 训练 ----
    def fit(
        self,
        ts: pd.DatetimeIndex,
        price: np.ndarray,
        load: np.ndarray | None = None,
        horizon: int = 24,
        max_epochs: int = 20,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        val_days: int = 45,
        seed: int = 42,
        block_stride: int = 1,
        verbose: bool = False,
    ) -> "MqRnn":
        """在传入序列上训练（调用方必须已把序列截断在 origin 之前）。

        尺度 `1 + mean(abs(price))` 只拟合传入的训练段，推理时固定。
        每个 epoch 用末尾 val_days 天的每日原点做直接预测的 scaled pinball，
        保留最优轮次（对应 product/ 版"验证集选轮"）。
        """
        import torch

        del load  # calendar-only 适配口径：不使用负荷协变量（与 product/ 版一致）
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(2)

        ts = pd.DatetimeIndex(ts)
        price = np.asarray(price, dtype="float64")
        if len(ts) != len(price):
            raise ValueError("ts/price length mismatch")
        if len(ts) < LOOKBACK + WEEK_LAG + val_days * 24 + horizon:
            raise ValueError("history too short for MQ-RNN fit")
        self.horizon = horizon
        self.scale = float(1.0 + np.abs(price).mean())
        self._build()
        torch.manual_seed(seed)

        enc, fut, y, mask = self._blocks(ts, price, block_stride)
        enc_t = torch.as_tensor(enc)
        fut_t = torch.as_tensor(fut)
        y_t = torch.as_tensor(y)
        mask_t = torch.as_tensor(mask)

        val_origins = np.arange(len(ts) - val_days * 24, len(ts) - horizon, 24)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        best, best_state = math.inf, None
        for epoch in range(1, max_epochs + 1):
            self.net.train()
            order = torch.randperm(len(enc_t))
            total, count = 0.0, 0
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                pred = self.net(enc_t[idx], fut_t[idx])
                loss = self._pinball_masked(pred, y_t[idx], mask_t[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError("MQ-RNN 训练损失非有限")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()
                n = int(mask_t[idx].sum())
                total += float(loss.detach()) * n
                count += n

            val_loss = float(
                np.mean(
                    [
                        pinball(
                            price[o + 1 : o + 1 + horizon] / self.scale,
                            self._direct_quantiles(ts, price, o) / self.scale,
                        )
                        for o in val_origins
                    ]
                )
            )
            if verbose:
                print(
                    f"    mqrnn epoch={epoch} train_pinball={total / max(count, 1):.4f}"
                    f" val_pinball={val_loss:.4f}",
                    flush=True,
                )
            if val_loss < best - 1e-5:
                best, best_state = val_loss, {
                    k: v.detach().clone() for k, v in self.net.state_dict().items()
                }
        if best_state is None:
            raise RuntimeError("MQ-RNN 没有有限的验证检查点")
        self.net.load_state_dict(best_state)
        self.best_val_pinball = float(best)
        return self

    # ---- 推理 ----
    def predict(
        self,
        ts: pd.DatetimeIndex,
        price: np.ndarray,
        load: np.ndarray | None,
        origin: int,
    ) -> PriceForecastOutput:
        del load
        ts = pd.DatetimeIndex(ts)
        q = self._direct_quantiles(ts, np.asarray(price, dtype="float64"), origin)
        target_ts = pd.date_range(ts[origin] + pd.Timedelta("1h"), periods=self.horizon, freq="h")
        return PriceForecastOutput(
            ts=target_ts,
            p10=q[:, 0],
            p50=q[:, 1],
            p90=q[:, 2],
            method=self.name,
            version=self.version,
            notes="MQ-RNN（arXiv:1711.11053，July 复现移植）：LSTM 编码 168h + "
            "forking-sequences 共享解码 + 直接多步分位数",
        )
