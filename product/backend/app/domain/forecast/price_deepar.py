"""DeepAR 的电价适配：共享 LSTM、Gaussian NLL、缩放与递归路径采样。

价格是有符号连续变量，不能采用计数分布或对预测做非负裁剪。该实现是
arXiv:1704.04110 的独立工程适配（保留为旧基线，不是用户指定的 MQ-RNN 论文），不依赖 GluonTS，也不包含论文预训练权重。
"""
from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

from app.domain.forecast.result import (
    ForecastModel, ForecastPoint, ForecastProvenance, TargetForecast,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # 基础 API 不要求安装 AI 依赖
    torch = nn = F = None

from app.domain.forecast.price_data import FEATURE_NAMES, calendar_features, validate_series
FORMAT = "price_deepar_v1"


def require_torch():
    if torch is None:
        raise RuntimeError("DeepAR 电价模型需要 PyTorch，请安装 requirements-ai.txt")


if nn is not None:
    class PriceDeepAR(nn.Module):
        def __init__(self, num_series: int = 1, hidden_size: int = 40,
                     num_layers: int = 2, embedding_size: int = 4):
            super().__init__()
            self.embedding = nn.Embedding(num_series, embedding_size)
            self.rnn = nn.LSTM(1 + len(FEATURE_NAMES) + embedding_size, hidden_size,
                               num_layers=num_layers, batch_first=True)
            self.mu = nn.Linear(hidden_size, 1)
            self.sigma = nn.Linear(hidden_size, 1)

        def forward(self, previous, features, series_ids, state=None):
            embedding = self.embedding(series_ids)[:, None, :].expand(-1, previous.shape[1], -1)
            encoded, state = self.rnn(torch.cat([previous[..., None], features, embedding], -1), state)
            return self.mu(encoded).squeeze(-1), F.softplus(self.sigma(encoded).squeeze(-1)) + 1e-4, state
else:
    class PriceDeepAR:
        def __init__(self, *args, **kwargs):
            require_torch()


def gaussian_nll(mu, sigma, target):
    return (0.5 * ((target - mu) / sigma).square() + sigma.log()
            + 0.5 * math.log(2 * math.pi)).mean()


def load_artifact(model_path: str | None = None):
    require_torch()
    raw_path = model_path or os.getenv("PRICE_MODEL_PATH", "")
    if not raw_path or not Path(raw_path).is_file():
        raise ValueError("未配置有效的 PRICE_MODEL_PATH；请先离线训练电价模型")
    path = Path(raw_path)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    metadata = artifact["metadata"]
    if metadata.get("format") != FORMAT or metadata.get("feature_names") != FEATURE_NAMES:
        raise ValueError("unsupported DeepAR artifact format or features")
    if metadata["resolution_minutes"] not in (15, 60) or metadata["context_length"] < 2:
        raise ValueError("invalid DeepAR resolution or context length")
    model = PriceDeepAR(**metadata["architecture"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, metadata, hashlib.sha256(path.read_bytes()).hexdigest()


def sample_paths(model, history: list[tuple[datetime, float]], future: list[datetime],
                 *, series_index: int = 0, num_samples: int = 200, seed: int = 42):
    """每条路径把自身上一步采样送回 LSTM，保留跨预测步的误差传播。"""
    require_torch()
    if not future or not 20 <= num_samples <= 2000:
        raise ValueError("future is required and num_samples must be in [20, 2000]")
    values = torch.tensor([v for _, v in history], dtype=torch.float32)
    scale = 1 + values.abs().mean()  # 仅由预测原点之前的 context 计算，支持负价/全零
    normalized = values / scale
    previous = torch.cat([normalized[:1], normalized[:-1]])[None, :]
    features = torch.tensor([[calendar_features(ts) for ts, _ in history]], dtype=torch.float32)
    series_ids = torch.full((num_samples,), series_index, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        _, _, state = model(previous, features, series_ids[:1])
        state = tuple(s.repeat(1, num_samples, 1) for s in state)
        last = normalized[-1].repeat(num_samples)
        paths = []
        for ts in future:
            x = torch.tensor(calendar_features(ts), dtype=torch.float32)[None, None, :].expand(num_samples, 1, -1)
            mu, sigma, state = model(last[:, None], x, series_ids, state)
            last = mu[:, 0] + sigma[:, 0] * torch.randn(num_samples, generator=generator)
            paths.append(last * scale)
    result = torch.stack(paths, dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("DeepAR produced non-finite price samples")
    return result


def predict_price_deepar(
    history: list[tuple[datetime, float]], start: datetime, periods: int,
    issued_at: datetime, *, timezone_name: str, series_id: str,
    transmission_and_fund: float = 0, include_uncertainty: bool = True,
    num_samples: int = 200, seed: int = 42, model_path: str | None = None,
) -> TargetForecast:
    model, metadata, digest = load_artifact(model_path)
    if metadata["unit"] != "CNY/kWh":
        raise ValueError("在线电价模型必须使用 CNY/kWh；禁止把外币/MWh 权重当作人民币价格")
    if metadata["timezone"] != timezone_name:
        raise ValueError("price model timezone does not match site timezone")
    if series_id not in metadata["series_ids"]:
        raise ValueError(f"price series_id {series_id!r} is absent from model metadata")
    if datetime.fromisoformat(metadata["selection_end"]) >= issued_at:
        raise ValueError("price model training/validation cutoff must precede issued_at")
    if periods <= 0 or periods > 288 or start < issued_at:
        raise ValueError("DeepAR price forecast requires a future start and at most 72 hours")
    # 不使用发布时点之后的数据库实测，即使回放库中已有未来数据。
    history = [(ts.astimezone(start.tzinfo), v) for ts, v in history if ts < issued_at and ts < start]
    context_length = int(metadata["context_length"])
    if len(history) < context_length:
        raise ValueError(f"DeepAR price needs at least {context_length} native history points")
    history = history[-context_length:]
    resolution = int(metadata["resolution_minutes"])
    validate_series(history, resolution)
    step = timedelta(minutes=resolution)
    first = history[-1][0] + step
    if (issued_at - first).total_seconds() > resolution * 60:
        raise ValueError("price history is stale at issued_at")
    if start.minute % resolution or start.second or start.microsecond:
        raise ValueError("forecast start must align with model native resolution")
    end = start + timedelta(minutes=15 * periods)
    future = []
    ts = first
    while ts < end:
        future.append(ts)
        ts += step
    if len(future) > 5 * 1440 // resolution:
        raise ValueError("price forecast bridge exceeds five days")
    paths = sample_paths(model, history, future,
                         series_index=metadata["series_ids"].index(series_id),
                         num_samples=num_samples, seed=seed)
    quantiles = torch.quantile(paths, torch.tensor([0.1, 0.5, 0.9]), dim=0)
    # 小时结算价向 15 分钟接口按所属小时展开，不伪造 15 分钟预测能力。
    timestamps = [start + timedelta(minutes=15 * i) for i in range(periods)]
    indices = [int((ts - first).total_seconds() // (resolution * 60)) for ts in timestamps]
    curves = [[round(float(row[i]) + transmission_and_fund, 6) for i in indices] for row in quantiles]
    note = ("DeepAR LSTM+Gaussian；递归采样 p10/p50/p90，点预测为 p50；允许负电价；"
            "预测电能量价后仅加一次输配及基金附加；区间尚未经本站实测覆盖率校准")
    if resolution == 60:
        note += "；原生小时模型，按小时结算价展开为15分钟接口点"
    return TargetForecast(
        target="price", unit="CNY/kWh", model=ForecastModel(name="price_deepar", version="v1"),
        points=[ForecastPoint(ts=ts, value=v) for ts, v in zip(timestamps, curves[1])],
        p10=curves[0] if include_uncertainty else None,
        p90=curves[2] if include_uncertainty else None,
        provenance=ForecastProvenance(
            provider="local_price_history", checkpoint_sha256=digest,
            training_status="offline_trained",
            window=f"context={history[0][0].isoformat()}..{history[-1][0].isoformat()}; "
                   f"native={resolution}min; samples={num_samples}; seed={seed}; "
                   f"selection_end={metadata['selection_end']}; series={series_id}",
        ), notes=note,
    )
