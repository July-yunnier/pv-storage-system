"""MES 与未来已知特征驱动的高级负荷预测可选模块。

该模块延迟加载 PyTorch，因此未安装 AI 依赖时，现有统计预测、调度和页面仍可运行。
模型同时编码历史 15 分钟序列和目标日 MES/天气/日历特征，避免只用历史
MES 却声称利用了目标日排产计划。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any

from app.domain.forecast.result import ForecastModel, ForecastPoint, TargetForecast

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15

try:  # 可选依赖：基础后端不因未安装 torch 而启动失败
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - 是否安装由部署环境决定
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("高级负荷模型需要 PyTorch，请安装 requirements-ai.txt")


if nn is not None:

    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.keys = nn.Linear(hidden_dim, hidden_dim)
            self.score = nn.Linear(hidden_dim, 1)

        def forward(self, sequence):  # type: ignore[no-untyped-def]
            query = sequence[:, -1:, :]
            scores = self.score(torch.tanh(self.query(query) + self.keys(sequence)))
            weights = F.softmax(scores, dim=1)
            return torch.sum(weights * sequence, dim=1), weights


    class EliteEnergyTaskNet(nn.Module):
        """历史编码器 + 目标日已知特征解码器 + 总厂/车间多任务头。"""

        def __init__(
            self,
            history_feature_dim: int,
            target_feature_dim: int,
            num_mes_plans: int = 11,
            mes_embed_dim: int = 8,
            hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            self.num_mes_plans = num_mes_plans
            self.mes_embedding = nn.Embedding(num_mes_plans, mes_embed_dim)
            self.history_lstm = nn.LSTM(
                history_feature_dim + mes_embed_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.3,
            )
            self.attention = BahdanauAttention(hidden_dim)
            decoder_dim = hidden_dim + target_feature_dim + mes_embed_dim
            self.target_decoder = nn.Sequential(
                nn.Linear(decoder_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.total_load_head = nn.Linear(hidden_dim, 1)
            self.workshop_load_head = nn.Linear(hidden_dim, 1)

        def _check_mes(self, labels) -> None:  # type: ignore[no-untyped-def]
            if labels.dtype != torch.long:
                raise ValueError("MES labels must use torch.long")
            if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_mes_plans):
                raise ValueError(
                    f"MES labels must be in [0, {self.num_mes_plans - 1}], "
                    "其中 0 应保留为 unknown"
                )

        def forward(  # type: ignore[no-untyped-def]
            self,
            history_features,
            history_mes_labels,
            target_features,
            target_mes_labels,
        ):
            if target_features.shape[1] != SLOTS_PER_DAY:
                raise ValueError("target_features must contain exactly 96 slots")
            if target_mes_labels.shape[1] != SLOTS_PER_DAY:
                raise ValueError("target_mes_labels must contain exactly 96 slots")
            self._check_mes(history_mes_labels)
            self._check_mes(target_mes_labels)

            historical_mes = self.mes_embedding(history_mes_labels)
            encoded, _ = self.history_lstm(
                torch.cat([history_features, historical_mes], dim=-1)
            )
            context, weights = self.attention(encoded)
            repeated_context = context.unsqueeze(1).expand(-1, SLOTS_PER_DAY, -1)
            target_mes = self.mes_embedding(target_mes_labels)
            decoded = self.target_decoder(
                torch.cat([repeated_context, target_features, target_mes], dim=-1)
            )
            total = self.total_load_head(decoded).squeeze(-1)
            workshop = self.workshop_load_head(decoded).squeeze(-1)
            return total, workshop, weights

else:  # 让导入和 API 文档生成保持可用，真正实例化时给出明确错误

    class BahdanauAttention:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class EliteEnergyTaskNet:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def multi_task_loss(
    total_pred,
    workshop_pred,
    total_target,
    workshop_target=None,
    *,
    workshop_weight: float = 0.4,
    consistency_weight: float = 0.1,
):  # type: ignore[no-untyped-def]
    """多任务损失，并惩罚车间预测高于总厂预测的物理矛盾。"""
    require_torch()
    main = F.mse_loss(total_pred, total_target)
    consistency = torch.relu(workshop_pred - total_pred).pow(2).mean()
    if workshop_target is None:
        return main + consistency_weight * consistency
    auxiliary = F.mse_loss(workshop_pred, workshop_target)
    return main + workshop_weight * auxiliary + consistency_weight * consistency


class OnlineResidualKalman:
    """日内残差滤波；修正影响随预见期衰减，不平移全天所有未来点。"""

    def __init__(self, q: float = 1e-4, r: float = 1e-2, decay: float = 0.95) -> None:
        if q < 0 or r <= 0 or not 0 < decay <= 1:
            raise ValueError("q >= 0, r > 0 and 0 < decay <= 1 are required")
        self.q, self.r, self.decay = q, r, decay
        self.x, self.p = 0.0, 1.0

    def update(self, measurement_residual: float) -> float:
        self.p += self.q
        gain = self.p / (self.p + self.r)
        self.x += gain * (measurement_residual - self.x)
        self.p *= 1 - gain
        return self.x

    def correct_remaining(self, baseline: list[float], observed_slot: int) -> list[float]:
        if not 0 <= observed_slot < len(baseline):
            raise ValueError("observed_slot is out of range")
        corrected = list(baseline)
        for slot in range(observed_slot + 1, len(corrected)):
            horizon = slot - observed_slot
            corrected[slot] = max(corrected[slot] + self.x * self.decay**horizon, 0.0)
        return corrected


def fine_tune_for_factory(
    pretrained_model,
    target_factory_loader,
    *,
    epochs: int = 3,
    lr: float = 1e-4,
    tune_mes_embedding: bool = True,
):  # type: ignore[no-untyped-def]
    """复制模型后微调，避免永久冻结调用方的预训练模型。"""
    require_torch()
    model = copy.deepcopy(pretrained_model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [model.total_load_head, model.workshop_load_head, model.target_decoder]
    if tune_mes_embedding:
        modules.append(model.mes_embedding)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=lr
    )
    model.eval()  # 冻结的 LSTM/Attention 关闭 dropout
    for module in modules:
        module.train()

    for epoch in range(epochs):
        last_loss: float | None = None
        for hist_x, hist_mes, target_x, target_mes, y_total, y_shop in target_factory_loader:
            hist_x = hist_x.to(device)
            hist_mes = hist_mes.to(device=device, dtype=torch.long)
            target_x = target_x.to(device)
            target_mes = target_mes.to(device=device, dtype=torch.long)
            y_total, y_shop = y_total.to(device), y_shop.to(device)
            optimizer.zero_grad()
            pred_total, pred_shop, _ = model(hist_x, hist_mes, target_x, target_mes)
            loss = multi_task_loss(pred_total, pred_shop, y_total, y_shop)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        if last_loss is None:
            raise ValueError("目标工厂微调数据集为空")
    return model


@dataclass(frozen=True)
class AdvancedLoadInputs:
    history_features: list[list[float]]
    history_mes_labels: list[int]
    target_features: list[list[float]]
    target_mes_labels: list[int]


def _standardize(rows: list[list[float]], mean: list[float], std: list[float]) -> list[list[float]]:
    if not rows or len(rows[0]) != len(mean) or len(mean) != len(std):
        raise ValueError("feature dimensions do not match model metadata")
    return [
        [(value - mean[i]) / (std[i] if abs(std[i]) > 1e-12 else 1.0) for i, value in enumerate(row)]
        for row in rows
    ]


def predict_with_artifact(
    state_dict,
    metadata: dict,
    inputs: AdvancedLoadInputs,
    target_date: date,
    tz: tzinfo,
) -> TargetForecast:
    """纯推理：对已加载的工件（state_dict + metadata）输出 96 点总负荷预测。

    不做文件 I/O、不读环境变量；工件加载与缓存见 infrastructure/model_store.py。
    输出非负、车间不超过总厂。
    """
    require_torch()
    model = EliteEnergyTaskNet(
        history_feature_dim=int(metadata["history_feature_dim"]),
        target_feature_dim=int(metadata["target_feature_dim"]),
        num_mes_plans=int(metadata["num_mes_plans"]),
        mes_embed_dim=int(metadata.get("mes_embed_dim", 8)),
        hidden_dim=int(metadata.get("hidden_dim", 128)),
    )
    model.load_state_dict(state_dict)
    model.eval()

    history = _standardize(
        inputs.history_features,
        metadata["history_feature_mean"],
        metadata["history_feature_std"],
    )
    future = _standardize(
        inputs.target_features,
        metadata["target_feature_mean"],
        metadata["target_feature_std"],
    )
    if len(inputs.history_mes_labels) != len(history):
        raise ValueError("history MES labels and history features have different lengths")
    if len(future) != SLOTS_PER_DAY or len(inputs.target_mes_labels) != SLOTS_PER_DAY:
        raise ValueError("target features and MES plan must both contain 96 slots")

    with torch.no_grad():
        total_z, workshop_z, _ = model(
            torch.tensor([history], dtype=torch.float32),
            torch.tensor([inputs.history_mes_labels], dtype=torch.long),
            torch.tensor([future], dtype=torch.float32),
            torch.tensor([inputs.target_mes_labels], dtype=torch.long),
        )
    output_mean = float(metadata["output_mean_kw"])
    output_std = float(metadata["output_std_kw"])
    total = [max(float(value) * output_std + output_mean, 0.0) for value in total_z[0]]
    workshop = [
        min(max(float(value) * output_std + output_mean, 0.0), total[index])
        for index, value in enumerate(workshop_z[0])
    ]
    # workshop 保留在模型内部和实验导出中；系统现有 TargetForecast 返回总厂负荷。
    del workshop

    day_start = datetime.combine(target_date, time.min, tzinfo=tz)
    points = [
        ForecastPoint(
            ts=day_start + timedelta(minutes=SLOT_MINUTES * slot),
            value=round(value, 6),
        )
        for slot, value in enumerate(total)
    ]
    return TargetForecast(
        target="load_power",
        unit="kW",
        model=ForecastModel(name="load_mes_multitask_attention", version="v1"),
        points=points,
        notes="MES目标日排产+未来特征+历史负荷的多任务注意力模型；输出已反标准化并限制非负",
    )


def predict_from_artifact(
    inputs: AdvancedLoadInputs,
    target_date: date,
    tz: tzinfo,
    model_path: str | None = None,
) -> TargetForecast:
    """兼容入口：经 model_store 加载（带缓存）后纯推理。新代码请改用
    ``application.load_features.predict_advanced``。"""
    from app.infrastructure.model_store import (  # 延迟导入：保持 domain 不硬依赖 infra
        default_load_model_path,
        load_bundle,
    )

    path = model_path or default_load_model_path()
    bundle = load_bundle(path)
    return predict_with_artifact(bundle.state_dict, bundle.metadata, inputs, target_date, tz)
