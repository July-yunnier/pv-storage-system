"""算法注册表：系统全部预测算法的单一事实源（元数据 + 可用性探测）。

此前算法清单硬编码在三处（forecast_service 的选择链、replay_service 的方法矩阵、
前端方法标签），新增算法要改业务代码。现在业务层一律从本注册表查询：

- API 入参校验：`validate_product_choice(target, key)`（未知选项报可用清单）
- 回放方法矩阵：`replay_methods("pv_power")`
- 算法发现接口：`GET /api/v1/algorithms` → `catalog()`

新增一个算法的全部工作（详见 docs/13-algorithm-contracts.md §4.2）：
1. 在 domain/forecast/ 实现纯函数预测（不做 I/O、不读环境变量）；
2. 在本文件 _CATALOG 登记一行元数据；
3. 在 forecast_service 的分发表里挂一个分支。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.infrastructure.model_store import load_model_unavailable_reason

# 预测对象
PV = "pv_power"
LOAD = "load_power"
PRICE = "price"


@dataclass(frozen=True)
class ForecastAlgorithm:
    """一个预测算法的登记卡。"""

    name: str  # 规范方法名（回放方法名 / 输出 model.name 的实验口径）
    target: str
    kind: str  # statistical / physics / ml / baseline / oracle / policy
    summary: str  # 一句话说明
    product_key: str | None = None  # ForecastOptions 中 pv_model/load_model 的取值；None=不对产品开放
    model_name: str | None = None  # 产品路径输出 TargetForecast.model.name（None=与 name 相同）
    replay: bool = False  # 进入回放方法矩阵
    intraday: bool = False  # 日内滚动模式（由 intraday=True 触发，非常规日前选项）
    requires_history_days: int = 0
    requires_weather: bool = False
    requires_torch: bool = False
    requires_artifact: bool = False


_CATALOG: tuple[ForecastAlgorithm, ...] = (
    # ---- 光伏 ----
    ForecastAlgorithm(
        name="day_persistence", target=PV, kind="baseline",
        summary="昨日延续基线：明天=昨天（回放基准，skill 的 0 分参照）",
        replay=True, requires_history_days=1,
    ),
    ForecastAlgorithm(
        name="slot_mean", target=PV, kind="statistical",
        summary="同刻均值基线：近 12 天同时刻平均",
        product_key="statistical", model_name="pv_diurnal_stats",
        replay=True, requires_history_days=3,
    ),
    ForecastAlgorithm(
        name="slot_mean_bias", target=PV, kind="statistical",
        summary="同刻均值 + 昨日偏差延续（alpha=0.25）",
        replay=True, requires_history_days=3,
    ),
    ForecastAlgorithm(
        name="physics", target=PV, kind="physics",
        summary="pvlib 物理链：气象预报 → 太阳位置 → POA → PVWatts；无历史冷启动",
        product_key="physics", model_name="pvlib_pvwatts",
        replay=True, requires_weather=True,
    ),
    ForecastAlgorithm(
        name="blend", target=PV, kind="physics+statistical",
        summary="融合：物理链 × 实测校准系数（auto 默认路径，有历史时优先）",
        model_name="pvlib_blend", replay=True,
        requires_weather=True, requires_history_days=3,
    ),
    ForecastAlgorithm(
        name="oracle", target=PV, kind="oracle",
        summary="事后上帝视角：用实测当预测（理论上界，仅回放参考，非产品能力）",
        replay=True,
    ),
    ForecastAlgorithm(
        name="pv_intraday_deviation", target=PV, kind="statistical",
        summary="日内滚动修正：实测持续外推 + 偏差随时效衰减回归基线",
        intraday=True, requires_history_days=3,
    ),
    # ---- 负荷 ----
    ForecastAlgorithm(
        name="load_diurnal_mean", target=LOAD, kind="statistical",
        summary="同刻均值基线：近 12 天同时刻平均",
        product_key="statistical", requires_history_days=3,
    ),
    ForecastAlgorithm(
        name="load_mes_multitask_attention", target=LOAD, kind="ml",
        summary="MES 排产 + 多任务注意力（LSTM）：30 天历史 + 目标日排产/气象特征；"
                "工件仿真预训练，真实数据经 fine_tune 落地",
        product_key="advanced", requires_torch=True, requires_artifact=True,
        requires_weather=True, requires_history_days=30,
    ),
    ForecastAlgorithm(
        name="load_intraday_kalman", target=LOAD, kind="statistical",
        summary="日内滚动修正：同刻均值基线 + 当日实测残差卡尔曼滤波",
        intraday=True, requires_history_days=3,
    ),
    # ---- 电价 ----
    ForecastAlgorithm(
        name="tou_policy", target=PRICE, kind="policy",
        summary="TOU 政策电价查表（确定量，无不确定性）；现货价格预测另立算法线",
    ),
)


def catalog(target: str | None = None) -> list[ForecastAlgorithm]:
    return [a for a in _CATALOG if target is None or a.target == target]


def get(name: str) -> ForecastAlgorithm | None:
    return next((a for a in _CATALOG if a.name == name), None)


def product_options(target: str) -> list[str]:
    """产品 API 可选的选择键（pv_model/load_model 取值），按登记顺序。"""
    return [a.product_key for a in _CATALOG if a.target == target and a.product_key]


def find_by_product_key(target: str, key: str) -> ForecastAlgorithm | None:
    return next(
        (a for a in _CATALOG if a.target == target and a.product_key == key), None
    )


def validate_product_choice(target: str, key: str) -> None:
    """产品选择键校验：未知取值抛 ValueError（携带可用清单）。"""
    if find_by_product_key(target, key) is None:
        options = "、".join(product_options(target)) or "（无）"
        raise ValueError(f"未知的{target}算法选项: {key!r}；可用: {options}")


def replay_methods(target: str) -> list[str]:
    """回放实验方法矩阵（按登记顺序）。"""
    return [a.name for a in _CATALOG if a.target == target and a.replay]


def unavailable_reason(entry: ForecastAlgorithm) -> str | None:
    """可用性探测：返回 None 表示当前环境可用，否则返回原因。"""
    if entry.requires_torch or entry.requires_artifact:
        return load_model_unavailable_reason()
    return None


def catalog_view() -> list[dict]:
    """GET /algorithms 的序列化视图。"""
    view: list[dict] = []
    for a in _CATALOG:
        reason = unavailable_reason(a)
        view.append({
            "name": a.name,
            "target": a.target,
            "kind": a.kind,
            "summary": a.summary,
            "product_key": a.product_key,
            "model_name": a.model_name or a.name,
            "replay": a.replay,
            "intraday": a.intraday,
            "requires": {
                "history_days": a.requires_history_days,
                "weather": a.requires_weather,
                "torch": a.requires_torch,
                "artifact": a.requires_artifact,
            },
            "available": reason is None,
            "unavailable_reason": reason,
        })
    return view
