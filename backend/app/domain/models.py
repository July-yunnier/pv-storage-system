from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.shift_calendar import ShiftCalendar


class SiteLocation(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timezone: str = "Asia/Shanghai"
    province: str | None = None
    city: str | None = None


class PVConfig(BaseModel):
    capacity_kwp: float = Field(gt=0)
    tilt_deg: float = Field(default=22.5, ge=0, le=90)
    azimuth_deg: float = Field(default=180, ge=0, le=360)
    system_loss: float = Field(default=0.14, ge=0, lt=1)


class LoadConfig(BaseModel):
    annual_kwh: float | None = Field(default=None, ge=0)
    peak_kw: float | None = Field(default=None, ge=0)
    profile_type: str | None = None
    # 班次/排产日历：高级负荷模型的 MES 标签来源；None = 默认两班制
    mes_calendar: ShiftCalendar | None = None


class StorageConfig(BaseModel):
    pcs_kw: float = Field(gt=0)
    capacity_kwh: float = Field(gt=0)
    round_trip_eff: float = Field(default=0.88, gt=0, le=1)
    soc_min: float = Field(default=0.10, ge=0, le=1)
    soc_max: float = Field(default=0.95, ge=0, le=1)
    cycles_per_day_max: float = Field(default=2, gt=0)
    allow_grid_export: bool = False

    @model_validator(mode="after")
    def validate_soc_range(self) -> "StorageConfig":
        if self.soc_max <= self.soc_min:
            raise ValueError("soc_max must be greater than soc_min")
        return self

    @property
    def charge_eff(self) -> float:
        return self.round_trip_eff**0.5

    @property
    def discharge_eff(self) -> float:
        return self.round_trip_eff**0.5


class SiteCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: str | None = None
    name: str = Field(min_length=1, max_length=120)
    location: SiteLocation
    pv: PVConfig
    load: LoadConfig = Field(default_factory=LoadConfig)
    storage: StorageConfig
    tariff_id: str | None = "T-DEFAULT"
    demand_limit_kw: float | None = Field(default=None, gt=0)

    @field_validator("site_id")
    @classmethod
    def normalize_site_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class SitePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    tariff_id: str | None = None


class SiteRead(SiteCreate):
    site_id: str
    created_at: datetime
    updated_at: datetime


MeasurementType = Literal["pv_power", "load_power", "storage_soc", "grid_power"]

#: 现货电价单独一张表：单位是 CNY/kWh 且允许负值（负电价是真实市场现象），
#: 与功率测量（kW、非负）不是同一口径，不能复用 measurements。
SpotPriceQuality = Literal["measured", "estimated", "missing", "invalid"]


class SpotPricePoint(BaseModel):
    ts: datetime
    price_cny_per_kwh: float | None = None
    quality: SpotPriceQuality = "measured"

    @field_validator("ts")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must include a timezone")
        return value


class SpotPriceUpload(BaseModel):
    """现货电价导入。允许负电价，不允许无时区时间戳。"""

    source: str = "api"
    points: list[SpotPricePoint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_points(self) -> "SpotPriceUpload":
        timestamps = [point.ts for point in self.points]
        if timestamps != sorted(timestamps):
            raise ValueError("points must be sorted by ts")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("points must not contain duplicate timestamps")
        return self


class SpotPriceRead(SpotPricePoint):
    source: str


class MeasurementPoint(BaseModel):
    ts: datetime
    value: float | None = None
    quality: Literal["measured", "estimated", "missing", "invalid"] = "measured"

    @field_validator("ts")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must include a timezone")
        return value


class MeasurementUpload(BaseModel):
    data_type: MeasurementType
    unit: str = "kW"
    resolution_minutes: int = Field(default=15, gt=0)
    source: str = "api"
    points: list[MeasurementPoint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_points(self) -> "MeasurementUpload":
        timestamps = [point.ts for point in self.points]
        if timestamps != sorted(timestamps):
            raise ValueError("points must be sorted by ts")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("points must not contain duplicate timestamps")
        return self


class MeasurementRead(MeasurementPoint):
    data_type: MeasurementType
    unit: str
    source: str


TierName = Literal["sharp", "peak", "flat", "valley"]


class TariffTier(BaseModel):
    name: TierName
    price: float = Field(ge=0, description="电能量价格，CNY/kWh")
    hours: list[str] = Field(min_length=1)

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, values: list[str]) -> list[str]:
        for item in values:
            parts = item.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid hour range: {item}")
            for part in parts:
                hour, minute = part.split(":")
                if not (0 <= int(hour) <= 23 and int(minute) in (0, 15, 30, 45)):
                    raise ValueError(f"invalid time in range: {item}")
        return values


class TariffCreate(BaseModel):
    tariff_id: str | None = None
    name: str = Field(min_length=1, max_length=120)
    mode: Literal["tou", "spot", "tou_plus_spot"] = "tou"
    transmission_and_fund: float = Field(default=0.21, ge=0)
    tiers: list[TariffTier] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tiers(self) -> "TariffCreate":
        names = [tier.name for tier in self.tiers]
        if len(names) != len(set(names)):
            raise ValueError("tariff tier names must be unique")
        return self


class TariffRead(TariffCreate):
    tariff_id: str
    created_at: datetime
    updated_at: datetime


class PricePoint(BaseModel):
    ts: datetime
    price: float
    energy_price: float
    tier: TierName | None = None
    p10: float | None = None
    p90: float | None = None


class ScheduleRequest(BaseModel):
    site_id: str
    date: date
    initial_soc: float = Field(default=0.5, ge=0, le=1)
    strategy: Literal["rule_v1"] = "rule_v1"
    resolution_minutes: int = Field(default=15, gt=0)
    source: Literal["auto", "measurements", "forecast"] = "auto"


ForecastTarget = Literal["pv_power", "load_power", "price"]


class AdvancedLoadInput(BaseModel):
    """高级负荷模型输入；未来 MES/天气接口接通前可由仿真回放直接提供。"""

    history_features: list[list[float]] = Field(min_length=1)
    history_mes_labels: list[int] = Field(min_length=1)
    target_features: list[list[float]] = Field(min_length=96, max_length=96)
    target_mes_labels: list[int] = Field(min_length=96, max_length=96)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "AdvancedLoadInput":
        if len(self.history_features) != len(self.history_mes_labels):
            raise ValueError("history_features and history_mes_labels lengths differ")
        if any(not row for row in self.history_features + self.target_features):
            raise ValueError("feature rows must not be empty")
        history_dim = len(self.history_features[0])
        target_dim = len(self.target_features[0])
        if any(len(row) != history_dim for row in self.history_features):
            raise ValueError("history feature dimensions are inconsistent")
        if any(len(row) != target_dim for row in self.target_features):
            raise ValueError("target feature dimensions are inconsistent")
        return self


class ForecastOptions(BaseModel):
    include_uncertainty: bool = True
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    # 算法选择键的取值集合由算法注册表（app.application.algorithm_registry）定义，
    # 这里刻意用宽 str：新算法在注册表登记即可用，不必改请求 schema；
    # 非法取值在 forecast_service 入口统一 422。
    pv_model: str = "auto"  # "auto" 为服务策略键（有历史走融合，否则物理链）
    load_model: str = "statistical"
    # 高级模型输入：None = 服务端自动构造（历史负荷+气象+班次日历）；显式传入用于实验/回放
    advanced_load_input: AdvancedLoadInput | None = None
    intraday: bool = False
    explain: bool = False  # true 时 provenance 附带气象原始点与回测明细


class ForecastRequest(BaseModel):
    site_id: str
    targets: list[ForecastTarget] = Field(min_length=1)
    horizon_hours: int = Field(default=24, gt=0)
    resolution_minutes: int = Field(default=15, gt=0)
    issued_at: datetime | None = None
    options: ForecastOptions = Field(default_factory=ForecastOptions)

    @field_validator("issued_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("issued_at must include a timezone")
        return value


class SizingCandidates(BaseModel):
    pcs_kw: list[float] = Field(min_length=1)
    duration_hours: list[float] = Field(min_length=1)

    @field_validator("pcs_kw", "duration_hours")
    @classmethod
    def positive_values(cls, values: list[float]) -> list[float]:
        if any(value <= 0 for value in values):
            raise ValueError("candidate values must be positive")
        return values


class FinancialOverrides(BaseModel):
    battery_price_per_kwh: float | None = Field(default=None, ge=0)
    pcs_price_per_kw: float | None = Field(default=None, ge=0)
    install_ratio: float | None = Field(default=None, ge=0)
    opex_ratio: float | None = Field(default=None, ge=0)
    project_years: int | None = Field(default=None, ge=1)
    battery_life_years: int | None = Field(default=None, ge=1)
    replacement_cost_ratio: float | None = Field(default=None, ge=0)
    discount_rate: float | None = Field(default=None, ge=-0.99)
    residual_ratio: float | None = Field(default=None, ge=0)


class SizingRequest(BaseModel):
    site_id: str
    candidates: SizingCandidates
    representative_days: int = Field(default=3, ge=1, le=14)
    financial: FinancialOverrides = Field(default_factory=FinancialOverrides)


class ReportCreate(BaseModel):
    site_id: str
    sizing_run_id: str


ReportStatus = Literal["pending", "running", "succeeded", "failed"]


class ReportJob(BaseModel):
    report_id: str
    site_id: str
    report_type: str
    sizing_run_id: str | None
    status: ReportStatus
    error: str | None
    file_path: str | None
    created_at: datetime
    finished_at: datetime | None


class ReportJobStatusResponse(BaseModel):
    report_id: str
    site_id: str
    report_type: str
    status: ReportStatus
    error: str | None
    created_at: datetime
    finished_at: datetime | None
    download_url: str


class DispatchPoint(BaseModel):
    ts: datetime
    action: Literal["charge", "discharge", "idle"]
    power_kw: float = Field(ge=0)
    soc_start: float
    soc_end: float
    pv_power_kw: float
    load_power_kw: float
    price: float
    tier: TierName | None
    grid_import_kw: float = Field(ge=0)
    grid_export_kw: float = Field(ge=0)
    pv_curtailment_kw: float = Field(ge=0)
    # 充电来源拆分（charge 时）：光伏余电直充 vs 电网购电充电；默认 0 兼容旧记录
    charge_from_pv_kw: float = Field(default=0, ge=0)
    charge_from_grid_kw: float = Field(default=0, ge=0)


class PlanSegment(BaseModel):
    """计划说明的一段：连续相同动作的时段 + 人话解释。"""

    start: str  # HH:mm
    end: str  # HH:mm（段终点，闭区间的下一槽）
    action: Literal["charge_pv", "charge_grid", "discharge"]
    tier: TierName | None
    avg_power_kw: float
    text: str


class BenefitBreakdown(BaseModel):
    """预期收益的构成（单位：元 / kWh），供前端"电费去哪了"展示。

    恒等式：expected_benefit = discharge_value − grid_charge_cost（不允许上网口径下精确成立）。
    中午光伏余电充电不花电网的钱，其价值通过随后的放电体现。
    """

    discharge_value_yuan: float  # 放电时段替代电网购电的价值
    grid_charge_cost_yuan: float  # 谷段从电网购电充电的成本
    discharge_kwh: float
    grid_charge_kwh: float
    surplus_charge_kwh: float  # 光伏余电充入电量（免费，不含效率损耗）


class ScheduleResponse(BaseModel):
    site_id: str
    date: date
    strategy: str
    expected_benefit_yuan: float
    baseline_cost_yuan: float
    optimized_cost_yuan: float
    equivalent_full_cycles: float
    schedule: list[DispatchPoint]
    constraints_applied: list[str]
    benefit_breakdown: BenefitBreakdown | None = None
    run_id: str | None = None
    data_source: Literal["measurements", "forecast"] = "measurements"
    # 计划说明（分时段人话解释）；None = 旧记录，读取路径会补算
    explanation: list[PlanSegment] | None = None


class DispatchRunSummary(BaseModel):
    run_id: str
    site_id: str
    date: date
    strategy: str
    initial_soc: float
    expected_benefit_yuan: float
    baseline_cost_yuan: float
    optimized_cost_yuan: float
    equivalent_full_cycles: float
    data_source: Literal["measurements", "forecast"] = "measurements"
    auto_generated: bool = False
    created_at: datetime


class DispatchRunDetail(ScheduleResponse):
    input_digest: str
    created_at: datetime


class LiveLatestPoint(BaseModel):
    ts: datetime
    value: float


class LivePriceNow(BaseModel):
    price: float
    tier: TierName | None
    next_change_at: datetime | None = None
    next_tier: TierName | None = None


class LiveTodayPlan(BaseModel):
    run_id: str
    expected_benefit_yuan: float
    created_at: datetime


class LiveStatus(BaseModel):
    """实时看板快照：当前功率/SOC/电价档 + 今日数据覆盖与今日计划。"""

    site_id: str
    now_local: datetime
    latest_pv: LiveLatestPoint | None
    latest_load: LiveLatestPoint | None
    latest_soc: LiveLatestPoint | None
    latest_grid: LiveLatestPoint | None
    price_now: LivePriceNow
    today_plan: LiveTodayPlan | None
    pv_points_today: int
    load_points_today: int
    expected_points_today: int


class LiveSimulateResponse(BaseModel):
    site_id: str
    inserted: dict[str, int]
    up_to_slot: int


class RollingRevisionRead(BaseModel):
    revision: int
    created_at: datetime
    initial_soc: float
    expected_benefit_yuan: float
    trigger_reason: str


class RollingState(BaseModel):
    """日内滚动修正状态：原计划 + 修正历史 + 最新修正计划（仅剩余时段）。"""

    site_id: str
    date: date
    base_run_id: str | None = None
    base_expected_benefit_yuan: float | None = None
    revisions: list[RollingRevisionRead]
    latest_schedule: list[DispatchPoint] | None = None
    latest_revision: int
    cooldown_until: datetime | None = None
    deviation_now: float | None = None
    last_trigger_reason: str | None = None


def generate_site_id() -> str:
    return f"S{uuid4().hex[:8].upper()}"


def generate_tariff_id() -> str:
    return f"T{uuid4().hex[:8].upper()}"
