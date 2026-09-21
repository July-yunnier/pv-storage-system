"""站点档案数据模型。

用标准库 dataclass 而非 pydantic：
- 零依赖，克隆下来就能跑，不用先装包
- 后面接 FastAPI 时，pydantic 可直接从 dataclass 适配

单位约定（写在字段名里，避免 kW/kWp/kWh 混淆）
    _kwp   光伏装机容量（峰值功率）
    _kw    功率
    _kwh   电量
    _deg   角度
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfileType = Literal["single-shift", "two-shift", "three-shift", "continuous"]


@dataclass(frozen=True)
class Location:
    """站点位置。

    决定了三件事：光伏资源（辐照度）、电价政策（省级目录电价）、气候特征。
    所以它不是一个可选的装饰字段。
    """

    lat: float
    lon: float
    province: str = ""
    city: str = ""
    timezone: str = "Asia/Shanghai"

    def __post_init__(self) -> None:
        if not -90 <= self.lat <= 90:
            raise ValueError(f"纬度越界: {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"经度越界: {self.lon}")


@dataclass(frozen=True)
class PVConfig:
    """光伏系统参数。"""

    capacity_kwp: float
    tilt_deg: float = 22.5
    azimuth_deg: float = 180.0          # 180 = 正南
    module_type: str = "mono-si"
    temp_coeff: float = -0.0035         # 功率温度系数 /℃
    system_loss: float = 0.14           # 线损+积灰+失配+逆变器综合损失

    def __post_init__(self) -> None:
        if self.capacity_kwp <= 0:
            raise ValueError("capacity_kwp 必须为正")
        if not 0 <= self.system_loss < 1:
            raise ValueError("system_loss 应在 [0, 1) 区间")

    @property
    def derate_factor(self) -> float:
        """系统综合折减系数。"""
        return 1.0 - self.system_loss


@dataclass(frozen=True)
class LoadConfig:
    """用电负荷参数。"""

    annual_kwh: float
    peak_kw: float
    profile_type: ProfileType = "two-shift"

    def __post_init__(self) -> None:
        if self.peak_kw <= 0:
            raise ValueError("peak_kw 必须为正")

    @property
    def avg_kw(self) -> float:
        """年均负荷功率（8760 小时折算）。"""
        return self.annual_kwh / 8760.0


@dataclass(frozen=True)
class StorageConfig:
    """储能系统参数。

    ⚠️ 关于效率的两个口径，别混用：
        round_trip_eff   一次完整充放循环的综合效率（行业惯例 0.85~0.92）
        one_way_eff      单程效率 = sqrt(round_trip_eff)，用于逐步计算
    很多项目把两者搞混，导致收益被高估约 10%。
    """

    pcs_kw: float
    capacity_kwh: float
    chemistry: str = "LFP"
    round_trip_eff: float = 0.88
    depth_of_discharge: float = 0.90
    soc_min: float = 0.10
    soc_max: float = 0.95
    cycles_per_day_max: float = 2.0

    def __post_init__(self) -> None:
        if self.pcs_kw <= 0:
            raise ValueError("pcs_kw 必须为正")
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh 必须为正")
        if not 0.5 <= self.round_trip_eff <= 1.0:
            raise ValueError("round_trip_eff 应在 [0.5, 1.0]")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError(f"SOC 区间非法: [{self.soc_min}, {self.soc_max}]")

    @property
    def usable_kwh(self) -> float:
        """可用电量（受 SOC 上下限约束）。"""
        return self.capacity_kwh * (self.soc_max - self.soc_min)

    @property
    def one_way_eff(self) -> float:
        """单程效率。逐步充放计算时用这个。"""
        return self.round_trip_eff ** 0.5

    @property
    def duration_hours(self) -> float:
        """额定放电时长（小时）。"""
        return self.capacity_kwh / self.pcs_kw

    @property
    def soc_energy_kwh(self) -> tuple[float, float]:
        """SOC 上下限对应的电量 (min_kwh, max_kwh)。"""
        return self.capacity_kwh * self.soc_min, self.capacity_kwh * self.soc_max


@dataclass(frozen=True)
class Site:
    """站点档案——全系统的根对象。"""

    id: str
    name: str
    location: Location
    pv: PVConfig
    load: LoadConfig
    storage: StorageConfig | None = None

    def describe(self) -> str:
        lines = [
            f"站点 {self.id} · {self.name}",
            f"  位置    {self.location.lat:.4f}, {self.location.lon:.4f} "
            f"({self.location.province}{self.location.city})",
            f"  光伏    {self.pv.capacity_kwp:.0f} kWp  "
            f"倾角 {self.pv.tilt_deg}° 方位 {self.pv.azimuth_deg}°",
            f"  负荷    年 {self.load.annual_kwh:,.0f} kWh  "
            f"峰值 {self.load.peak_kw:.0f} kW  年均 {self.load.avg_kw:.1f} kW",
        ]
        if self.storage:
            s = self.storage
            lines.append(
                f"  储能    {s.pcs_kw:.0f} kW / {s.capacity_kwh:.0f} kWh "
                f"({s.duration_hours:.1f}h)  效率 {s.round_trip_eff:.2f}  "
                f"可用 {s.usable_kwh:.0f} kWh"
            )
        else:
            lines.append("  储能    未配置")
        return "\n".join(lines)


def site_from_config(cfg: dict) -> Site:
    """从 YAML 配置字典构造 Site。

    配置文件结构见 config/site.example.yaml。
    """
    loc = cfg["location"]
    pv = cfg["pv"]
    ld = cfg["load"]
    st = cfg.get("storage")

    return Site(
        id=cfg["id"],
        name=cfg["name"],
        location=Location(
            lat=loc["lat"], lon=loc["lon"],
            province=loc.get("province", ""), city=loc.get("city", ""),
            timezone=loc.get("timezone", "Asia/Shanghai"),
        ),
        pv=PVConfig(
            capacity_kwp=pv["capacity_kwp"],
            tilt_deg=pv.get("tilt_deg", 22.5),
            azimuth_deg=pv.get("azimuth_deg", 180.0),
            module_type=pv.get("module_type", "mono-si"),
            temp_coeff=pv.get("temp_coeff", -0.0035),
            system_loss=pv.get("system_loss", 0.14),
        ),
        load=LoadConfig(
            annual_kwh=ld["annual_kwh"],
            peak_kw=ld["peak_kw"],
            profile_type=ld.get("profile_type", "two-shift"),
        ),
        storage=StorageConfig(
            pcs_kw=st["pcs_kw"],
            capacity_kwh=st["capacity_kwh"],
            chemistry=st.get("chemistry", "LFP"),
            round_trip_eff=st.get("round_trip_eff", 0.88),
            depth_of_discharge=st.get("depth_of_discharge", 0.90),
            soc_min=st.get("soc_min", 0.10),
            soc_max=st.get("soc_max", 0.95),
            cycles_per_day_max=st.get("cycles_per_day_max", 2.0),
        ) if st else None,
    )
