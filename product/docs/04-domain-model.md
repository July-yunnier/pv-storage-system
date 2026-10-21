# 领域模型与数据字典

## 1. 核心对象

```text
Site
 ├── SiteLocation
 ├── PVConfig
 ├── LoadConfig
 ├── StorageConfig
 └── TariffReference

MeasurementSeries
ForecastRun
Tariff
DispatchRequest
DispatchResult
SizingResult
ReportJob
```

## 2. 时间序列规范

| 项 | 规则 |
|---|---|
| 时间戳 | ISO 8601，必须带时区 |
| 默认时区 | `Asia/Shanghai` |
| 颗粒度 | MVP 固定 15 分钟；天气可为 60 分钟 |
| 顺序 | `points` 按时间升序 |
| 缺失值 | 使用 `null`，不使用 0 代替 |
| 功率 | kW；光伏装机使用 kWp |
| 能量 | kWh |
| 电价 | CNY/kWh |
| SOC | 0 到 1 的小数 |

## 3. Site 字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| site_id | string | 是 | 站点唯一标识 |
| name | string | 是 | 站点名称 |
| location.lat | number | 是 | 纬度 |
| location.lon | number | 是 | 经度 |
| location.timezone | string | 是 | IANA 时区 |
| location.province | string | 否 | 省份 |
| pv.capacity_kwp | number | 是 | 光伏额定容量 |
| pv.system_loss | number | 否 | 系统损耗比例 |
| load.annual_kwh | number | 否 | 年负荷 |
| load.peak_kw | number | 否 | 峰值负荷 |
| storage.pcs_kw | number | 是 | PCS 功率 |
| storage.capacity_kwh | number | 是 | 电池容量 |
| storage.round_trip_eff | number | 是 | 往返效率 |
| storage.soc_min | number | 是 | SOC 下限 |
| storage.soc_max | number | 是 | SOC 上限 |

## 4. 统一价格点

```json
{
  "ts": "2026-09-20T08:00:00+08:00",
  "price": 0.65,
  "energy_price": 0.44,
  "tier": "flat",
  "p10": null,
  "p90": null
}
```

`price` 是到户电价；`energy_price` 是可以由现货价格替代的电能量部分；`tier` 用于界面标识，不作为现货价格存在的必要条件。

## 5. 测量点

```json
{
  "ts": "2026-09-20T12:00:00+08:00",
  "value": 623.4,
  "quality": "measured",
  "source": "csv-upload"
}
```

质量状态至少包括：`measured`、`estimated`、`missing`、`invalid`。

## 6. DispatchPoint

```json
{
  "ts": "2026-09-20T08:00:00+08:00",
  "action": "discharge",
  "power_kw": 500.0,
  "soc_start": 0.90,
  "soc_end": 0.76,
  "grid_import_kw": 620.0,
  "grid_export_kw": 0.0
}
```

MVP 约定：`power_kw` 始终为非负值，充放电方向通过 `action` 表示；禁止反送电时 `grid_export_kw` 必须为 0。

## 7. 数据库初步表

| 表 | 用途 |
|---|---|
| sites | 站点基础信息 |
| tariffs | 电价模板 |
| measurements | 原始或清洗后的测量点 |
| forecast_runs | 预测请求和版本 |
| forecast_points | 预测时间序列 |
| dispatch_runs | 调度请求和结果 |
| dispatch_points | 调度时间序列 |
| sizing_runs | 容量规划结果 |
| report_jobs | 报告任务状态 |

## 8. 关键状态

报告和计算任务统一使用：

```text
pending → running → succeeded
                   ↘ failed
```

失败状态必须保存可读错误原因和可定位的内部错误标识。

## 9. MQ-RNN 电价扩展（2026-10-21）

测量类型新增 `price`，表示电能量部分，单位严格为 CNY/kWh，允许负电价。
原生频率为 15 或 60 分钟。`ForecastOptions` 新增 `price_model`、
`price_series_id`。`price_num_samples`、`price_seed` 仅供旧 DeepAR 基线使用。预测结果中的价格已加一次
`transmission_and_fund`，与入库目标口径有明确区别。详见 [MQ-RNN 电价适配](15-price-mqrnn.md)。
