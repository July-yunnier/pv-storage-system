# API 契约（初稿）

## 1. 通用约定

- Base URL：`/api/v1`
- 协议：REST + JSON
- 编码：UTF-8
- 时间：ISO 8601 带时区
- 响应：成功返回资源对象或任务对象，失败返回统一错误结构
- 空值：`null`，不能用 0 代表缺失

统一错误格式：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "resolution_minutes must be 15",
    "details": [
      {"field": "resolution_minutes", "reason": "unsupported value"}
    ],
    "request_id": "req_01"
  }
}
```

## 2. 站点接口

### `POST /sites`

创建站点。

### `GET /sites`

查询站点列表。

### `GET /sites/{site_id}`

获取站点详情。

### `PATCH /sites/{site_id}`

修改站点配置。修改需要记录更新时间，未来可增加配置版本。

## 3. 测量数据

### `POST /sites/{site_id}/measurements`

```json
{
  "data_type": "pv_power",
  "unit": "kW",
  "resolution_minutes": 15,
  "points": [
    {"ts": "2026-09-20T00:00:00+08:00", "value": 0.0},
    {"ts": "2026-09-20T12:00:00+08:00", "value": 623.4}
  ]
}
```

允许的 `data_type`：`pv_power`、`load_power`、`storage_soc`、`grid_power`。

## 4. 电价接口

### `POST /tariffs`

创建或更新电价模板。

### `GET /tariffs`

查询电价模板。

### `GET /sites/{site_id}/tariff-points`

返回指定时间范围内的统一价格点。

## 5. 天气接口

### `GET /sites/{site_id}/weather`

根据站点经纬度返回天气数据。外部天气源需要通过 Adapter 封装，接口不能把 Open-Meteo 的具体字段直接暴露成优化器依赖。

## 6. 预测接口

### `POST /forecast`

```json
{
  "site_id": "S001",
  "targets": ["pv_power", "load_power", "price"],
  "horizon_hours": 72,
  "resolution_minutes": 15,
  "issued_at": "2026-09-19T06:00:00+08:00",
  "options": {
    "include_uncertainty": true,
    "quantiles": [0.1, 0.5, 0.9]
  }
}
```

返回的每个 target 必须包含 `unit`、`model`、`points`，可选包含 `accuracy`、`p10`、`p90`。

## 7. 调度接口

### `POST /storage/schedule`

输入至少包含：站点、日期、预测曲线、初始 SOC、策略版本和约束选项。

输出至少包含：策略、预期收益、基准收益、完整调度点、实际使用约束和计算版本。

```json
{
  "site_id": "S001",
  "date": "2026-09-20",
  "initial_soc": 0.50,
  "strategy": "rule_v1"
}
```

## 8. 容量配置接口

### `POST /storage/sizing`

输入包含站点、代表日或全年代理日、候选容量范围、财务假设和优化选项。

输出包含：推荐配置、经济性、敏感性和 assumptions。

## 9. 报告接口

### `POST /reports`

创建报告任务，返回 `report_id` 和 `status`。

### `GET /reports/{report_id}`

查询状态和下载信息。

## 10. 版本与幂等

- 上传数据请求建议支持 `Idempotency-Key`。
- Provider、预测模型和优化器都必须返回名称与版本。
- 调度结果必须保存输入摘要，保证结果可追溯。

## 11. MQ-RNN 电价扩展（2026-10-21）

`POST /forecast` 新增 `options.price_model=auto|tou|mqrnn|deepar`，
`price_series_id`（默认 site_id）。保留的采样参数仅用于显式旧 DeepAR 基线。`auto` 对 TOU 展开政策价、对现货使用 MQ-RNN；
预测支持日前 24/48/72 小时，缺少模型或价格上下文返回 422。
价格测量新增 `data_type=price`、`unit=CNY/kWh`，允许负数，仅含电能量部分。
CSV 上传新增可选表单 `resolution_minutes`（默认 15，小时价格可指定 60）。
完整请求及返回口径见 [MQ-RNN 电价适配](15-price-mqrnn.md)。
