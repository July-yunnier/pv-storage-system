# 电价事件数据：导入、时点回放与成对实验（T4）

当前状态：工程链路已具备，真实数据验收未完成。已有一条可核验的寒潮报道；没有完整检修、预警覆盖、连续燃料指数与同期广东现货价格，不能宣称事件特征改善了精度。实际审计见 [T4 报告](../reports/price_forecast/t4_event_calendar.md)。

## 数据契约

V2 YAML 顶层字段为 `schema_version: 2`、`data_kind: real|example`、`market`（如 `guangdong`）、`timezone`（如 `Asia/Shanghai`）、`records`。`name`、`imported_at` 可选。所有记录必须有：

| 字段 | 规则 |
|---|---|
| `record_type` / `record_id` | `event`、`coverage` 或 `fuel`；稳定的逻辑 ID |
| `published_at` | 带时区的公开发布时间；不得用事件开始时间或事后采集时间冒充 |
| `source_url` | 原始 HTTP(S) 公告/报道地址 |
| `source_file` / `source_sha256` | 包目录内的相对证据路径及 SHA256；不允许越界路径 |

哈希只校验归档内容是否改变，不能代替对来源真实性、区域、完整性和发布时间的人工核验。二手报道采用报道发布时间，并披露其范围限制。修订沿用同一 ID，增加新的 `published_at`；已发布版本不可覆盖。燃料的同一观测不允许使用多个 ID。

| 类型 | 专属字段 | 含义 |
|---|---|---|
| `event` | `kind`、`start`、`end`、`severity`、`status` | kind 为 maintenance/cold_wave/heat_wave；半开区间 `[start,end)`；severity 有限且在 0–1；status 为 active/cancelled |
| `coverage` | `kind`、`start`、`end`、`status` | 对某类别、某区间的来源完整性声明；必须有证据，不能由“文件里没有事件”推断 |
| `fuel` | `series_id`、`observation_ts`、`index`、`valid_until` | 正的有限指数；`observation_ts <= published_at < valid_until`；整个包只能使用一个一致口径的指数序列 |

每个时间字段都必须显式带时区，例如 `2024-01-19T21:54:00+08:00`。未知值保留缺失，不补零，不虚构 coverage 或未来发布时间。价格数据统一元/kWh；燃料指数为同一序列内的无量纲比较基础，不直接当作电价。

## 定期导入

将经核验的公告规范化为 CSV，字段为上表字段的并集；每行只填写所属类型需要的字段。`source_file` 在输入时相对于 `--source-root`；`source_sha256` 可由导入器生成。`#` 开头的整行可作注释，合成输入首行必须标注“示例”。已提交的真实输入可参考 [records.csv](../reports/price_forecast/t4_event_data/records.csv)。

从仓库根目录执行：

```bash
python backend/scripts/import_event_calendar.py \
  --records reports/price_forecast/t4_event_data/records.csv \
  --source-root reports/price_forecast/t4_event_data/raw \
  --output /absolute/data/events/guangdong/calendar.yaml \
  --market guangdong --timezone Asia/Shanghai --data-kind real
```

同一命令可在每次收到新导出后重复执行；也可接入现有每日作业，在核验后的 CSV 更新完成后调用。本变更不创建系统定时任务，不包含自动抓取器。导入先验证全部输入、归档到 `sources/<sha256>.<suffix>`，再原子替换 YAML；重复相同版本幂等，冲突版本拒绝。既有包的 market/timezone/data_kind 不允许改变。请保留每次标准输出的导入报告。

在线用 `PRICE_EVENT_CALENDAR=/absolute/data/events/guangdong/calendar.yaml` 显式启用；该变量作用于当前服务进程，应仅配置给相同市场的站点。每次请求重新加载并校验归档哈希，定期导入后无需重启。未设置变量时不再自动加载仓库的广东合成示例。V1 配置仍兼容，但仅作为演示数据，不得用于真实效果验收。

训练入口增加 `--event-calendar /absolute/.../calendar.yaml`；工件 manifest 保存日历摘要、日历哈希和来源哈希。既有预测 API、响应字段与模型工件加载协议不变；7 个事件特征的名称及顺序全部保持。旧工件不会因导入事件自动获得新效果，需重新训练。

## 特征与可用时点

调用 `EventCalendar.features(targets, issued_at=...)`，targets 和 issued_at 均带时区。每个训练 origin 只用 `published_at <= origin` 的最新版本；取消记录按发布时间生效，不会改写过去。多日递归预测始终冻结在原始发报时刻，不把未来一天的新公告提前纳入。

- `in_maintenance`：已知窗口内为 1；寒潮/高温取已知事件最大 severity。预警等级到 severity 的映射是编码约定，必须在 note 中记录，并不等于实测冲击强度。
- 没有活动事件时，只有有证据的 coverage 覆盖目标时刻才取 0；否则为 NaN。单条已知事件不能证明其他类别或整个来源完整。
- `days_to_maintenance`：窗口内为 0；否则距已知下一次检修的本地自然日数，上限 60。没有已知未来检修时，需完整覆盖未来 60 天才返回 60，否则为 NaN。
- `event_pressure`：三类事件都已知时，按 0.6/0.9/0.8 加权并截到 1；任一类别未知则为 NaN。
- 燃料只使用发报时已经发布的最近观测，且只保持到明确的 `valid_until`，不使用预测目标日之后才发布的值。近 365 天至少 30 个不同观测才计算经验分位；七日变化率用距最新观测至少七个日历日前的最近有效观测作基数，不是“第八条记录”。过期或缺少基数则为 NaN。

先审计真实日期，不需要价格数据：

```bash
python backend/scripts/audit_event_calendar.py \
  --calendar reports/price_forecast/t4_event_data/calendar.yaml \
  --start 2024-01-20 --days 2 --output /absolute/results/event-audit
```

输出带时区的逐小时特征与每列缺失数，按前一本地日 23:00 发报。退出 3 表示数据不完整（正常的拒绝信号），0 仅表示特征完整，均不代表效果已验证。

## 成对离线协议

输入价格 CSV 列为 `ts,price`，必须连续、升序、无重复、整小时、价格有限，允许负电价。`ts` 必须带时区。价格元数据 JSON 必须包含：`data_kind: real`、与日历一致的 `market` / `timezone`、`unit: CNY/kWh`、`price_kind: spot`、`price_sha256`、`source_url`、`source_file`、`source_sha256`。证据文件位于元数据目录内。广东目录分时电价截图和 GEFCom/PJM 价格不能替代同期广东现货数据。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python backend/scripts/run_event_ablation.py \
  --prices /absolute/data/spot_hourly.csv --price-meta /absolute/data/spot_meta.json \
  --calendar /absolute/data/events/guangdong/calendar.yaml \
  --output /absolute/results/events-new-run --start 2024-06-01 \
  --days 60 --train-days 120 --cal-days 45 --rounds 300 --stride 3 \
  --seed 42 --bootstrap 2000 --block-days 7
```

上述日期仅展示调用方式，不表示已有这些真实数据。输出目录必须为空，避免旧结果混入失败运行。真实模式至少 30 个连续评价日，每日必须有 24 个不同本地小时，不接受 DST 异常日。逐日重新拟合同一 LightGBM、校准窗口和种子，唯一消融为是否启用事件。拟合和校准只用发报前真值，目标日价格先屏蔽；预检核查所有实际训练/校准 origin 和预测 origin 的 7 个事件特征，缺失时先写 `feature_audit.json` 并拒绝训练。

结果为 predictions.csv、by_day.csv、summary.csv、feature_audit.json、run_meta.json。MAE、pinball、Winkler 单位为元/kWh，PICP80 为比例。summary 差值统一为“有事件 − 无事件”，95% CI 是固定种子的成对、循环移动日块 bootstrap；块长默认 7 天，2000 次，评价窗口至少容纳两个块。MAE/pinball/Winkler 负值通常更好；PICP80 应结合名义 0.80、区间宽度与 Winkler 解读，不能以越高越好评价。CI 只对所选窗口和协议的日均差异提供不确定性估计，不支持因果或跨市场结论；输出不自动把 `effect_verified` 设为 true。

实现沿用 CPU LightGBM、单线程拟合，没有新增依赖或 GPU 组件。默认窗口按天顺序训练，bootstrap 逐次计算，不保留所有模型。合成 CLI 冒烟的实测资源见 T4 报告；完整真实规模的 CPU / 8 GB 验证仍待真实数据。`--example-smoke` 只允许 example 价格和日历，结果明确标为 `synthetic_engineering_only`，不得混入正式结论。
