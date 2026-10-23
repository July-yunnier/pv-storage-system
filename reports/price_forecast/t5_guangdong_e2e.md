# T5 广东真实电价端到端核验

**状态：导入规范化与历史训练入口已验证；广东真实端到端验收未完成。**

## 来源核验

- 用户提供的[电查查](https://www.dianchacha.cn/home/)页面可访问，但其公开“电价”目录是燃煤基准价、销售电价等年度/政策数据；首页现货资讯展示的是山西日均摘要，没有可审计的广东逐时/15 分钟出清下载。
- 找到公开的[源电社区广东实时页面](https://yuandian.space/forecasts/fc-rt-guangdong/)。页面明确标注“日前 / 预测实时 / 实际实时”、数据版本 `vppforecast-province-rt/guangdong/2026-03-31`、文件名 `province_rt_holdout_guangdong_mar2026.csv`，单位元/MWh，24 个时点/日。
- 本次只归档页面公开明确标注的 `实际实时` 数组，不把预测数组当作标签。原始页面见 `t5_guangdong_data/raw/yuandian_fc-rt-guangdong_20260331.html`，SHA256：`8e8ace60fe17102430c1f3012dc589d4c1fe78989f60d5ba7141f02954d76761`。

## 可复核样本

提取 2026-03-31 00:00–23:00 Asia/Shanghai 共 24 点，原始单位元/MWh，转换后元/kWh：

- 最小值 0.475650，最大值 1.007120；均值 0.716805，中位数 0.754240。
- 转换 CSV：`t5_guangdong_data/yuandian_actual_rt_20260331.csv`，SHA256：`2bd84837f66f75f12878a591d8552c769092ad4d48e49e2dd6affe49ff5a31ef`。
- CSV 首行保留 `# 来源页面明确标注的“实际实时”曲线；原单位元/MWh；非合成示例`，避免与合成 fixture 混淆。

## 工程改动

`train_price_model.py` 新增 `--as-of`：接受必须带时区的 ISO 8601 截止时刻，按站点时区对齐整点，并把最终 `as_of` 写入 `manifest.json`。裸 datetime 会在打开数据库前拒绝，避免历史数据被错误地截到当前时间。文档已同步。

## 验证

- `tests/test_train_price_model.py tests/test_price_ingest.py`：**16 passed**。
- 数据文件通过导入格式检查，24 行、时间戳为本地无时区文本，需调用 `--tz Asia/Shanghai` 解释；金额转换为元/kWh 后统计如上。
- 完整 T5 训练验收未执行：公开可复核窗口只有 31 天，当前归档的一天样本不足训练入口的 `--min-days 45`，也不足默认 `--cal-days 45` 所需 90 天有效小时。没有生成生产工件、MAE 或 PICP80 结论。

## 后续输入

需要广东交易中心或其他可审计来源提供至少 45 天逐时/15 分钟原始现货序列，最好覆盖 90 天以上以启用默认校准；同时需明确日前/实时字段、节点或统一结算点口径、发布时间与时间戳含义。拿到完整 CSV 后可执行：

```bash
cd backend
python scripts/import_spot_price.py --site S-GD --csv <真实CSV> --profile guangdong --source guangdong_spot
python scripts/train_price_model.py --site S-GD --db data/storage.db --as-of 2026-03-31T23:00:00+08:00 --cal-days 45
```

当前结论仅支持“广东数据导入适配器与历史训练入口就绪”，不支持“广东真实端到端已完成”或“预测效果已验证”。
