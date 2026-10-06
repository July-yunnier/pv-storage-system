# 负荷与电价模型迭代

## 代码布局

`app/domain/models.py` 保留 HTTP API 数据定义。预测算法共用 `app/domain/forecast/models.py` 的模型工厂，负荷、电价分别创建实例、训练、归一化、保存权重。`load_forecast.py` 与 `price_forecast_service.py` 是同结构的独立推理入口，不能加载另一个任务的 checkpoint。

传统模型：SmartPersistence（本任务定义为起点功率/价格持续性）、HistoryMean、ARIMA、SARIMA、Prophet、SVR、LightGBM、XGBoost。

深度模型：LSTM、GRU、BiLSTM、TFT、Informer、Autoformer、PatchTST、DLinear、RLinear、NHiTS。后六种适用架构使用 NeuralForecast 官方网络前向函数；训练循环自建以严格禁用随机抽窗/打乱。RLinear 是 RevIN+线性映射的项目参考变体，不冒充官方预训练权重。

Informer 在48h短窗口设置 factor=48，使每个query参与注意力计算。这是短窗口密集极限配置，不声称体现长序列稀疏注意力的效率优势。修复默认推理随机抽query导致重载输出变化的问题；float32累加顺序仍可能产生微小舍入差异。

## 运行

### GEFCom验证早停与独立消融预实验（当前口径）

不要重新运行旧的 `--epochs 3` 粗筛，也不要把旧20轮预算称为已收敛。新脚本先验证集调参及逐项消融，冻结全部选择后再测试。负荷/电价分别保存权重与结果，电价无Cap。旧测试段已被查看，必须如实标注探索性历史暴露，不可称为未接触留出集。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-forecast/bin/python \
  scripts/run_audited_preexperiments.py \
  --data /path/to/gefcom2014_benchmark \
  --output /path/to/gefcom_preexperiment_convergence_v3 \
  --max-epochs 120 --patience 10 --resume

.venv-forecast/bin/python scripts/report_audited_preexperiments.py \
  --output /path/to/gefcom_preexperiment_convergence_v3
```

每轮日志实时保存，只有完成试验才可按哈希严格resume。新协议或新参数必须用独立目录。只有触发validation_early_stop才称达到所声明的早停条件，达到轮数上限标为max_budget_not_converged。最终报告由已完成的指标生成到输出目录REPORT.md。`docs/CONVERGED_PREEXPERIMENT_REPORT.md`仅作历史结果纠错记录；提交前清单见 `docs/PREEXPERIMENT_SELF_CHECKLIST.md`。以下旧粗筛命令仅保留历史文档，不再执行。

在 backend 目录，创建独立虚拟环境并安装完整依赖。不要修改系统 Python 或已有 `.venv-ai`。

```bash
python3 -m venv .venv-forecast
.venv-forecast/bin/python -m pip install -r requirements.txt
.venv-forecast/bin/python scripts/run_unified_gefcom_experiments.py \
  --data /path/to/gefcom2014_benchmark \
  --output data/forecast_experiments \
  --epochs 3 --batch-size 256 --ablations
MPLCONFIGDIR=/tmp/energy-mpl .venv-forecast/bin/python scripts/plot_unified_results.py \
  --output data/forecast_experiments
```

Windows 使用 `.venv-forecast\Scripts\python.exe`。实际依赖版本在 `environment.lock.txt` 中，复现本次结果用该文件而不是宽松版本范围。代码 SHA-256、Git commit 和未提交差异 hash 全部保存。`--resume` 校验训练配置、源数据 hash 和训练实现 hash 后才复用结果。

训练与需要原生库的模型推理均使用独立进程，防止树模型OpenMP与PyTorch原生库冲突，也保证拟合实例相互隔离。简单持续性/历史均值原生处理，其他推理worker超时120秒。`environment.lock.txt`仅锁定项目依赖闭包，针对本次Python/操作系统；`environment.full.freeze.txt`保留完整本机环境，可能包含Conda本机路径，不用于安装。

结束后验收：先运行 `scripts/verify_forecast_artifacts.py --output ...`，再运行 `scripts/finalize_forecast_iteration.py --output ...`。后者独立重算指标、检查80组矩阵/样本ID/切分/模型hash/跨任务权重，并保存HANDOFF.md与涉及文件的代码快照。

## 数据与实验口径

- GEFCom-L 与 GEFCom-P 独立实验，数据没有被拼成同一站点。市场电价不是国内政策 TOU 电价。
- 数据为原生小时级，预测目标为 t+1h、t+4h。GEFCom 不提供真实 15min 标签或天空图像，不做假插值。
- `raw.csv` 保存真实输入，`samples.csv` 保存固定窗口与切分。固定 seed=42，完整源时间网格日序划分开发集:test 为70:30；开发集内部 train:validation 为80:20。缺失窗口与边界起点排除后，有效样本数比例略有变化。
- 历史窗口48h；滞后为0/1/2/3/6/12/24/48/168h。所有输入只到预测起点。天气协变量只使用历史值，未把目标日天气实况当作 NWP。
- 归一化参数只拟合 train。深度网络每轮按原始时间顺序遍历全部训练窗口，不 shuffle。前半验证集选择最佳 epoch，后半验证集校准残差区间；测试集不参与选择。
- ARIMA/SARIMA 固定用训练尾部90天估计参数，固定参数对后续历史滚动过滤。此预算明确记在配置中，不把它写成全训练期估计。
- MAE、RMSE、1-MAE/Cap、1-RMSE/Cap 均保存。默认 Cap 是训练目标绝对最大值代理，不是已确认的装机容量；电价没有物理装机容量。代理归一化96%不构成官方比赛达标证据。可用 `--load-cap`、`--price-cap` 提供经比赛口径确认的分母。
- 日历是小时、星期、月份周期编码；没有伪造无法确认地区的节假日，也没有伪造湿度/云量。负荷与电价无可靠晴/多云/阴天标签，天气分组明确 unavailable。
- LightGBM 运行 `full/no_exogenous/no_calendar` 消融，使用相同样本 ID。网络结构不支持外生特征时按官方能力使用，不能声称全部模型完成天气融合。
- P10/P50/P90 为验证残差校准，不是模型内部概率头。另存中央80% split-conformal 区间、实际覆盖率和宽度。时间序列不满足交换性时，不承诺理论无条件覆盖。

## 原生模型推理

`POST /api/v1/forecast/ml` 接收 `task`（load或price）、`model`、`horizon_hours`（1或4）、`unit`、`forecast_origin`、至少169个连续小时 `history`（ts/value），以及与历史同长度的 `historical_exogenous`。外生变量按 checkpoint 中 `exog_names` 提供。返回一个 t+h 预测点与验证残差 P10/P90。

服务器用 `FORECAST_ARTIFACT_ROOT` 指向实验输出根目录。客户端不能传任意文件路径；只加载服务器信任的 checkpoint 并核验 SHA-256、任务、模型、频率、单位。不要加载来历不明的 pickle/joblib 权重。ARIMA推理使用调用方提供的完整历史过滤状态；截短历史时和离线全历史过滤可能略有差异。

现有 `/forecast` 的国内96点政策价保留。`/storage/schedule` 的 `price_model=ml` 必须同时传 `source=forecast` 与96个时间对齐的 `predicted_prices`，单位必须CNY/kWh；不可把GEFCom小时级USD/MWh结果直接接入。政策TOU路径是兼容模式，不假装是机器学习。

调度可同时传 `predicted_loads`、`predicted_pv`（每点ts/value，单位kW）替代旧服务，接入独立训练模型的整日曲线。传入这些曲线必须显式 `source=forecast`。电价每点沿用PricePoint结构：ts/price/energy_price，tier可选；所有曲线必须准确覆盖站点当地目标日00:00至23:45的96点网格。此入口不自行将GEFCom点预测转换为国内全天曲线。

## 调度

AI路径只使用传入预测曲线，前瞻窗口累计计算高峰储能 reserve；未来实测PV不得用于当前决策。高峰削峰优先，充电须满足效率后的价差条件、需量上限和PCS上限。能量不足时记录 `unmet_peak_shaving_slots`，不承诺硬性削峰一定可达。`reserve_soc` 是安全底线，并非可以在低价时提前放空的电量。

## 产物与协议

每个实验保存配置、训练日志、归一化、实际权重、权重hash、测试与验证预测CSV、指标JSON、代码版本、引用和协议。负荷、电价分别保存总表和图。`open_source_inventory.json` 自动保存安装依赖元数据；模型 `citations.json` 列出论文、官方库地址和协议。GEFCom数据的本地档案不能确认再分发协议，因此不得把代码开源协议套给数据。
