# 负荷与电价离线预实验

本目录仅整合不改动在线业务的代码。来源为`feat/product-mvp-impl`的`c6d0832`，从旧`product/backend/`迁至当前`backend/`，不恢复重复镜像目录。

## 运行范围

正式主实验尚未开展；GEFCom-L和GEFCom-P独立训练/权重，不拼站点、不按未知云量分组。原生小时数据只评估1h和4h，不插值伪造15min目标。负荷Cap是训练历史最大值统计代理，电价没有物理Cap。测试段的历史暴露必须披露。

```bash
cd backend
python -m pip install -r requirements.txt -r requirements-forecast.txt
python scripts/run_audited_preexperiments.py --help
python scripts/run_audited_preexperiments.py --data /path/to/prepared_gefcom --output data/forecast_experiments/audited --max-epochs 120 --patience 10
python scripts/audit_preexperiment_artifacts.py --output data/forecast_experiments/audited
python scripts/report_audited_preexperiments.py --output data/forecast_experiments/audited
```

数据路径为参数，不含用户个人目录绑定。需要已有整理格式：`GEFCom2014-L/load/{1h,4h}/all.csv`与`GEFCom2014-P/price/{1h,4h}/all.csv`。原始数据、已训练权重和本机输出不在本次提交内，不能声称clone后已具备这些产物。LightGBM的系统OpenMP依赖按平台另外安装。

当前主流程为`run_audited_preexperiments.py`：先验证调参/早停与消融，冻结选择后再最终测试。保存seed=42、数据与代码哈希、environment.lock、训练日志、权重、预测与指标。达到预算上限不是确认收敛。不再运行epoch=3粗筛。

`run_converged_preexperiments.py`保留历史辅助函数供主流程导入，不是推荐独立入口；历史20轮结果不能称为已收敛。`plot_unified_results.py`仅供历史通用结果绘图。本次不迁入旧`run_unified_gefcom_experiments.py`：它默认epoch=3且包含电价Cap接口，尚不满足当前科研约束；该版本保留在原分支，不重新运行。

`verify_frozen_pv_releases.py`需要另行提供光伏项目路径，只读推理，不改权重、不训练。该项目尚未合并进main，缺少原输入时不可运行，不会用其他模型代替。

## 本次未接入

未改动`app/api/routes_forecast.py`、`app/domain/models.py`、在线调度/回放/观测/光伏算法；未注册新的在线ML路由。新分位数模型权重也未自动接入在线API。在线融合及光伏发布包须另行确认与适配，见根目录整合清单。
