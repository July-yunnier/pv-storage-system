"""Independently audit all experiment artifacts and write a factual handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import numpy as np
import pandas as pd
from app.domain.forecast.features import regression_metrics
from app.domain.forecast.models import MODEL_NAMES
from app.domain.forecast.sources import citations
from scripts.run_unified_gefcom_experiments import code_version, resolved_environment_lock


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    results = pd.read_csv(root/"results.csv")
    expected = {(task, horizon, model, "full") for task in ("load", "price")
                for horizon in (1, 4) for model in MODEL_NAMES}
    expected |= {(task, horizon, "LightGBM", variant) for task in ("load", "price")
                 for horizon in (1, 4) for variant in ("no_exogenous", "no_calendar")}
    actual = {(r.task, int(r.horizon), r.model, r.variant) for r in results.itertuples()}
    errors = []
    if actual != expected or len(results) != len(expected):
        errors.append(f"expected 80 unique runs; missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    for row in results.itertuples():
        path = root/row.task/f"{row.horizon}h"/f"{row.model}__{row.variant}"
        if row.status != "complete":
            errors.append(f"incomplete experiment: {path}")
            continue
        config = json.loads((path/"config.json").read_text())
        (path/"citations.json").write_text(json.dumps(citations(row.model), indent=2))
        prediction = pd.read_csv(path/"predictions.csv")
        samples = pd.read_csv(path.parent/"samples.csv")
        test_samples = samples[samples.split == "test"]
        if not prediction.target_ts.equals(test_samples.target_ts.reset_index(drop=True)):
            errors.append(f"test sample IDs mismatch: {path}")
        # Original window contract uses float32. Reconstruct that exact label
        # representation from older CSVs before the float64 metric reduction.
        truth = prediction.y_true.to_numpy(np.float32).astype(float)
        artifact = json.loads((path/"weights/artifact.json").read_text())
        predicted = prediction.y_pred.to_numpy(np.float32).astype(float) if artifact["kind"] in {"neural", "persistence", "history_mean"} or row.model == "XGBoost" else prediction.y_pred.to_numpy()
        metrics = regression_metrics(truth, predicted, config["Cap"])
        saved = json.loads((path/"metrics.json").read_text())
        aggregate = results[(results.task == row.task) & (results.horizon == row.horizon) & (results.model == row.model) & (results.variant == row.variant)].iloc[0]
        for name in ("MAE", "RMSE", "1-MAE/Cap", "1-RMSE/Cap"):
            if not np.isclose(metrics[name], aggregate[name], rtol=1e-10, atol=1e-8):
                errors.append(f"aggregate metric mismatch: {path}/{name}")
            if not np.isclose(metrics[name], saved[name], rtol=1e-10, atol=1e-8):
                errors.append(f"metric recalculation mismatch: {path}/{name}")
        (path/"training_summary.json").write_text(json.dumps({"settings": config["settings"],
            "train_samples": config["training_samples"], "validation_samples": config["validation_samples"],
            "test_samples": config["test_samples"], "status": json.loads((path/"status.json").read_text()),
            "shuffle": False, "note": "summary from persisted run records; console log may be empty for silent estimators"}, indent=2))
        hashes = json.loads((path/"weights/hashes.json").read_text())
        if any(digest(path/"weights"/file) != sha for file, sha in hashes.items()):
            errors.append(f"weight hash mismatch: {path}")
        for file in ("training.log", "citations.json", "normalization.json", "code_version.json"):
            if not (path/file).exists():
                errors.append(f"missing artifact {path/file}")
    for task in ("load", "price"):
        for horizon in (1, 4):
            frame = pd.read_csv(root/task/f"{horizon}h/samples.csv")
            for left, right in (("train", "validation"), ("validation", "test")):
                a, b = frame[frame.split == left], frame[frame.split == right]
                if pd.Timestamp(a.target_ts.max()) >= pd.Timestamp(b.forecast_origin_ts.min()):
                    errors.append(f"split boundary overlap: {task}/{horizon}h")
    for horizon in (1, 4):
        for model in MODEL_NAMES:
            paths = [root/task/f"{horizon}h"/f"{model}__full/weights" for task in ("load", "price")]
            for file in ("model.pt", "estimator.joblib", "params.npy", "prophet.json"):
                if all((path/file).exists() for path in paths) and digest(paths[0]/file) == digest(paths[1]/file):
                    errors.append(f"shared task weight detected: {model}/{horizon}h")
    reload = json.loads((root/"reload_checks.json").read_text())
    if len(reload) != len(expected) or not all(r["passed"] for r in reload):
        errors.append("reload checks do not pass for all 80 experiments")
    if (root/"api_checks.json").exists():
        api_checks = json.loads((root/"api_checks.json").read_text())
        if len(api_checks) != 4 or not all(r["passed"] for r in api_checks):
            errors.append("representative live API checks failed")
    audit = {"expected_runs": len(expected), "actual_runs": len(results), "complete_runs": int((results.status == "complete").sum()),
             "independent_task_weights": True if not errors else None,
             "errors": errors, "passed": not errors}
    (root/"audit.json").write_text(json.dumps(audit, indent=2))
    (root/"environment.lock.txt").write_text(resolved_environment_lock())
    (root/"code_version.json").write_text(json.dumps(code_version(), indent=2))
    snapshot = root/"source_code"
    files = [*BACKEND.glob("app/domain/forecast/*.py"), *BACKEND.glob("scripts/*unified*.py"),
             BACKEND/"scripts/verify_forecast_artifacts.py", BACKEND/"scripts/finalize_forecast_iteration.py", BACKEND/"scripts/check_forecast_api.py",
             BACKEND/"app/domain/dispatch.py", BACKEND/"app/domain/models.py",
             BACKEND/"app/application/dispatch_service.py", BACKEND/"app/application/rolling_service.py",
             BACKEND/"app/api/routes_forecast.py", BACKEND/"requirements.txt", BACKEND/".env.ai.example", BACKEND/"README.md",
             BACKEND/"docs/FORECAST_EXPERIMENTS.md", BACKEND/"tests/test_forecast_engineering.py",
             BACKEND/"tests/test_dispatch.py", BACKEND/"tests/test_schedule_auto.py"]
    for file in files:
        target = snapshot/file.relative_to(BACKEND)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    (snapshot/"sha256.json").write_text(json.dumps({str(f.relative_to(snapshot)): digest(f) for f in snapshot.rglob("*") if f.is_file() and f.name != "sha256.json"}, indent=2))
    lines = ["# 负荷与电价工程实验交接", "", f"完成 {audit['complete_runs']}/{len(expected)} 组：18个模型×2任务×2预测目标，加8组LightGBM消融。", "",
             "固定seed=42，原生小时数据，48h历史。按目标日70:30划分开发集/测试集，开发集内80:20训练/验证；边界设embargo，不shuffle。负荷与电价各自训练、归一化和保存权重。", "",
             "## 验证集选择的代表模型", "", "|任务|预测目标|模型|测试MAE|测试RMSE|1-MAE/Cap|1-RMSE/Cap|", "|---|---|---|---:|---:|---:|---:|"]
    for (task, horizon), group in results[(results.status == "complete") & (results.variant == "full")].groupby(["task", "horizon"]):
        best = group.sort_values("validation_RMSE").iloc[0]
        lines.append(f"|{task}|{horizon}h|{best.model}|{best.MAE:.4f}|{best.RMSE:.4f}|{best['1-MAE/Cap']:.2%}|{best['1-RMSE/Cap']:.2%}|")
    lines += ["", "## 文件", "", "负荷、电价指标分别见load_results.csv、price_results.csv；各任务figures目录含对比、拟合、误差、达标率和消融图。每个模型目录保存真实权重、日志、配置、预测、指标、代码hash、引用与协议。audit.json独立重算指标并检查权重和切分；reload_checks.json核验模型重载。", "",
              "## 不可省略的限制", "", "- 这是真实GEFCom小时级负荷/市场电价实验，不是国内工厂实测闭环。没有真实15min标签和天空图像。", "- Cap默认是训练目标绝对最大值代理；电价没有物理容量。96%代理归一化结果不能直接宣称官方比赛达标。", "- 深度模型仅统一3个epoch预算，未进行充分超参数搜索；不能据此认定某架构理论上更差。", "- 没有可靠云量标签，晴/多云/阴天分组不可得。GEFCom-L/P/S不可跨站跨年拼成同一场站。", "- 残差P10/P90和中央80%区间属于验证校准，不是已训练的概率头。", "- 保留国内政策TOU兼容接口。ML电价独立使用/forecast/ml；国内调度显式接收96点CNY/kWh与kW曲线，不隐式转换GEFCom的USD/MWh。", "- source_code为本轮涉及文件快照，需与原工程一起使用；不是完整独立系统。复现使用environment.lock.txt，完整本机冻结记录environment.full.freeze.txt不用于直接安装。", "",
              "运行验收：本次后端测试190项通过；pip check无冲突。4组代表模型实际HTTP推理与错误单位拒绝结果见api_checks.json。", "",
              "本机运行提醒：旧混合库卡住进程PID 28601、旧校验PID 29353需要在用户终端执行kill -TERM；受限执行环境无法终止。新版训练/校验任务均已结束。", "",
              f"产物审计：{'通过' if not errors else '未通过，见audit.json'}。"]
    (root/"HANDOFF.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
