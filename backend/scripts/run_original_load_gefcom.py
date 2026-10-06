"""Add an independently trained original-load adapter; preserve previous results."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import torch
from app.domain.forecast.features import build_windows, regression_metrics
from app.domain.forecast.load_gefcom_adapter import (
    GEFComEliteEnergyTaskNet, adapter_batch, adapter_predict, adapter_stats)
from app.domain.forecast.quantile import interval_coverage, pinball_numpy, set_seed
from scripts.run_converged_preexperiments import empirical_interval, source, sha


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def fit(data, args, folder):
    set_seed(42)
    stats = adapter_stats(data)
    tr, va = data.indices("train"), data.indices("validation")
    model = GEFComEliteEnergyTaskNet(len(stats["history_mean"]), len(stats["target_mean"]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=1e-4)
    train_batch = adapter_batch(data, stats, tr)
    labels = torch.as_tensor((data.y[tr, -1] - stats["load_mean"]) / stats["load_std"])
    best, stale, best_state, log, best_epoch = float("inf"), 0, None, [], None
    stop_reason = "max_budget_not_converged"
    started = time.monotonic()
    for epoch in range(1, args.max_epochs + 1):
        model.train(); train_sum = 0.0
        for start in range(0, len(tr), args.batch_size):
            batch = [v[start:start + args.batch_size] for v in train_batch]
            optimizer.zero_grad()
            prediction = model(*batch)[0][:, -1]
            loss = torch.nn.functional.mse_loss(prediction, labels[start:start + args.batch_size])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); train_sum += float(loss.detach()) * len(prediction)
        vp = adapter_predict(model, data, stats, va, args.batch_size)
        validation_mse = float(np.mean((vp - data.y[va, -1])**2))
        record = {"epoch": epoch, "train_mse_zscore": train_sum / len(tr),
                  "validation_mse": validation_mse,
                  "validation_mae": float(np.mean(np.abs(vp - data.y[va, -1]))),
                  "elapsed_seconds": time.monotonic() - started}
        log.append(record)
        if validation_mse < best - .001:
            best, stale, best_epoch = validation_mse, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, folder / "model.pt")
        else:
            stale += 1
        save(folder / "training_log.json", log)
        print(f"{data.horizon}h epoch={epoch} val_MSE={validation_mse:.6f} best={best:.6f} stale={stale} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        if stale >= args.patience:
            stop_reason = "validation_early_stop"
            break
    if best_state is None:
        raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best_state)
    artifact = {"kind": "original_load_gefcom_point_adapter", "task": "load",
                "model": "EliteEnergyTaskNet_hourly_unknown_mes", "horizon_hours": data.horizon,
                "history_hours": data.history_size, "hidden_dim": 128,
                "normalization": stats, "best_validation_mse": best, "best_epoch": best_epoch,
                "epochs_completed": len(log), "stop_reason": stop_reason,
                "elapsed_seconds": time.monotonic() - started, "weight_sha256": sha(folder / "model.pt"),
                "workshop_head": "frozen_unlabelled_not_evaluated", "mes": "unknown_zero_embedding_frozen",
                "future_features": "calendar_only_no_future_observed_weather",
                "loss": "last_horizon_total_load_MSE_no_auxiliary_or_consistency_loss"}
    save(folder / "artifact.json", artifact)
    return artifact


def compare(existing, data, prediction):
    keys = ["forecast_origin_ts", "target_ts"]
    other = pd.read_csv(existing)
    current = data.rows.iloc[data.indices("test")][keys].copy()
    for frame in (other, current):
        for key in keys:
            frame[key] = pd.to_datetime(frame[key], utc=True)
    if len(other) != len(current) or not other[keys].reset_index(drop=True).equals(current[keys].reset_index(drop=True)):
        raise ValueError("previous prediction timestamps differ; no silently changed comparison population")
    truth = data.y[data.indices("test"), -1]
    if not np.allclose(other.y_true, truth, atol=1e-4, rtol=0):
        raise ValueError("previous test labels differ")
    return other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 10 or args.patience < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("invalid full-convergence settings; coarse screening forbidden")
    if args.output.resolve() == args.previous.resolve() or args.previous.resolve() in args.output.resolve().parents:
        parser.error("output must be separate from preserved results")
    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in args.previous.rglob("*") if p.is_file())
    before = {str(p.relative_to(args.previous)): sha(p) for p in existing}
    original = BACKEND / "app/domain/forecast/load_advanced.py"
    code = [original, Path(__file__), BACKEND / "app/domain/forecast/load_gefcom_adapter.py",
            BACKEND / "app/domain/forecast/features.py", BACKEND / "scripts/run_converged_preexperiments.py"]
    protocol = {"scope": "GEFCom exploratory adapter comparison ONLY; main experiment NOT started",
                "seed": 42, "shuffle": False, "historical_test_exposure": True,
                "original_model": "load_advanced.EliteEnergyTaskNet; inherits original parameter modules",
                "preserved": ["two_layer_LSTM_hidden128_dropout0.3", "Bahdanau_attention", "target_decoder_GELU_dropout0.2", "total_and_workshop_heads", "MES_embedding_dim8", "MSE_point_objective"],
                "adaptations": ["48h_history_for_matched_comparison_not_original30days", "hourly_horizon1_or4_not_96slot_day", "all_MES_unknown_zero_frozen", "workshop_frozen_no_fabricated_labels", "calendar_only_future_features", "train_only_zscore", "validation_early_stop"],
                "interval": "second_validation_half_residual_postcalibration_NOT_native_quantiles",
                "cap": "historical_training_max_statistical_proxy_NOT_physical_capacity",
                "settings": {k: getattr(args, k) for k in ("max_epochs", "patience", "learning_rate", "batch_size")},
                "previous_file_hashes": before, "code_sha256": {str(p.relative_to(BACKEND)): sha(p) for p in code},
                "data_sha256": {f"{h}h": sha(args.data / f"GEFCom2014-L/load/{h}h/all.csv") for h in (1, 4)},
                "pv": "not_loaded_or_modified", "dispatch": "not_executed", "online_model": "unchanged"}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (args.output / "protocol.json").exists():
        if not args.resume or json.loads((args.output / "protocol.json").read_text()) != protocol:
            raise ValueError("existing output/protocol differs; use new output directory")
    save(args.output / "protocol.json", protocol)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (args.output / "environment.lock").write_text(f"# Python {sys.version}\n# {platform.platform()}\n" + "\n".join(packages) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND, capture_output=True, text=True)
    save(args.output / "code_version.json", {"base_commit": git.stdout.strip(), "uncommitted_extension": True, "authoritative_sha256": protocol["code_sha256"]})
    raw, _ = source(args.data, "load")
    entries = []
    for horizon in (1, 4):
        data = build_windows(raw, "load", horizon, 48)
        prior_manifest = pd.read_csv(args.previous / f"load/{horizon}h/split_manifest.csv")
        for col in ("forecast_origin_ts", "target_ts"):
            prior_manifest[col] = pd.to_datetime(prior_manifest[col], utc=True)
        if not prior_manifest[["forecast_origin_ts", "target_ts", "split"]].equals(data.rows[["forecast_origin_ts", "target_ts", "split"]]):
            raise ValueError("split differs from preserved experiment")
        folder = args.output / f"load/{horizon}h"
        folder.mkdir(parents=True, exist_ok=True)
        data.rows.to_csv(folder / "split_manifest.csv", index=False)
        if (folder / "artifact.json").exists() and args.resume:
            info = json.loads((folder / "artifact.json").read_text())
            if sha(folder / "model.pt") != info["weight_sha256"]:
                raise ValueError("weight hash mismatch")
        else:
            info = fit(data, args, folder)
        entries.append({"horizon": horizon, "folder": str(folder), "artifact": info})
    save(args.output / "selection_frozen.json", {"fingerprint": fingerprint, "entries": entries,
         "basis": "validation_MSE_only; all_horizons_frozen_before_test; no_test_based_tuning"})
    results, audit = [], []
    for entry in entries:
        horizon = entry["horizon"]; folder = Path(entry["folder"]); info = entry["artifact"]
        data = build_windows(raw, "load", horizon, 48)
        tr, va, te = (data.indices(s) for s in ("train", "validation", "test"))
        model = GEFComEliteEnergyTaskNet(len(info["normalization"]["history_mean"]), len(info["normalization"]["target_mean"]))
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        vp = adapter_predict(model, data, info["normalization"], va, args.batch_size)
        point = adapter_predict(model, data, info["normalization"], te, args.batch_size)
        lo, hi = empirical_interval(data.y[va, -1], vp)
        qs = np.column_stack([point + lo, point, point + hi])
        cutoff = data.rows.iloc[tr].target_ts.max()
        cap = float(raw.loc[raw.ts <= cutoff, "load"].max())
        metrics = {**regression_metrics(data.y[te, -1], point, cap),
                   "task": "load", "horizon": horizon, "variant": "Original_load_GEFCom_adapter",
                   "Pinball": pinball_numpy(data.y[te, -1], qs),
                   "interval_coverage_80": interval_coverage(data.y[te, -1], qs[:, 0], qs[:, 2]),
                   "interval_type": "validation_residual_postcalibration_NOT_native",
                   "point_not_guaranteed_median": True, "residual_offsets": [lo, hi],
                   "interval_mean_width": float(hi-lo), "cap_is_physical": False,
                   "normalization": "train_only_zscore", "stop_reason": info["stop_reason"],
                   "best_epoch": info["best_epoch"], "epochs_completed": info["epochs_completed"],
                   "scope": "pre-experiment_ONLY", "historical_test_exposure": True}
        save(folder / "metrics.json", metrics)
        preds = data.rows.iloc[te][["forecast_origin_ts", "target_ts"]].copy()
        preds["y_true"] = data.y[te, -1]
        preds["p10"], preds["point"], preds["p90"] = qs[:, 0], point, qs[:, 2]
        preds.to_csv(folder / "predictions.csv", index=False)
        reload = adapter_predict(model, data, info["normalization"], te[:min(512, len(te))], args.batch_size)
        if not np.allclose(reload, point[:len(reload)], atol=1e-4, rtol=0):
            raise ValueError("reload prediction differs")
        audit.append({"horizon": horizon, "reload_finite": True, "hash": sha(folder / "model.pt"), "test_count": len(te), "same_population_as_previous": True})
        results.append(metrics)
        for label in ("Persistence", "ARIMA", "LSTM_baseline", "TCN_baseline", "TCN_optimized"):
            other = compare(args.previous / f"load/{horizon}h/evaluation/{label}/predictions.csv", data, point)
            oldmetrics = json.loads((args.previous / f"load/{horizon}h/evaluation/{label}/metrics.json").read_text())
            recalculated = regression_metrics(other.y_true.to_numpy(), other.p50.to_numpy(), cap)
            if abs(recalculated["MAE"]-oldmetrics["MAE"]) > 1e-5:
                raise ValueError("old metrics fail independent recomputation")
            results.append({**oldmetrics, "variant": label})
    after = {str(p.relative_to(args.previous)): sha(p) for p in existing}
    if before != after:
        raise ValueError("preserved results changed")
    pd.DataFrame(results).to_csv(args.output / "load_comparison.csv", index=False)
    save(args.output / "completion.json", {"previous_results_unchanged": True, "experiments": 2, "audit": audit,
         "scope": "pre-experiment_only", "budget_limited": [e["horizon"] for e in entries if e["artifact"]["stop_reason"] != "validation_early_stop"],
         "main_experiment": "not_started", "online_weights_replaced": False, "pv_modified": False})
    lines = ["# 原负荷模型 GEFCom 适配对照", "", "本部分仅为 GEFCom 预实验，主实验尚未开展。既有结果未改动，在线权重未替换，光伏未改动。", "",
             "## 模型口径", "", "继承原 EliteEnergyTaskNet 的两层128维LSTM、Bahdanau注意力、目标解码器与总厂/车间头。MES固定unknown零嵌入；无车间标签，车间头冻结且不评价。原30天/15分钟/96点任务改为与既有预实验一致的48小时历史和小时级1h/4h目标。未来仅输入可提前计算的日历，不输入未来天气实况。不属于完全不变的原工厂模型，也不构成纯数据替换的因果消融。", "",
             "## 指标对照", "", "|时距|方案|MAE|RMSE|Pinball|80%覆盖率|1-MAE/Cap|1-RMSE/Cap|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda r: (r["horizon"], r["variant"])):
        lines.append(f"|{r['horizon']}h|{r['variant']}|{r['MAE']:.4f}|{r['RMSE']:.4f}|{r['Pinball']:.4f}|{r['interval_coverage_80']:.2%}|{r['1-MAE/Cap']:.2%}|{r['1-RMSE/Cap']:.2%}|")
    lines += ["", "## 局限与解释", "", "原适配模型为MSE点预测，区间使用验证集后半段残差后校准，不是原生概率头；中心值不保证是条件中位数。验证集已用于早停，残差校准并非独立校准集，不保证80%覆盖。LSTM/TCN为原生Pinball头，概率评价不可忽略这一差异。负荷Cap为历史训练最大值统计代理，不代表物理容量或正式96%验收。单位保持GEFCom源单位，不转换为目标工厂kW。", "", "既有测试集曾在历史探索中暴露，本次新增实验不能恢复一个从未见过的测试集。所有选择仅依据验证集，并在测试评估前冻结；结果仍为探索性预实验。真实数据不能证明目标工厂的MES效果或落地收益。电价实验未改动，系统闭环未运行。", "", "## 训练状态", ""]
    for e in entries:
        i = e["artifact"]
        lines.append(f"- {e['horizon']}h：最佳epoch={i['best_epoch']}，已运行{i['epochs_completed']}轮，停止原因={i['stop_reason']}。")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("DONE: original adapter comparison; prior results unchanged; pre-experiment only", flush=True)


if __name__ == "__main__":
    main()
