"""Build the Chinese report, scientific figures and selected model releases."""
from __future__ import annotations

import json
import shutil
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

from experiments.baselines import clear_sky_power, frozen_historical_mean
from experiments.metrics import evaluate
from experiments.optimize import build_features
from experiments.predict_optimized import predict_feature_matrices
from experiments.protocol import ROOT, dump_json, sample_indices, sha256

RUNS = ROOT / "runs/optimization_v2"
REPORT = ROOT / "reports/optimization_v2"
RELEASE = ROOT / "product/backend/data/model_releases/optimization_v2"
HORIZONS = (15, 60, 240)


def read(path):
    return json.loads(Path(path).read_text())


def table(frame):
    return frame.to_markdown(index=False, floatfmt=".3f")


def paired_day_bootstrap(base, chosen, cap, timezone):
    keys = ["issued_at", "horizon_minutes", "target_at", "actual_kw"]
    if not base[keys].equals(chosen[keys]):
        raise ValueError("paired cases differ")
    result = {}
    for h in HORIZONS:
        b, s = base[base.horizon_minutes == h], chosen[chosen.horizon_minutes == h]
        days = pd.to_datetime(s.issued_at, utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
        e = pd.DataFrame(dict(base=(b.predicted_kw.to_numpy()-b.actual_kw.to_numpy())**2,
                             selected=(s.predicted_kw.to_numpy()-s.actual_kw.to_numpy())**2,
                             n=np.ones(len(s)), day=days.to_numpy()))
        grouped = e.groupby("day")[["base", "selected", "n"]].sum().to_numpy()
        rng = np.random.default_rng(42)
        totals = grouped[rng.integers(0, len(grouped), size=(2000, len(grouped)))].sum(1)
        gain = (np.sqrt(totals[:, 0]/totals[:, 2])-np.sqrt(totals[:, 1]/totals[:, 2]))/cap*100
        result[str(h)] = dict(days=len(grouped), resamples=2000,
                             accuracy_gain_pp_ci95=np.quantile(gain, [.025, .975]).tolist())
    return result


def extra_baselines(directory, selected, config):
    dataset = directory.name.replace("_images", "")
    source = ROOT / "data/processed" / dataset
    frame = pd.read_pickle(source / "frame.pkl")
    manifest = config["dataset_manifest"]
    clock = pd.DatetimeIndex(frame.ts).tz_convert("UTC")
    issue = clock.get_indexer(pd.to_datetime(selected.issued_at, utc=True))
    target = clock.get_indexer(pd.to_datetime(selected.target_at, utc=True))
    if (issue < 0).any() or (target < 0).any():
        raise ValueError("missing baseline timestamps")
    means, _ = frozen_historical_mean(frame, config["split"]["train_end"])
    predictions = {"historical_mean": means[target]}
    cap = manifest["cap_kw"]
    if dataset == "skippd":
        predictions["pvlib_clear_sky_horizontal"] = clear_sky_power(frame, manifest)[target]
        predictions["persistence"] = frame.power_kw.to_numpy()[issue]
    else:
        cell = pvlib.temperature.faiman(frame.tsi.clip(lower=0), frame.temperature, 1.)
        physical = np.clip(np.asarray(pvlib.pvsystem.pvwatts_dc(frame.tsi.clip(lower=0), cell, cap, -.004)), 0, cap)
        predictions["pvlib_irradiance_persistence"] = physical[issue]
    for name, values in predictions.items():
        folder = directory / name
        folder.mkdir(exist_ok=True)
        output = selected.drop(columns=[c for c in ("p10_kw", "p50_kw", "p90_kw") if c in selected]).copy()
        output.predicted_kw = values
        output.to_csv(folder / "predictions.csv", index=False, float_format="%.8f")
        dump_json(folder / "metrics.json", evaluate(output, cap))
        dump_json(folder / "config.json", dict(config, model=name, fit="train only", physics="proxy; no target weather"))
        dump_json(folder / "weights.json", dict(formula=name, fit_before=config["split"]["train_end"]))
        dump_json(folder / "code_version.json", config["code"])
    return frame


def export_selected(directory, frame, config):
    selection = read(directory / "selection.json")
    weights = []
    for j, name in enumerate(selection["selected_per_horizon"]):
        if name == "validation_ensemble":
            weights.append(dict(zip(selection["choices"], selection["ensemble_weights"][j])))
        else:
            weights.append({name: 1.})
    # Keep even very small fitted weights: the release reproduces the frozen blend.
    needed = set().union(*(w.keys() for w in weights)) - {"reference_baseline"}
    estimators = {name: joblib.load(directory / (name + ".joblib")) for name in needed}
    _, slots = frozen_historical_mean(frame, config["split"]["train_end"])
    train = frame.ts < pd.Timestamp(config["split"]["train_end"])
    cap = config["dataset_manifest"]["cap_kw"]
    bundle = dict(format_version=1, dataset=directory.name, horizons_minutes=list(HORIZONS), cap_kw=cap,
                  feature_names=config["features"], estimators=estimators, weights_per_horizon=weights,
                  historical_slot_means_kw=slots, historical_slot_default_kw=float(frame.loc[train].power_kw.mean()),
                  residual_quantiles_kw=(np.asarray(selection["calibration_residual_quantiles"])*cap).tolist(),
                  training_config=config, selection=selection)
    sky = directory.name.startswith("skippd")
    indices, future = sample_indices(frame, config["split"], 30 if sky else 64, 1 if sky else 15)
    idx, partitions, matrices, names, baseline, _ = build_features(frame, indices, future,
                                      config["dataset_manifest"], config["split"], config["images"])
    test = partitions["test"]
    # Deterministic points span the entire test period, including endpoints.
    cases = test[np.unique(np.linspace(0, len(test)-1, min(1000, len(test))).astype(int))]
    expected = np.column_stack([np.load(directory / (name + "_internal.npz"))["prediction"][cases, j]
                               if name != "validation_ensemble" and name != "reference_baseline"
                               else baseline[cases, j] if name == "reference_baseline"
                               else sum(np.load(directory / (k + "_internal.npz"))["prediction"][cases, j] * w
                                        if k != "reference_baseline" else baseline[cases, j] * w
                                        for k, w in weights[j].items())
                               for j, name in enumerate(selection["selected_per_horizon"])]) * cap
    RELEASE.mkdir(parents=True, exist_ok=True)
    checkpoint = RELEASE / (directory.name + ".joblib")
    joblib.dump(bundle, checkpoint, compress=3)
    loaded = joblib.load(checkpoint)
    prediction, _ = predict_feature_matrices(loaded, [x[cases] for x in matrices], names, baseline[cases])
    np.testing.assert_allclose(prediction, expected, rtol=3e-6, atol=cap*3e-6)
    dump_json(RELEASE / (directory.name + ".json"), dict(
        checkpoint=checkpoint.name, sha256=sha256(checkpoint), bytes=checkpoint.stat().st_size,
        verified_reload_cases=len(cases), max_reload_error_kw=float(np.abs(prediction-expected).max()),
        feature_names=names, historical_slot_means_kw=slots,
        historical_slot_default_kw=bundle["historical_slot_default_kw"], weights_per_horizon=weights,
        selection=selection, config=config, inference_api="experiments.predict_optimized.predict_feature_matrices"))
    return checkpoint


def generate():
    REPORT.mkdir(parents=True, exist_ok=True)
    dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "joblib", "threadpoolctl", "pvlib", "h5py", "matplotlib", "tabulate"]
    (REPORT / "requirements-lock.txt").write_text("\n".join(f"{name}=={version(name)}" for name in dependencies)+"\n")
    comparison, candidates, confidence, daylight, monthly, weather, uncertainty, release_rows = [], [], [], [], [], [], [], []
    selected_frames = {}
    for directory in sorted(RUNS.iterdir()):
        if not (directory / "test_summary.json").exists():
            raise RuntimeError(f"unfinished experiment: {directory}")
        config = read(directory / "search_config.json")
        selected = pd.read_csv(directory / "selected/predictions.csv")
        base = pd.read_csv(directory / "reference_baseline/predictions.csv")
        cap = config["dataset_manifest"]["cap_kw"]
        frame = extra_baselines(directory, selected, config)
        checkpoint = export_selected(directory, frame, config)
        release_rows.append(dict(dataset=directory.name, checkpoint=checkpoint.name,
                                 size_mb=checkpoint.stat().st_size/1024**2, sha256=sha256(checkpoint)))
        chosen, original = read(directory / "selected/metrics.json"), read(directory / "reference_baseline/metrics.json")
        ci = paired_day_bootstrap(base, selected, cap, config["dataset_manifest"]["timezone"])
        dump_json(directory / "paired_day_bootstrap.json", ci)
        evidence = REPORT / "evidence" / directory.name
        evidence.mkdir(parents=True, exist_ok=True)
        for name in ("search_config.json", "validation_search.json", "selection.json", "test_summary.json", "paired_day_bootstrap.json"):
            shutil.copy2(directory / name, evidence / name)
        for folder in directory.iterdir():
            if folder.is_dir() and (folder / "metrics.json").exists():
                shutil.copy2(folder / "metrics.json", evidence / (folder.name + "_metrics.json"))
        dump_json(REPORT / (directory.name + "_selection.json"), read(directory / "selection.json"))
        selected_frames[directory.name] = (selected, base, cap, config["dataset_manifest"]["timezone"])
        for h in HORIZONS:
            s, b = chosen["by_horizon"][str(h)], original["by_horizon"][str(h)]
            comparison.append(dict(dataset=directory.name, horizon_min=h, n=s["n"],
                baseline_mae_accuracy_pct=b["one_minus_mae_over_cap"]*100,
                baseline_rmse_accuracy_pct=b["one_minus_rmse_over_cap"]*100,
                mae_kw=s["mae_kw"], rmse_kw=s["rmse_kw"], mae_accuracy_pct=s["one_minus_mae_over_cap"]*100,
                rmse_accuracy_pct=s["one_minus_rmse_over_cap"]*100,
                rmse_reduction_pct=(1-s["rmse_kw"]/b["rmse_kw"])*100,
                pass_both="是" if s["pass_both"] else "否",
                monthly_pass_pct=s["monthly_pass_both_rate"]*100))
            confidence.append(dict(dataset=directory.name, horizon_min=h,
                gain_pp=(b["rmse_kw"]-s["rmse_kw"])/cap*100,
                ci_low_pp=ci[str(h)]["accuracy_gain_pp_ci95"][0],
                ci_high_pp=ci[str(h)]["accuracy_gain_pp_ci95"][1], days=ci[str(h)]["days"]))
            if "daylight" in s:
                daylight.append(dict(dataset=directory.name, horizon_min=h, n=s["daylight"]["n"],
                    mae_accuracy_pct=s["daylight"]["one_minus_mae_over_cap"]*100,
                    rmse_accuracy_pct=s["daylight"]["one_minus_rmse_over_cap"]*100))
            for month, v in s["monthly"].items():
                monthly.append(dict(dataset=directory.name, horizon_min=h, month_utc=month,
                    mae_accuracy_pct=v["one_minus_mae_over_cap"]*100,
                    rmse_accuracy_pct=v["one_minus_rmse_over_cap"]*100, pass_both=v["pass_both"], n=v["n"]))
            for label, v in s["weather"].items():
                weather.append(dict(dataset=directory.name, horizon_min=h, weather=label, n=v["n"],
                    mae_accuracy_pct=v.get("one_minus_mae_over_cap", np.nan)*100,
                    rmse_accuracy_pct=v.get("one_minus_rmse_over_cap", np.nan)*100))
            uncertainty.append(dict(dataset=directory.name, horizon_min=h,
                coverage_pct=s["coverage_p10_p90"]*100, width_kw=s["mean_interval_width_kw"],
                p10_pinball_kw=s["pinball_kw"][0], p50_pinball_kw=s["pinball_kw"][1], p90_pinball_kw=s["pinball_kw"][2]))
        validation = {r["model"]:r["validation_rmse_over_cap"] for r in read(directory / "validation_search.json")}
        validation["validation_ensemble"] = read(directory / "selection.json")["validation_ensemble_rmse_over_cap"]
        for folder in sorted(directory.iterdir()):
            if folder.is_dir() and (folder / "metrics.json").exists():
                metric = read(folder / "metrics.json")
                for j, h in enumerate(HORIZONS):
                    m = metric["by_horizon"][str(h)]
                    candidates.append(dict(dataset=directory.name, algorithm=folder.name, horizon_min=h,
                        validation_rmse_over_cap=validation.get(folder.name, [np.nan]*3)[j],
                        test_mae_accuracy_pct=m["one_minus_mae_over_cap"]*100,
                        test_rmse_accuracy_pct=m["one_minus_rmse_over_cap"]*100))
        print("report and release", directory.name, flush=True)
    frames = {"comparison":pd.DataFrame(comparison), "candidate_matrix":pd.DataFrame(candidates),
              "confidence_intervals":pd.DataFrame(confidence), "daylight":pd.DataFrame(daylight),
              "monthly":pd.DataFrame(monthly), "weather":pd.DataFrame(weather),
              "uncertainty":pd.DataFrame(uncertainty), "model_releases":pd.DataFrame(release_rows)}
    for name, frame in frames.items():
        frame.to_csv(REPORT / (name + ".csv"), index=False, float_format="%.8f")
    make_figures(frames, selected_frames)
    write_report(frames)


def save_figure(name):
    plt.tight_layout()
    plt.savefig(REPORT / name, dpi=160, bbox_inches="tight")
    plt.close()


def make_figures(frames, selected_frames):
    c = frames["comparison"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        sub = c[c.horizon_min == h]
        y = np.arange(len(sub))
        ax.barh(y-.18, sub.baseline_rmse_accuracy_pct, .36, label="Persistence reference", color="#8c929a")
        ax.barh(y+.18, sub.rmse_accuracy_pct, .36, label="Validation-selected", color="#178b80")
        ax.set_yticks(y, sub.dataset.str.replace("state_grid_", "SG"))
        ax.set_xlim(45, 100); ax.axvline(96, color="#c94444", linestyle="--")
        ax.set_title(f"{h} min"); ax.set_xlabel("1 - RMSE / Cap (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.04), ncol=2, fontsize=8)
    save_figure("accuracy_comparison.png")
    pivot = frames["monthly"].query("dataset == 'state_grid_1'").pivot(index="month_utc", columns="horizon_min", values="rmse_accuracy_pct")
    pivot.plot(marker="o", figsize=(9, 4)); plt.axhline(96, color="#c94444", linestyle="--")
    plt.ylabel("1 - RMSE / Cap (%)"); plt.title("SG1: monthly accuracy (UTC target month)")
    save_figure("monthly_accuracy.png")
    c.pivot(index="dataset", columns="horizon_min", values="monthly_pass_pct").plot.bar(figsize=(12, 4), color=["#178b80", "#497bbb", "#ca685d"])
    plt.ylabel("Monthly pass rate, both metrics (%)"); plt.xticks(rotation=35, ha="right")
    save_figure("pass_rate.png")
    for name in ("state_grid_1", "skippd_images"):
        if name not in selected_frames:
            continue
        selected, base, cap, timezone = selected_frames[name]
        s = selected[selected.horizon_minutes == 60].copy()
        local_day = pd.to_datetime(s.target_at, utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
        counts = local_day.value_counts().sort_index()
        # First complete observed test day, fixed by coverage, not prediction error.
        day = counts[counts >= counts.max()*.9].index[0]
        mask = local_day.to_numpy() == day
        times = pd.to_datetime(s.target_at[mask], utc=True).dt.tz_convert(timezone)
        b = base[base.horizon_minutes == 60]
        plt.figure(figsize=(11, 4))
        plt.plot(times, s.actual_kw[mask]/cap, color="#222222", label="Actual")
        plt.plot(times, b.predicted_kw[mask]/cap, color="#93989e", label="Reference")
        plt.plot(times, s.predicted_kw[mask]/cap, color="#178b80", label="Selected")
        plt.fill_between(times, s.p10_kw[mask]/cap, s.p90_kw[mask]/cap, color="#178b80", alpha=.15, label="P10-P90")
        plt.title(f"{name}: 60 min, {day} ({timezone})"); plt.ylabel("Power / Cap"); plt.legend()
        save_figure(name + "_curve.png")
    plt.figure(figsize=(10, 4))
    selected, base, cap, _ = selected_frames["state_grid_1"]
    for data, label, color in ((base, "Reference", "#93989e"), (selected, "Selected", "#178b80")):
        e = np.abs(data.query("horizon_minutes == 240").actual_kw - data.query("horizon_minutes == 240").predicted_kw)/cap
        e = np.sort(e)
        plt.plot(e, np.arange(1,len(e)+1)/len(e), label=label, color=color)
    plt.xlim(0, 1); plt.xlabel("Absolute error / Cap"); plt.ylabel("Cumulative fraction"); plt.legend()
    plt.title("SG1: 4 h absolute-error distribution")
    save_figure("error_distribution.png")
    matrix = frames["candidate_matrix"]
    ablation = matrix[matrix.dataset.isin(["skippd", "skippd_images"])]
    if len(ablation):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
        for ax, h in zip(axes, HORIZONS):
            p = ablation[ablation.horizon_min == h].pivot(index="algorithm", columns="dataset", values="test_rmse_accuracy_pct")
            p.plot.barh(ax=ax, color=["#497bbb", "#178b80"], legend=False)
            ax.set_title(f"{h} min"); ax.set_xlim(70, 100); ax.set_xlabel("RMSE accuracy (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=2)
        save_figure("image_ablation.png")
    verified = frames["weather"].query("dataset == 'skippd_images' and n > 0 and weather != 'unknown'")
    if len(verified):
        verified.pivot(index="weather", columns="horizon_min", values="rmse_accuracy_pct").plot.bar(figsize=(8, 4))
        plt.axhline(96, color="#c94444", linestyle="--"); plt.ylabel("1 - RMSE / Cap (%)")
        plt.title("Verified source weather labels only")
        save_figure("weather_accuracy.png")
        selected, base, cap, timezone = selected_frames["skippd_images"]
        labels = verified.weather.unique()
        fig, axes = plt.subplots(len(labels), 1, figsize=(11, 3.5*len(labels)), squeeze=False)
        for ax, label in zip(axes.flat, labels):
            s = selected[(selected.horizon_minutes == 60) & (selected.weather == label)]
            dates = pd.to_datetime(s.target_at, utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
            day = sorted(dates.unique())[0]
            s = s[dates == day]
            times = pd.to_datetime(s.target_at, utc=True).dt.tz_convert(timezone)
            ref = base.set_index(["issued_at", "horizon_minutes"]).loc[pd.MultiIndex.from_frame(s[["issued_at", "horizon_minutes"]])]
            ax.plot(times, s.actual_kw.to_numpy()/cap, label="Actual", color="#222222")
            ax.plot(times, ref.predicted_kw.to_numpy()/cap, label="Reference", color="#93989e")
            ax.plot(times, s.predicted_kw.to_numpy()/cap, label="Selected", color="#178b80")
            ax.set_title(f"Verified {label}: {day}, 60 min")
            ax.set_ylabel("Power / Cap"); ax.legend()
        save_figure("weather_curves.png")


def write_report(frames):
    c = frames["comparison"]
    sg = c[c.dataset.str.startswith("state_grid_")]
    simple = c[["dataset", "horizon_min", "baseline_rmse_accuracy_pct", "mae_accuracy_pct", "rmse_accuracy_pct", "rmse_reduction_pct", "pass_both"]].copy()
    simple.columns = ["数据集", "时距/min", "基线RMSE准确率/%", "优化MAE准确率/%", "优化RMSE准确率/%", "RMSE降低/%", "两项均达96%"]
    avg = sg.groupby("horizon_min")[["baseline_rmse_accuracy_pct", "mae_accuracy_pct", "rmse_accuracy_pct", "rmse_reduction_pct"]].mean().reset_index()
    paths = [d for d in RUNS.iterdir() if (d / "search_config.json").exists()]
    configuration = []
    for d in sorted(paths):
        cfg = read(d / "search_config.json")
        configuration.append(dict(dataset=d.name, cap_kw=cfg["dataset_manifest"]["cap_kw"],
            train=cfg["indices"]["train"], val=cfg["indices"]["val"], test=cfg["indices"]["test"],
            train_end=cfg["split"]["train_end"], val_end=cfg["split"]["val_end"]))
    best = c.loc[c.rmse_accuracy_pct.idxmax()]
    lines = ["# 光伏预测算法优化实验报告", "",
        "本报告由已完成实验的配置、预测 CSV 和指标 JSON 自动生成。所有数值均来自真实数据测试，未把未完成训练或已有权重当作已验证结果。",
        "", "## 1. 结论", "",
        f"共完成 {len(paths)} 组数据/输入方案、每组七种学习算法、三种时距的比较，并在验证集拟合集成。最高单项 RMSE 准确率为 {best.rmse_accuracy_pct:.3f}%（{best.dataset}，{int(best.horizon_min)} min）。",
        f"国家电网八站共 24 个站点/时距组合，其中 {(sg.pass_both == '是').sum()} 个同时达到两项 96% 门槛。全部三种时距均达到 96% 的电站数为 {int(sg.groupby('dataset').pass_both.apply(lambda x:(x=='是').all()).sum())}/8。以下保留未达标结果。",
        "“准确率”在这里专指 1−MAE/Cap 和 1−RMSE/Cap；不是分类正确率。没有证据说明本轮已经得到全局最优算法，也不能保证在新电站上复现最高值。",
        "", "八站等权平均（每站先算指标，避免大容量电站支配统计）：", "", table(avg), "",
        "## 2. 固定协议与数据", "", table(pd.DataFrame(configuration)), "",
        "- seed=42；目标为真正经过 15、60、240 分钟的瞬时/分钟平均功率标签，不是未来窗口能量平均。",
        "- 按有观测记录的本地日期进行 60%/10%/30% 时序切分；训练与验证样本的最长目标不得跨越边界，历史窗口不得跨越缺测间隙。不同站点采样完整性会改变最终比例。",
        "- 天空数据：SKIPP’D，64×64 RGB、过去30个连续分钟；362,022条可用记录。分钟平均功率按分钟结束后可用处理。",
        "- 国家电网：八站15分钟数据、过去64步（16小时）；仅使用发布时刻及以前的功率、TSI/DNI/GHI和温度。这些气象列是实测，不能充当未来 NWP。",
        "- 功率除以固定名义 Cap，辐照度除以1000，温度除以40；Ridge 标准化器仅在训练集拟合。未按测试最大值归一化。仅预测裁剪到[0,Cap]，不裁剪测试真值。",
        "- 验证集前60%的观测日期用于候选/权重选择，后40%日期仅用于残差分位数校准。每个时距独立选型，选型目标是验证 MSE。没有选择测试集最高分的模型。",
        "- 本轮沿用此前已评估过的固定测试集，属于固定留出集的描述性改善证据；要作最终泛化/竞赛达标结论，仍须一份从未接触的新时间段或外部电站数据。",
        "- Cap 暂为数据出处的名义容量；SKIPP’D为30.1kW-DC，实际可用AC容量未提供。因此96%判定是此参考口径下的结果。",
        "- 月度统计按目标时间的UTC月份计算，与现有统一指标函数一致；按天自助统计和示例曲线使用电站本地日期。",
        "", "## 3. 算法设计与论文借鉴", "",
        "先建立持久性/Smart Persistence、训练集历史时槽均值、pvlib物理代理基线；随后比较七种模型：Ridge、HistGradientBoosting、ExtraTrees、LightGBM小/中容量、LightGBM持久性残差、LightGBM L1损失。所有超参数见各数据集 search_config.json 和候选矩阵。",
        "输入包含近期功率滞后、均值/标准差/极值、变化率、本地时刻与年周期、训练集时槽均值。天空方案加入晴空功率、晴空指数和太阳位置；气象方案加入历史实测气象。已知未来太阳轨迹、日历、训练集时槽均值可作为目标时刻特征，未来实测功率和气象不能。",
        "集成权重非负且和为1，通过验证集MSE求解，每个时距均允许退回基线。天空图像方案提取RGB统计、红蓝对比、亮区比例、16块空间亮度，以及30分钟均值和15分钟变化；有无图像使用相同样本和模型参数。其思路借鉴SUNSET的功率/天空时序混合输入和云检测的红蓝色差，并非SUNSET或FusionSF官方结构复现。",
        "P10/P50/P90来自独立验证校准日期的残差分位数，加到点预测后裁剪；这是轻量不确定性估计，既不是SkyGPT，也不是带条件覆盖保证的完整概率模型。点指标使用原选定点预测，P50单独保存。",
        "参考文献与代码：", "",
        "1. Sun等，2019，Short-term solar power forecast with deep learning: Exploring optimal input and output configuration：[论文](https://doi.org/10.1016/j.solener.2019.06.041)、[SUNSET代码](https://github.com/YuchiSun/SUNSET)。",
        "2. Nie等，2023，SKIPP’D数据集：[论文](https://doi.org/10.1016/j.solener.2023.03.043)、[数据/基准代码](https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset)。",
        "3. [天空云检测代码](https://github.com/yuhao-nie/Cloud-dection-in-sky-images)，用于了解红蓝色差与天空条件特征。",
        "4. Ke等，2017，LightGBM：[官方实现和文档](https://github.com/lightgbm-org/LightGBM)。本轮使用官方库，而未复制自定义树实现。",
        "5. [scikit-learn树集成](https://scikit-learn.org/stable/modules/ensemble.html)、[pvlib](https://pvlib-python.readthedocs.io/)。研究下载快照位于research目录；XGBoost资料已查阅，本轮未训练XGBoost，不能列作实验结果。",
        "", "## 4. 测试结果", "", table(simple), "",
        "![各站各时距对比](accuracy_comparison.png)", "",
        "MAE/RMSE的kW原值、样本数和两种归一化指标全部保存在comparison.csv。candidate_matrix.csv列出每种模型及历史均值/物理基线，不据测试矩阵重新选型。图中红线为96%。",
        "", "## 5. 改善的统计区间", "",
        "对同一天的预测作为整体重采样，2000次配对日块bootstrap，seed=42。gain_pp是优化相对基线的RMSE准确率百分点增量；95%区间下界大于0支持在本次测试日期上稳定改善。此方法保留日内关联，但未建模多日天气持续性，区间不是未来部署保证。", "",
        table(frames["confidence_intervals"]), "",
        "## 6. 白天、月份与天气", "",
        "整体指标包含原始数据中可用的夜间零功率，另列白天子集；国家电网白天是06:00–19:00本地时钟代理，缺少站点经纬度，不能当作精确太阳高度分类。SKIPP’D源数据仅白天，没有夜间评估。", "",
        table(frames["daylight"]), "",
        "![逐月准确率](monthly_accuracy.png)", "", "![逐月两指标达标率](pass_rate.png)", "",
        "天气只使用可核实的来源标签。未提供的天气保留unknown；来源仅标注cloudy时，不把它擅自分成多云和阴天。weather.csv保留晴天/多云/阴天无标签组的n=0，不能据此声称完成三类天气全面验证。", "",
        table(frames["weather"].query("dataset.str.startswith('skippd') and n > 0", engine="python")), "",
        "![可核实天气分组曲线](weather_curves.png)", "",
        "天气曲线使用每种已核实标签的首个测试日期，缺少标签的多云/阴天组无法绘制可靠曲线。", "",
        "## 7. 图像消融、误差与预测曲线", "",
        "![相同模型图像消融](image_ablation.png)", "",
        "消融中每种模型的测试差异可以比较图像输入的贡献；分别验证选出的两个集成还会改变权重，不能把集成差异全部归因于图像。若某些时距图像降低测试效果，仍照实保留。",
        "", "![四小时误差分布](error_distribution.png)", "",
        "![国家电网示例](state_grid_1_curve.png)", "", "![天空数据示例](skippd_images_curve.png)", "",
        "曲线使用按有效覆盖度选出的首个完整测试日期，不根据预测误差挑选；展示60分钟预测。阴影为校准残差区间。",
        "", "## 8. 概率输出验证", "", table(frames["uncertainty"]), "",
        "P10–P90名义覆盖80%，实际覆盖与区间宽度同时报告。覆盖偏离80%表明校准分布与测试分布存在变化；不能只提供窄区间而不检验覆盖。",
        "", "## 9. 可复现交付与运行", "",
        "所有候选在本地runs/optimization_v2保留配置、checkpoint、预测CSV、指标JSON和源码版本。报告evidence目录另附选型、候选指标和配置，精选权重随仓库交付，完整预测CSV留在本地。训练源数据按manifest中的SHA256固定，不把原始数据上传GitHub。验证集选型文件先于测试指标保存。",
        "本轮运行期间补了HDF5排序读取、缓存一致性校验、输出目录参数和报告/推理接口；候选模型、参数、选型方法保持固定。旧批次code_version记录的是当时磁盘源码树哈希，不能把它当作解释器内存中所有已加载源码的精确快照；已完善后续运行的loaded_algorithm_sha256及training_optimize.py保存。当前交付源码与已验证权重的预测行为进行了重新加载核对。",
        "精选权重位于product/backend/data/model_releases/optimization_v2；每个站点附JSON清单、特征顺序、历史均值状态、集成权重、校准残差和SHA256。已从序列化文件重新加载，并验证跨越整个测试期的最多1000个确定性样本能重现预测。", "",
        table(frames["model_releases"][["dataset", "checkpoint", "size_mb"]]), "",
        "```bash", "pip install -r experiments/requirements-optimization.txt",
        "python -m experiments.optimize --dataset all --output-root runs/optimization_v2",
        "python -m experiments.optimize --dataset skippd --images --output-root runs/optimization_v2",
        "python -m experiments.report_optimization",
        "python -m pytest experiments/tests/test_optimization.py -q", "```", "",
        "macOS缺少libomp时可在命令前加DYLD_LIBRARY_PATH指向已安装torch/lib。依赖版本在requirements及运行配置中保存。预处理数据缺失时先按EXPERIMENT_REPORT.md执行prepare。源码/参数变化时须使用新输出位置，不能混用旧预测缓存。",
        "推理接口为experiments.predict_optimized.load_release及predict_feature_matrices，输入严格匹配有序特征矩阵与归一化基线，返回kW点预测和三分位数。生产特征构建仍需对接站点实时数据和发布时刻可用性，尚未自动启用后端在线预测。",
        "验证：后端170项测试通过；优化测试覆盖未来观测不可改变发布时刻特征、精确时距/缺测过滤、凸集成及权重推理接口；每组精选模型重载检查覆盖1000个确定性测试样本。后端滚动测试固定中午并提供合法SOC，避免最后时槽和模拟SOC扰动干扰其创建修正的断言。",
        "", "## 10. 未完成项与下一阶段优先级", "",
        "本次没有完成SUNSET官方完整训练、原MobileNetV3+ConvLSTM+attention+fusion的合规完整重训、ConvNeXt/Swin/VMamba/Spatial Mamba及FusionSF/SkyGPT的全部训练比较。此前AMPDNet仅训练两轮，属于不完整训练，不列入本轮成熟模型排名。不能将旧权重的测试集选型或按各分区最大值缩放结果用于达标证明。",
        "1. 优先获取发布时刻可追溯的NWP预报和真实可用Cap；这对1小时/4小时预测尤其关键。需要用新的未接触测试期比较，不能用未来实测气象代替。",
        "2. 在GPU上用同样固定协议完成SUNSET与原融合模型基准，再逐个做骨干替换、时序模块及融合消融；只有验证集稳定获益才保留。",
        "3. 以本轮图像消融为依据，研究云运动/太阳遮挡特征与多时刻图像编码，并对少量可靠天气标签进行人工补充或独立标注。",
        "4. 概率模型可升级为条件分位数树/网络与滚动校准，必须保留分位数损失、覆盖率、区间宽度以及新时间段验证。",
        "5. 最后做新电站迁移、不同季节/坡向测试和服务延迟评估；本轮不做电价预测。", ""]
    (REPORT / "optimization_report.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(REPORT / "report_manifest.json", dict(seed=42, horizons=list(HORIZONS), threshold=.96,
                completed_datasets=len(paths), checkpoints=frames["model_releases"].to_dict("records"),
                report_sha256=sha256(REPORT / "optimization_report.md")))


if __name__ == "__main__":
    generate()
