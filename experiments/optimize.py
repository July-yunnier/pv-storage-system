"""Validation-only algorithm selection on frozen, chronological PV cases.

Run with DYLD_LIBRARY_PATH pointing at torch/lib on macOS without libomp.
No fit, clipping rule, blend weight or hyperparameter reads test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import h5py
import joblib
import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.baselines import clear_sky_power, frozen_historical_mean
from experiments.metrics import evaluate
from experiments.protocol import ROOT, code_version, dump_json, sample_indices, seed_everything, sha256

HORIZONS = (15, 60, 240)
LOADED_SOURCE = Path(__file__).read_text(encoding="utf-8")


def sky_descriptors(frame, image_path):
    cache = ROOT / "data/processed/skippd/image_descriptors_v1.npz"
    if cache.exists():
        saved = np.load(cache)
        if not np.array_equal(saved["image_index"], frame.image_index.to_numpy()):
            raise ValueError("image cache order differs from dataset")
        if not np.array_equal(saved["image_group"], frame.image_group.to_numpy()):
            raise ValueError("image cache groups differ from dataset")
        return saved["features"], saved["names"].tolist()
    # Downsample by area averaging; descriptors use the actual sky bitmap.
    # RGB moments, red-blue contrast and spatial tile brightness retain cloud
    # and sun-region cues without training a large CNN on this CPU.
    names = ["image_mean_r", "image_mean_g", "image_mean_b", "image_std_r", "image_std_g", "image_std_b",
             "image_rb_mean", "image_rb_std", "image_bright_fraction", "image_rb_positive_fraction"]
    names += [f"image_tile_{i}" for i in range(16)]
    parts = np.empty((len(frame), len(names)), np.float32)
    with h5py.File(image_path, "r") as f:
        for group in ("trainval", "test"):
            positions = np.flatnonzero(frame.image_group.to_numpy() == group)
            source = frame.image_index.to_numpy()[positions]
            for start in range(0, len(source), 1024):
                pos = positions[start:start+1024]
                idx = source[start:start+1024]
                order = np.argsort(idx)
                images = f[group]["images_log"][idx[order]][np.argsort(order)].astype(np.float32) / 255.
                mean = images.mean(axis=(1,2))
                std = images.std(axis=(1,2))
                rb = (images[:,:,:,0] - images[:,:,:,2]) / (images[:,:,:,0] + images[:,:,:,2] + .01)
                extra = np.stack([rb.mean((1,2)), rb.std((1,2)), (images.mean(3) > .85).mean((1,2)), (rb > 0).mean((1,2))], 1)
                tiles = images.mean(3).reshape(len(images),4,16,4,16).mean((2,4)).reshape(len(images),16)
                parts[pos] = np.concatenate([mean,std,extra,tiles],1)
            print("image descriptors",group,len(source),flush=True)
    np.savez_compressed(cache,features=parts,names=np.array(names),image_index=frame.image_index.to_numpy(),
                        image_group=frame.image_group.to_numpy(dtype=str))
    return parts,names


def build_features(frame, indices, future, manifest, split, images=False):
    sky = manifest["dataset"] == "SKIPPD"
    lookback, step = (30,1) if sky else (64,15)
    idx = np.concatenate(list(indices.values()))
    cap = manifest["cap_kw"]
    power = frame.power_kw.to_numpy(dtype=np.float32) / cap
    history = power[idx[:,None] - np.arange(lookback)[None,:]]
    names, columns = [], []

    def add(name, values):
        names.append(name)
        columns.append(np.asarray(values,np.float32))

    for lag in ([0,1,2,3,5,10,15,20,29] if sky else [0,1,2,3,4,8,12,16,24,32,48,63]):
        add(f"power_lag_{lag*step}min",history[:,lag])
    for window in ([3,5,10,15,30] if sky else [4,8,16,32,64]):
        a = history[:,:window]
        for stat, values in (("mean",a.mean(1)),("std",a.std(1)),("min",a.min(1)),("max",a.max(1))):
            add(f"power_{window*step}min_{stat}",values)
    add("power_delta",history[:,0] - history[:,1])
    add("power_slope",history[:,0] - history[:,-1])
    clock = pd.DatetimeIndex(frame.ts)
    local_minute = clock.hour * 60 + clock.minute
    for label, phase in (("clock",local_minute/1440), ("year",(clock.dayofyear-1)/365.25)):
        add(label+"_sin",np.sin(2*np.pi*phase[idx]))
        add(label+"_cos",np.cos(2*np.pi*phase[idx]))
    mean, _ = frozen_historical_mean(frame, split["train_end"])
    add("historical_slot_mean",mean[idx]/cap)
    if sky:
        clear = clear_sky_power(frame,manifest) / cap
        ratio = np.divide(power,clear,out=np.zeros_like(power),where=clear > .01)
        add("clear_now",clear[idx])
        add("clear_index_now",np.clip(ratio[idx],0,2))
        for window in (5,15,30):
            a = ratio[idx[:,None]-np.arange(window)[None,:]]
            add(f"clear_index_{window}_mean",np.clip(a,0,2).mean(1))
            add(f"clear_index_{window}_std",np.clip(a,0,2).std(1))
        solar = pvlib.solarposition.get_solarposition(clock,manifest["lat"],manifest["lon"])
        add("solar_elevation",solar.apparent_elevation.to_numpy()[idx]/90)
        add("solar_azimuth_sin",np.sin(np.deg2rad(solar.azimuth.to_numpy()[idx])))
        if images:
            descriptor, dn = sky_descriptors(frame,manifest["image_source"])
            for j,name in enumerate(dn):
                add(name,descriptor[idx,j])
                add(name+"_30_mean",np.stack([descriptor[idx-k,j] for k in range(30)],1).mean(1))
                add(name+"_delta_15",descriptor[idx,j]-descriptor[idx-15,j])
    else:
        clear = None
        for label in ("tsi","dni","ghi","temperature"):
            a = frame[label].to_numpy(np.float32)
            scale = 1000. if label != "temperature" else 40.
            a = a / scale
            for lag in (0,1,4,16):
                add(f"{label}_lag_{lag*15}min",a[idx-lag])
            for w in (4,16,64):
                past = a[idx[:,None]-np.arange(w)[None,:]]
                add(f"{label}_{w*15}_mean",past.mean(1))
                add(f"{label}_{w*15}_std",past.std(1))
            add(label+"_delta",a[idx]-a[idx-1])
    common = np.column_stack(columns).astype(np.float32)
    matrices, feature_names, baselines = [], [], []
    for j,horizon in enumerate(HORIZONS):
        target = clock[future[idx,j]]
        tm = target.hour*60 + target.minute
        extra = [np.sin(2*np.pi*tm/1440),np.cos(2*np.pi*tm/1440),
                 mean[future[idx,j]]/cap,np.full(len(idx),horizon/240.)]
        extra_names = ["target_clock_sin","target_clock_cos","target_historical_mean","horizon_fraction"]
        base = power[idx]
        if sky:
            tc = clear[future[idx,j]]
            extra += [tc,tc-clear[idx]]
            extra_names += ["target_clear_power","clear_change"]
            base = np.clip(ratio[idx] * tc,0,1)
        matrices.append(np.column_stack([common]+extra).astype(np.float32))
        feature_names.append(names+extra_names)
        baselines.append(base)
    counts = np.cumsum([0]+[len(v) for v in indices.values()])
    partitions = {name:np.arange(counts[i],counts[i+1]) for i,name in enumerate(indices)}
    labels = power[future[idx]]
    return idx,partitions,matrices,feature_names,np.column_stack(baselines),labels


def candidate_specs():
    return [
        ("ridge",dict(alpha=20.),False),
        ("hist_gradient_boosting",dict(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2.),False),
        ("extra_trees",dict(n_estimators=100,max_depth=18,min_samples_leaf=20,max_features=.8),False),
        ("lightgbm_small",dict(n_estimators=350,num_leaves=15,min_child_samples=80,learning_rate=.04,reg_lambda=5.),False),
        ("lightgbm_medium",dict(n_estimators=500,num_leaves=31,min_child_samples=60,learning_rate=.035,reg_lambda=5.),False),
        ("lightgbm_residual",dict(n_estimators=500,num_leaves=31,min_child_samples=60,learning_rate=.035,reg_lambda=5.),True),
        ("lightgbm_l1",dict(n_estimators=500,num_leaves=31,min_child_samples=60,learning_rate=.035,reg_lambda=5.,objective="regression_l1"),False),
    ]


def make_estimator(name, params):
    if name == "ridge":
        return make_pipeline(StandardScaler(),Ridge(**params))
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params,early_stopping=False,random_state=42)
    if name == "extra_trees":
        return ExtraTreesRegressor(**params,n_jobs=4,random_state=42)
    import lightgbm as lgb
    return lgb.LGBMRegressor(**params,verbosity=-1,n_jobs=4,random_state=42,deterministic=True,force_col_wise=True)


def blend_weights(predictions, labels):
    m = np.column_stack(predictions)
    start = np.zeros(m.shape[1]); start[np.argmin(((m-labels[:,None])**2).mean(0))] = 1.
    result = minimize(lambda w: np.mean((m@w-labels)**2), start,
                      jac=lambda w:2*m.T@(m@w-labels)/len(labels),method="SLSQP",
                      bounds=[(0,1)]*m.shape[1],constraints=[{"type":"eq","fun":lambda w:w.sum()-1,
                      "jac":lambda w:np.ones_like(w)}],options={"maxiter":200,"ftol":1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def rows(frame, idx, future, predictions):
    issue = pd.DatetimeIndex(frame.ts.iloc[idx]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    parts=[]
    for j,h in enumerate(HORIZONS):
        target=future[idx,j]
        parts.append(pd.DataFrame({"issued_at":issue,
            "target_at":pd.DatetimeIndex(frame.ts.iloc[target]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon_minutes":h,"actual_kw":frame.power_kw.iloc[target].to_numpy(),"predicted_kw":predictions[:,j],
            "weather":frame.weather.iloc[target].to_numpy(),"daylight":frame.daylight.iloc[target].to_numpy(),"split":"test"}))
    return pd.concat(parts,ignore_index=True)


def run(dataset, images=False, output_root=None):
    seed_everything()
    threadpool_limits(4)
    directory=ROOT/"data/processed"/dataset
    frame=pd.read_pickle(directory/"frame.pkl")
    manifest=json.loads((directory/"manifest.json").read_text())
    split=json.loads((directory/"split.json").read_text())
    sky=dataset=="skippd"
    indices,future=sample_indices(frame,split,30 if sky else 64,1 if sky else 15)
    idx,partitions,xs,names,base,y=build_features(frame,indices,future,manifest,split,images)
    train,valid,test=(partitions[k] for k in ("train","val","test"))
    # Tune and calibrate on disjoint validation days.
    vd=pd.DatetimeIndex(frame.ts.iloc[idx[valid]]).normalize()
    unique=vd.unique().sort_values()
    boundary=unique[int(len(unique)*.6)]
    tuning=valid[vd<boundary]; calibration=valid[vd>=boundary]
    cap=manifest["cap_kw"]
    out=(Path(output_root) if output_root else ROOT/"runs/optimization_v2")/(dataset+("_images" if images else ""))
    out.mkdir(parents=True,exist_ok=True)
    (out/"training_optimize.py").write_text(LOADED_SOURCE,encoding="utf-8")
    config=dict(dataset_manifest=manifest,split=split,seed=42,images=images,
                features=names,indices={k:len(v) for k,v in partitions.items()},
                validation_tuning_cases=len(tuning),validation_calibration_cases=len(calibration),
                validation_calibration_start=boundary.isoformat(),selection="tuning validation only; mean normalized MSE across horizons",
                protocol="frozen pv-v1 cases, features only through issue; v2 method search",
                baseline="smart_persistence" if sky else "persistence",candidate_specs=candidate_specs(),
                software={k:__import__(k).__version__ for k in ("numpy","pandas","sklearn")},
                loaded_algorithm_sha256=hashlib.sha256(LOADED_SOURCE.encode()).hexdigest(),code=code_version())
    dump_json(out/"search_config.json",config)
    cache_predictions={"reference_baseline":base}; fitted={}; search=[]
    started=time.monotonic()
    for name,params,residual in candidate_specs():
        checkpoint=out/(name+".joblib")
        prediction_file=out/(name+"_internal.npz")
        signature=hashlib.sha256(json.dumps(dict(params=params,residual=residual,features=names,
                        manifest=manifest,split=split,algorithm_sha256=sha256(__file__)),sort_keys=True).encode()).hexdigest()
        if checkpoint.exists() and prediction_file.exists():
            saved=np.load(prediction_file)
            if not np.array_equal(saved["idx"],idx):
                raise ValueError("cached prediction cases differ")
            if "signature" not in saved or saved["signature"].item()!=signature:
                raise ValueError("cached predictions use a different source/configuration; choose a fresh output directory")
            prediction=saved["prediction"]
            models=joblib.load(checkpoint)
        else:
            models=[]; prediction=np.empty_like(y)
            for j,h in enumerate(HORIZONS):
                m=make_estimator(name,params)
                target=y[train,j]-base[train,j] if residual else y[train,j]
                m.fit(xs[j][train],target)
                pred=m.predict(xs[j])+(base[:,j] if residual else 0)
                prediction[:,j]=np.clip(pred,0,1)
                models.append(m)
                print(dataset,name,h,"validation RMSE",round(float(np.sqrt(np.mean((prediction[tuning,j]-y[tuning,j])**2))),5),flush=True)
            joblib.dump(models,checkpoint,compress=3)
            np.savez_compressed(prediction_file,prediction=prediction,idx=idx,signature=signature)
        cache_predictions[name]=prediction
        fitted[name]=models
        search.append({"model":name,"params":params,"residual":residual,
                       "validation_rmse_over_cap":np.sqrt(((prediction[tuning]-y[tuning])**2).mean(0)).tolist(),
                       "validation_mae_over_cap":np.abs(prediction[tuning]-y[tuning]).mean(0).tolist(),
                       "elapsed_seconds":time.monotonic()-started})
        dump_json(out/"validation_search.json",search)
    choices=list(cache_predictions)
    weights=[]
    ensemble=np.empty_like(y)
    for j,h in enumerate(HORIZONS):
        w=blend_weights([cache_predictions[k][tuning,j] for k in choices],y[tuning,j])
        weights.append(w.tolist())
        ensemble[:,j]=np.column_stack([cache_predictions[k][:,j] for k in choices])@w
    cache_predictions["validation_ensemble"]=ensemble
    # Baseline included in validation selection, so complex models need to earn
    # their place. Compare all candidate test results only after this freeze.
    best_names=[min(cache_predictions,key=lambda k:np.mean((cache_predictions[k][tuning,j]-y[tuning,j])**2)) for j in range(3)]
    selected=np.column_stack([cache_predictions[k][:,j] for j,k in enumerate(best_names)])
    selection=dict(choices=choices,ensemble_weights=weights,selected_per_horizon=best_names,
                   validation_ensemble_rmse_over_cap=np.sqrt(((ensemble[tuning]-y[tuning])**2).mean(0)).tolist(),
                   selected_rmse_over_cap=np.sqrt(((selected[tuning]-y[tuning])**2).mean(0)).tolist(),
                   frozen_before_test=True,calibration_residual_quantiles=np.quantile(y[calibration]-selected[calibration],[.1,.5,.9],axis=0).tolist())
    dump_json(out/"selection.json",selection)
    cache_predictions["selected"]=selected
    for name,pred in cache_predictions.items():
        target=out/name
        target.mkdir(exist_ok=True)
        result_rows=rows(frame,idx[test],future,pred[test]*cap)
        if name=="selected":
            q=np.array(selection["calibration_residual_quantiles"])*cap
            for j,h in enumerate(HORIZONS):
                mask=result_rows.horizon_minutes==h
                for k,column in enumerate(("p10_kw","p50_kw","p90_kw")):
                    result_rows.loc[mask,column]=np.clip(result_rows.loc[mask,"predicted_kw"]+q[k,j],0,cap)
        result_rows.to_csv(target/"predictions.csv",index=False,float_format="%.8f")
        metrics=evaluate(result_rows,cap)
        dump_json(target/"metrics.json",metrics)
        dump_json(target/"config.json",dict(config,model=name,selection=selection))
        dump_json(target/"code_version.json",config["code"])
        if name in fitted:
            dump_json(target/"weights.json",dict(checkpoint="../"+name+".joblib",sha256=sha256(out/(name+".joblib"))))
        else:
            dump_json(target/"weights.json",selection if name in ("selected","validation_ensemble") else {"formula":config["baseline"]})
    summary={k:{h:round(v["one_minus_rmse_over_cap"],5) for h,v in json.loads((out/k/"metrics.json").read_text())["by_horizon"].items()} for k in cache_predictions}
    dump_json(out/"test_summary.json",summary)
    print("COMPLETED",dataset,summary,"seconds",time.monotonic()-started,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="state_grid_1")
    parser.add_argument("--images",action="store_true")
    parser.add_argument("--output-root",help="fresh output directory for a revised experiment")
    args=parser.parse_args()
    datasets=["skippd"]+[f"state_grid_{i}" for i in range(1,9)] if args.dataset=="all" else args.dataset.split(",")
    for dataset in datasets:
        run(dataset,args.images and dataset=="skippd",args.output_root)
