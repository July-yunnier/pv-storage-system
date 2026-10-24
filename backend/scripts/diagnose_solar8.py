#!/usr/bin/env python3
"""Frozen-model, post-hoc Solar8 audit. Never replaces the original P0-A round.

Run from repository root with a CPU torch/numpy/pandas/scipy environment.
No fitting, parameter selection, or production interface changes.
"""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'reports/official_experiment/p0a_ampdnet/data'
MODELS = ROOT / 'backend/data/model_releases/ampdnet_delivery20'
OUT = ROOT / 'reports/official_experiment/solar8_diagnosis'
TASKS = {'15min': (48, 1), '1h': (96, 4), '4h': (192, 16)}
SEED = 42


def bootstrap(frame):
    """Paired resampling of local calendar days; ratio CI uses same draws."""
    daily = frame.groupby('date')[['mae_kw', 'baseline_mae_kw']].mean()
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(daily), size=(2000, len(daily)))
    m = daily.mae_kw.to_numpy()[draws].mean(axis=1)
    b = daily.baseline_mae_kw.to_numpy()[draws].mean(axis=1)
    skill = 1 - frame.mae_kw.mean() / frame.baseline_mae_kw.mean()
    return {'origins': len(frame), 'days': len(daily),
            'mae_kw': float(frame.mae_kw.mean()),
            'day_persistence_mae_kw': float(frame.baseline_mae_kw.mean()),
            'skill': float(skill),
            'delta_ci95_kw': np.quantile(b-m, [.025, .975]).tolist(),
            'skill_ci95': np.quantile(1-m/b, [.025, .975]).tolist(),
            'daily_weighting': 'equal days; per-origin point estimates; days can have different counts'}


def data_audit(df, cap, station):
    rows = []
    for window, f in [('all', df), ('tail20_rows', df.iloc[int(len(df)*.8):])]:
        continuous = f.ts.diff().dt.total_seconds().to_numpy() == 900
        r = {'station': station, 'window': window, 'rows': len(f),
             'first_ts': f.ts.iloc[0].isoformat(), 'last_ts': f.ts.iloc[-1].isoformat()}
        for col, scale in [('power_kw', cap), ('tsi', 1000), ('dni', 1000), ('ghi', 1000), ('temperature_c', 1)]:
            delta = f[col].diff().to_numpy()
            valid = np.abs(delta[continuous]) / scale
            r[col+'_ramp_abs_p95'] = float(np.quantile(valid, .95))
            r[col+'_ramp_abs_max'] = float(valid.max())
            if col == 'power_kw':
                r['power_ramp_gt10pct_rate'] = float(np.mean(valid > .1))
                d1 = delta[1:]; d0 = delta[:-1]
                good = continuous[1:] & continuous[:-1]
                r['power_reversal_gt2pct_rate'] = float(np.mean(((d1*d0 < 0) & (abs(d1)>.02*cap) & (abs(d0)>.02*cap))[good]))
        for col in ['dni', 'ghi']:
            use = f.tsi.to_numpy() > 1
            ratio = f[col].to_numpy()[use] / f.tsi.to_numpy()[use]
            r[col+'_to_tsi_mean'] = float(ratio.mean())
            r[col+'_to_tsi_std'] = float(ratio.std())
            r[col+'_tsi_corr'] = float(f[col].corr(f.tsi))
        r['temperature_reversal_gt1deg_rate'] = float(np.mean(((f.temperature_c.diff()*f.temperature_c.diff().shift(1)<0) & (abs(f.temperature_c.diff())>1) & (abs(f.temperature_c.diff().shift(1))>1))[continuous]))
        rows.append(r)
    return rows


def infer_origins(df, station, task):
    length, steps = TASKS[task]
    meta_path = next(MODELS.glob(f'{task}_{station}_*s42.json'))
    meta = json.loads(meta_path.read_text())
    model = torch.jit.load(str(meta_path.with_name(meta['artifact_file'])), map_location='cpu').eval()
    features = df[['tsi', 'dni', 'ghi', 'temperature_c']].to_numpy(dtype=np.float32)
    low = np.array(meta['input_min'], dtype=np.float32)
    high = np.array(meta['input_max'], dtype=np.float32)
    features = (features - low) / (high - low)
    powers = df.power_kw.to_numpy()
    # Original train/val splits operate on overlapping windows, not on rows.
    count = len(df)-length-steps+1
    split2 = int(count*.8)
    # Discard targets overlapping the last validation target (purge steps-1).
    first_safe_start = split2 + steps - 1
    by_ts = {t:i for i,t in enumerate(df.ts)}
    starts = []
    for start in range(count):
        target = start+length
        end = target+steps-1
        if df.ts.iloc[target].date() != df.ts.iloc[target-1].date() or df.ts.iloc[end].date()!=df.ts.iloc[target-1].date():
            continue
        if df.ts.iloc[target-1]+pd.Timedelta(minutes=15*steps)!=df.ts.iloc[end]:
            continue
        if df.ts.iloc[target]-pd.Timedelta(days=1) not in by_ts:
            continue
        starts.append(start)
    # Full history only used to reproduce old protocol for Solar8. Controls use held-out targets only.
    if station != 'Solar8':
        starts = [s for s in starts if s >= first_safe_start]
    records = []
    for offset in range(0, len(starts), 128):
        batch_starts = starts[offset:offset+128]
        batch = np.stack([features[s:s+length] for s in batch_starts])
        with torch.inference_mode():
            norm = model(torch.from_numpy(batch)).numpy().reshape(len(batch_starts),steps)
        kw = np.clip((norm*(meta['output_max_mw']-meta['output_min_mw'])+meta['output_min_mw'])*1000, 0, meta['cap_kw'])
        for start,pred in zip(batch_starts, kw):
            t = start+length
            a = powers[t:t+steps]
            base = np.array([powers[by_ts[df.ts.iloc[i]-pd.Timedelta(days=1)]] for i in range(t,t+steps)])
            past = df.iloc[max(t-5,0):t]
            future = df.iloc[t-1:t+steps]
            records.append({'station':station,'task':task,'start_row':start,
                'issued_at':df.ts.iloc[t-1].isoformat(),'target_start':df.ts.iloc[t].isoformat(),
                'date':str(df.ts.iloc[t].date()),'origin_time':df.ts.iloc[t-1].strftime('%H:%M'),
                'heldout_purged':start>=first_safe_start,'original_stride52':start%52==0,
                'mae_kw':float(np.mean(abs(pred-a))), 'baseline_mae_kw':float(np.mean(abs(base-a))),
                'bias_kw':float(np.mean(pred-a)), 'actual_mean_kw':float(np.mean(a)),
                'past_tsi_ramp_mean':float(abs(np.diff(past.tsi)).mean()),
                'past_temp_ramp_mean':float(abs(np.diff(past.temperature_c)).mean()),
                'future_power_ramp_max_pu':float(abs(np.diff(future.power_kw)).max()/meta['cap_kw']),
                'future_tsi_ramp_mean':float(abs(np.diff(future.tsi)).mean())})
    frame = pd.DataFrame(records)
    metrics=[]
    for name, subset in [('all_history_stride52',frame[frame.original_stride52]),
                         ('test_purged_stride52',frame[frame.original_stride52 & frame.heldout_purged]),
                         ('test_purged_all_origins',frame[frame.heldout_purged])]:
        if len(subset):
            metrics.append({'station':station,'task':task,'protocol':name,
                            'start':subset.target_start.min(),'end':subset.target_start.max(),
                            'origin_times':','.join(sorted(subset.origin_time.unique())),**bootstrap(subset)})
    return frame,metrics


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    audit=[]; all_metrics=[]; frames=[]; hashes={}
    for station in ['Solar1','Solar2','Solar6','Solar7','Solar8']:
        path = DATA/f'{station}_processed_3.csv'
        df = pd.read_csv(path); df.ts=pd.to_datetime(df.ts)
        meta=json.loads(next(MODELS.glob(f'15min_{station}_*s42.json')).read_text())
        audit.extend(data_audit(df,float(meta['cap_kw']),station))
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for task in TASKS:
            print(station,task,flush=True)
            frame,metrics=infer_origins(df,station,task)
            frames.append(frame); all_metrics.extend(metrics)
    pd.DataFrame(audit).to_csv(OUT/'dataset_comparison.csv',index=False)
    all_frame=pd.concat(frames,ignore_index=True)
    all_frame.to_csv(OUT/'origin_errors.csv.gz',index=False,compression='gzip')
    (OUT/'metrics.json').write_text(json.dumps(all_metrics,indent=2,ensure_ascii=False))
    s8=all_frame[(all_frame.station=='Solar8') & all_frame.heldout_purged].copy()
    strata=[]; corrs=[]
    for task,f in s8.groupby('task'):
        for feature in ['past_tsi_ramp_mean','past_temp_ramp_mean','future_power_ramp_max_pu','future_tsi_ramp_mean']:
            rho,p=spearmanr(f[feature],f.mae_kw-f.baseline_mae_kw)
            corrs.append({'task':task,'feature':feature,'spearman_rho_error_disadvantage':float(rho),'p_descriptive_only':float(p)})
            cutoff=float(f[feature].quantile(.75))
            for label,m in [('lower75',f[feature]<=cutoff),('upper25',f[feature]>cutoff)]:
                if m.any():strata.append({'task':task,'feature':feature,'stratum':label,'cutoff':cutoff,**bootstrap(f[m])})
    (OUT/'strata.json').write_text(json.dumps(strata,indent=2))
    pd.DataFrame(corrs).to_csv(OUT/'exploratory_correlations.csv',index=False)
    (OUT/'run_meta.json').write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),
        'device':'cpu','threads':1,'bootstrap':{'seed':SEED,'replicates':2000,'unit':'local calendar day'},
        'input_sha256':hashes,'scope':'post-hoc diagnosis, no retraining; original P0-A not overwritten',
        'split':'last 20% of original overlapping windows; purge steps-1 to avoid val-target overlap',
        'weather':'only scalar irradiance and temperature; no sky images/cloud motion/clear-sky index',
        'strata':'exploratory test-period quartiles, descriptive only; no confirmatory causal claim'},indent=2))
    print('DONE',flush=True)

if __name__=='__main__':
    main()
