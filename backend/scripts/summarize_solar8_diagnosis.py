#!/usr/bin/env python3
"""Summarize frozen-model diagnosis, exclude raw-source gaps, and draw audit figure."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from diagnose_solar8 import bootstrap, TASKS, ROOT, OUT

def main():
    metrics=json.loads((OUT/'metrics.json').read_text())
    frame=pd.read_csv(OUT/'origin_errors.csv.gz')
    source=pd.read_csv(OUT/'solar8_original_presence.csv')
    source.ts=pd.to_datetime(source.ts)
    times={t:i for i,t in enumerate(source.ts)}
    bad=(~source.original_present).to_numpy()
    prefix=np.r_[0,np.cumsum(bad)]
    strict=[];flags=[]
    for row in frame[frame.station=='Solar8'].itertuples():
        length,steps=TASKS[row.task]
        t=row.start_row+length
        valid=prefix[t+steps]-prefix[row.start_row]==0
        for i in range(t,t+steps):
            y=times.get(source.ts.iloc[i]-pd.Timedelta(days=1))
            if y is None or bad[y]:valid=False
        flags.append(valid)
    s8=frame[frame.station=='Solar8'].copy()
    s8['all_input_target_baseline_original']=flags
    for task,f in s8.groupby('task'):
        for label,sel in [('test_purged_raw_complete_all_origins',f.heldout_purged & f.all_input_target_baseline_original),
                          ('test_purged_raw_complete_stride52',f.heldout_purged & f.all_input_target_baseline_original & f.original_stride52)]:
            subset=f[sel]
            if len(subset):
                strict.append({'station':'Solar8','task':task,'protocol':label,
                   'start':subset.target_start.min(),'end':subset.target_start.max(),
                   'origin_times':','.join(sorted(subset.origin_time.unique())),**bootstrap(subset)})
    (OUT/'raw_complete_metrics.json').write_text(json.dumps(strict,ensure_ascii=False,indent=2))
    s8.to_csv(OUT/'solar8_origin_errors_with_source_flags.csv.gz',index=False,compression='gzip')
    pd.DataFrame(metrics+strict).to_csv(OUT/'metrics_table.csv',index=False)
    clean=s8[s8.heldout_purged & s8.all_input_target_baseline_original]
    strata=[]; correlations=[]
    for task,f in clean.groupby('task'):
        for feature in ['past_tsi_ramp_mean','past_temp_ramp_mean','future_power_ramp_max_pu','future_tsi_ramp_mean']:
            rho,_=spearmanr(f[feature],f.mae_kw-f.baseline_mae_kw)
            correlations.append({'task':task,'feature':feature,'rho_error_disadvantage':float(rho)})
            q=float(f[feature].quantile(.75))
            for label,select in [('lower75',f[feature]<=q),('upper25',f[feature]>q)]:
                if select.any():strata.append({'task':task,'feature':feature,'stratum':label,'cutoff':q,**bootstrap(f[select])})
    (OUT/'raw_complete_strata.json').write_text(json.dumps(strata,ensure_ascii=False,indent=2))
    pd.DataFrame(correlations).to_csv(OUT/'raw_complete_correlations.csv',index=False)
    # Actual hourly stratification; bootstrap by local day within each hour group.
    hourly=[]
    for (task,hour),f in s8[s8.heldout_purged & s8.all_input_target_baseline_original].groupby(['task',s8.origin_time.str[:2]]):
        hourly.append({'task':task,'hour':hour,**bootstrap(f)})
    (OUT/'hourly_metrics.json').write_text(json.dumps(hourly,ensure_ascii=False,indent=2))
    audit=pd.read_csv(OUT/'dataset_comparison.csv')
    test=audit[audit.window=='tail20_rows'].set_index('station')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].bar(test.index,test.power_ramp_gt10pct_rate*100,color=['#94a3b8']*4+['#ef4444'])
    axes[0,0].set(title='15-min power ramps >10% nominal capacity',ylabel='% of contiguous 15-min steps')
    axes[0,1].bar(test.index,test.temperature_c_ramp_abs_p95,color=['#94a3b8']*4+['#ef4444'])
    axes[0,1].set(title='Temperature step size (95th percentile)',ylabel='degC per 15 min')
    subset=[m for m in strict if m['protocol']=='test_purged_raw_complete_all_origins']
    axes[1,0].bar([m['task'] for m in subset],[m['skill'] for m in subset],color='#2563eb')
    for i,m in enumerate(subset):
        lo,hi=m['skill_ci95'];axes[1,0].plot([i,i],[lo,hi],color='black')
    axes[1,0].axhline(0,color='gray',lw=1)
    axes[1,0].set(title='Solar8 raw-complete held-out all-origin skill',ylabel='MAE skill (95% day-bootstrap CI)')
    h=pd.DataFrame(hourly)
    for task in ['15min','1h']:
        f=h[h.task==task];axes[1,1].plot(f.hour.astype(int),f.skill,marker='o',label=task)
    axes[1,1].axhline(0,color='gray',lw=1);axes[1,1].legend()
    axes[1,1].set(title='Raw-complete held-out skill by origin hour',xlabel='Hour (Asia/Shanghai)',ylabel='MAE skill')
    fig.suptitle('Solar8 diagnostic audit — post-hoc, frozen artifacts; no causal weather claim')
    fig.savefig(OUT/'solar8_diagnostic.png',dpi=160)
    print(json.dumps([m for m in metrics+strict if m['station']=='Solar8'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
