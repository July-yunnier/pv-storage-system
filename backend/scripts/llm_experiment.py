"""Isolated experimental E_LLM extension; never replaces the physical execute loop."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from agents.llm_energy_agent import clamp_policy


def is_event_day(event: dict | None) -> bool:
    """Events are selected from issue-time data; ordinary days stay on E2."""
    return bool(event and str(event.get("kind", "none")) != "none")


def actions_formulator_guided(forecast_pv, forecast_load, prices, policy, *, initial_soc=.5, deg=1.0):
    """Translate high-level constraints into actions through the existing rule planner."""
    from agents.llm_energy_agent import clamp_formulator_policy
    high = clamp_formulator_policy(policy)
    effective = dict(high)
    effective["soc_min_safe"] = max(
        high["soc_min_safe"], min(.95, high["soc_min_safe"] + .05 * high["reserve_weight"])
    )
    actions = actions_llm_guided(forecast_pv, forecast_load, prices, effective,
                                 initial_soc=initial_soc, deg=deg)
    start, end = high["discharge_window_start"], high["discharge_window_end"]
    return [(a, kw) if not (a == "discharge" and not (start <= i <= end)) else ("idle", 0.)
            for i, (a, kw) in enumerate(actions)]


def tail_metrics(costs, alpha=.90):
    """Return fixed daily-cost VaR/CVaR and worst-day metrics in yuan."""
    values = np.asarray(costs, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite daily costs are required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    var = float(np.quantile(values, alpha, method="linear"))
    tail = values[values >= var]
    return {"alpha": alpha, "var_cost_yuan": var,
            "cvar_cost_yuan": float(tail.mean()),
            "worst_day_cost_yuan": float(values.max()),
            "worst_day_index": int(values.argmax()), "n_days": int(len(values))}


def count_manual_interventions(records) -> int:
    """Count auditable human review/fallback/safety changes, never model calls."""
    return int(sum(bool(r.get("manual_review_required")) or r.get("fallback", False)
                   or r.get("policy_changed_by_safety_layer", False) for r in records))


def actions_llm_guided(forecast_pv, forecast_load, prices, policy, *, initial_soc=.5, deg=1.0):
    """Plan from forecasts only; safe reserve is advisory, executor keeps hard bounds.

    Signed offset shifts the original peak-price mask, without circular day wrap.
    Reserve constrains forecast-side discharge. Actual SOC always belongs to execute.
    """
    # Import lazily to avoid circular imports and share the existing constants.
    from scripts.run_official_experiment import CAP, PCS, SOC_MAX, ETA, DT, PRICES
    policy = clamp_policy(policy)
    pv=np.asarray(forecast_pv,float); load=np.asarray(forecast_load,float); pr=np.asarray(prices,float)
    if pv.ndim!=1 or not (pv.shape==load.shape==pr.shape) or not np.isfinite(np.r_[pv,load,pr]).all():
        raise ValueError('aligned finite forecast arrays required')
    offset=policy['peak_discharge_offset']
    peak=pr>=PRICES['peak']-1e-6
    shifted=np.zeros(len(pr),dtype=bool)
    for index in np.flatnonzero(peak):
        target=int(index)+offset
        if 0<=target<len(pr):shifted[target]=True
    energy=initial_soc*CAP*deg; reserve=policy['soc_min_safe']*CAP*deg
    maximum=SOC_MAX*CAP*deg; actions=[]
    for i,net in enumerate(load-pv):
        if pr[i]<=PRICES['valley']+1e-6 or net < -50:
            kw=max(0.,min(PCS,(maximum-energy)/ETA/DT))
            energy+=kw*DT*ETA
            actions.append(('charge',kw))
        elif shifted[i] and net>0:
            kw=max(0.,min(PCS,(energy-reserve)*ETA/DT,net))
            energy-=kw*DT/ETA
            actions.append(('discharge',kw))
        else:actions.append(('idle',0.))
    return actions


def comparison(agent, baseline, ci):
    delta=np.asarray(baseline,float)-np.asarray(agent,float)
    return {'total_increment_yuan':float(delta.sum()),'daily_mean_increment_yuan':float(delta.mean()),
            'daily_increment_bootstrap_ci95_yuan':ci(delta),
            'n_days':len(delta),'positive_means':'E_LLM costs less than comparator'}


def extend_experiment(df, out: Path, advisor):
    """Append actual offline runs to P0/P1/P3/P4 and an explicit PJM demo P2.

    Every advisor call is archived before actual execution; history excludes D.
    Original reports are preserved through the caller's separate output directory.
    """
    from scripts.run_official_experiment import execute, actions_rule, ci_bootstrap, ROOT, sha
    archives=[]
    archive_path=out/'agent_policy_archive.jsonl'
    archive_path.write_text('',encoding='utf-8')
    mode='mock_offline' if advisor.mock_offline else 'llm_api'

    def choose(stage, issue, timestamps, pv, load, prices, history, soc):
        forecast={'issued_at':issue,'target_timestamps':timestamps,
                  'pv_forecast_kw':list(map(float,pv)),'load_forecast_kw':list(map(float,load)),
                  'price_forecast_cny_per_kwh':list(map(float,prices)),
                  'initial_soc':soc,'resolution_minutes':15}
        policy=advisor.advise(forecast,history)
        archives.append({'stage':stage,'forecast':forecast,'historical_summary':history,
                         'policy':policy,'diagnostics':dict(advisor.last_diagnostics)})
        # Persist the issued forecast/policy before settlement reads today's actuals.
        with archive_path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(archives[-1],ensure_ascii=False,allow_nan=False)+'\n')
        return policy

    def history_forecast(history):
        return (history.groupby('slot').pv.mean().reindex(range(96)).to_numpy(),
                history.groupby('slot').load.mean().reindex(range(96)).to_numpy())

    def describe(history):
        return {'history_end':history.ts.iloc[-1].isoformat(),
                'history_days':int(history.date.nunique()),
                'pv_mean_kw':float(history.pv.mean()),'load_mean_kw':float(history.load.mean())}

    def check(result):
        if result['soc_violations'] or not .10-1e-9<=result['end_soc']<=.95+1e-9:
            raise RuntimeError('execution safety check failed')

    def read(stage):return json.loads((out/stage/'summary.json').read_text())
    def write(stage,value):
        (out/stage).mkdir(exist_ok=True)
        (out/stage/'summary.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

    dates=sorted(df.date.unique()); soc=.5; rows=[]
    for day in dates[-30:]:
        actual=df[df.date==day].sort_values('slot'); hist=df[df.date<day].sort_values('ts')
        pv,load=history_forecast(hist); prices=actual.price.to_numpy()
        policy=choose('P0', (actual.ts.iloc[0]-pd.Timedelta(minutes=15)).isoformat(),
                      [t.isoformat() for t in actual.ts],pv,load,prices,describe(hist),soc)
        result=execute(actions_llm_guided(pv,load,prices,policy,initial_soc=soc),
                       actual.pv.to_numpy(),actual.load.to_numpy(),prices,soc=soc)
        check(result); soc=result['end_soc']
        rows.append({'date':str(day),'arm':'E_LLM',**result})
    p0days=pd.read_csv(out/'p0_replay/day_results.csv')
    costs=np.array([r['cost_yuan'] for r in rows])
    baseline={a:p0days[p0days.arm==a].set_index('date').loc[[r['date'] for r in rows]].cost_yuan.to_numpy() for a in ['E0','E1','E2','E3']}
    for row,base in zip(rows,baseline['E0']):row['saving_vs_e0_yuan']=float(base-row['cost_yuan'])
    pd.concat([p0days,pd.DataFrame(rows)],ignore_index=True).to_csv(out/'p0_replay/day_results.csv',index=False)
    p0=read('p0_replay');p0['totals_yuan']['E_LLM']=float(costs.sum());p0['saving_vs_e0_yuan']['E_LLM']=float((baseline['E0']-costs).sum())
    p0['E_LLM']={'mode':mode,'soc_violations':int(sum(r['soc_violations'] for r in rows)),
                 'end_soc':soc,'comparisons':{a:comparison(costs,v,ci_bootstrap) for a,v in baseline.items()},
                 'conclusion_boundary':'mock is deterministic policy plumbing, not evidence of LLM economic advantage',
                 'offset_unit':'signed 15-minute slots; -4..4; positive delays discharge',
                 'reserve_scope':'forecast-side discharge floor; physical execution remains 10%..95%'}
    # P1: same physical plans, monthly demand settlement only; no claim of peak-shaving policy optimization.
    p1=read('p1_demand_charge');rate=p1['demand_rate_yuan_per_kw_month'];peak=max(r['peak_import_kw'] for r in rows)
    total=float(costs.sum()+rate*peak)
    p1['arms'].append({'arm':'E_LLM','energy_cost_yuan':float(costs.sum()),'max_demand_kw':peak,
                       'demand_charge_yuan':peak*rate,'total_cost_yuan':total,
                       'saving_vs_e0_yuan':p1['arms'][0]['total_cost_yuan']-total})
    p1['E_LLM']={'mode':mode,'scope':'same P0 plan with assumed demand settlement; not demand-aware optimization'}
    write('p1_demand_charge',p1)
    p1days=pd.read_csv(out/'p1_demand_charge/day_results.csv')
    llm_demand=pd.DataFrame([{'date':r['date'],'arm':'E_LLM','energy_cost_yuan':r['cost_yuan'],'demand_kw':r['peak_import_kw']} for r in rows])
    pd.concat([p1days,llm_demand],ignore_index=True).to_csv(out/'p1_demand_charge/day_results.csv',index=False)
    # P3: compare costs on the exact same P0 dates against the already-run LP.
    p3day=pd.read_csv(out/'p3_lp/day_results.csv');p3=read('p3_lp')
    p3day['E_LLM_cost_yuan']=costs;p3day['E_LLM_saving_yuan']=baseline['E0']-costs
    p3day.to_csv(out/'p3_lp/day_results.csv',index=False)
    p3['E_LLM']={'mode':mode,'soc_violations':0,'vs_E2':comparison(costs,baseline['E2'],ci_bootstrap),
                 'vs_LP':comparison(costs,p3day.LP_cost_yuan,ci_bootstrap)};write('p3_lp',p3)
    # P2: actual PJM price settlement, with disclosed synthetic cross-region scenario alignment.
    prices_frame=pd.read_csv(ROOT/'reports/price_forecast/t3_price_benchmark/predictions.csv')
    prices_frame=prices_frame[prices_frame.model=='lgbm_calibrated'].sort_values('ts')
    spot=[];s_rule=s_agent=.5
    for index,(day,part) in enumerate(prices_frame.groupby('day',sort=True)):
        if len(part)!=24:raise ValueError('P2 requires 24 hourly prices')
        source_date=dates[30+index%(len(dates)-30)]
        actual=df[df.date==source_date].sort_values('slot');hist=df[df.date<source_date].sort_values('ts')
        pv,load=history_forecast(hist)
        forecast_price=np.repeat(part.p50.to_numpy(),4); settlement_price=np.repeat(part.actual.to_numpy(),4)
        times=pd.date_range(pd.Timestamp(part.ts.iloc[0]),periods=96,freq='15min')
        policy=choose('P2',(times[0]-pd.Timedelta(minutes=15)).isoformat(),[t.isoformat() for t in times],pv,load,forecast_price,
                      {'history_days':int(hist.date.nunique()),'history_end':(times[0]-pd.Timedelta(minutes=15)).isoformat(),
                       'pv_mean_kw':float(hist.pv.mean()),'load_mean_kw':float(hist.load.mean()),
                       'source_day':str(source_date),'source_history_timezone':'Asia/Shanghai',
                       'alignment':'synthetic source-day alignment to PJM day; not Guangdong actuals'},s_agent)
        rule=execute(actions_rule(pv,load,forecast_price,pv,load),actual.pv.to_numpy(),actual.load.to_numpy(),settlement_price,soc=s_rule)
        llm=execute(actions_llm_guided(pv,load,forecast_price,policy,initial_soc=s_agent),actual.pv.to_numpy(),actual.load.to_numpy(),settlement_price,soc=s_agent)
        check(llm);s_rule=rule['end_soc'];s_agent=llm['end_soc']
        spot.append({'date':day,'source_date':str(source_date),'E2_cost_yuan':rule['cost_yuan'],'E_LLM_cost_yuan':llm['cost_yuan'],'soc_violations':llm['soc_violations']})
    spot=pd.DataFrame(spot);spot.to_csv(out/'p2_spot_price/agent_day_results.csv',index=False)
    p2={'E_LLM':{'mode':mode,'n_days':len(spot),'soc_violations':int(spot.soc_violations.sum()),
         'vs_E2':comparison(spot.E_LLM_cost_yuan,spot.E2_cost_yuan,ci_bootstrap),
         'scope':'PJM Zone1 calibrated forecasts/actual settlement + synthetic PV/load alignment; demo only',
         'limitation':'Fixed TOU charge/discharge thresholds retained; PJM price scale may never reach peak threshold. Equal costs do not establish LLM effectiveness.'}}
    write('p2_spot_price',p2)
    # P4 new comparison cohort: 30-day warmup; 365 synthetic days, all forecasts from past only.
    # Existing P4 had actual-load/cold-start approximations; keep its results separate.
    annual=[];history=df[df.date.isin(dates[:30])].copy();s_rule=s_agent=.5
    for day_index in range(365):
        source_date=dates[30+day_index%(len(dates)-30)]
        actual=df[df.date==source_date].sort_values('slot');pv,load=history_forecast(history);pr=actual.price.to_numpy()
        simulation_day=pd.Timestamp('2021-01-01',tz='Asia/Shanghai')+pd.Timedelta(days=day_index)
        times=pd.date_range(simulation_day,periods=96,freq='15min');deg=1-.02*day_index/365
        policy=choose('P4',(times[0]-pd.Timedelta(minutes=15)).isoformat(),[t.isoformat() for t in times],pv,load,pr,
                      {'history_days':30+day_index,'history_available_before':times[0].isoformat(),
                       'source_day':str(source_date),'provenance':'synthetic cyclic replay; forecasts fitted to previous simulated days'},s_agent)
        rule=execute(actions_rule(pv,load,pr,pv,load),actual.pv.to_numpy(),actual.load.to_numpy(),pr,soc=s_rule,deg=deg)
        llm=execute(actions_llm_guided(pv,load,pr,policy,initial_soc=s_agent,deg=deg),actual.pv.to_numpy(),actual.load.to_numpy(),pr,soc=s_agent,deg=deg)
        check(llm);s_rule=rule['end_soc'];s_agent=llm['end_soc']
        e0=execute([('idle',0.)]*96,actual.pv.to_numpy(),actual.load.to_numpy(),pr,deg=deg)
        annual.append({'date':str(simulation_day.date()),'source_date':str(source_date),'E0_cost_yuan':e0['cost_yuan'],
                       'E2_cost_yuan':rule['cost_yuan'],'E_LLM_cost_yuan':llm['cost_yuan'],'soc_violations':llm['soc_violations'],
                       'E2_end_soc':s_rule,'E_LLM_end_soc':s_agent})
        history=pd.concat([history,actual],ignore_index=True)
    annual=pd.DataFrame(annual);annual.to_csv(out/'p4_annual/agent_day_results.csv',index=False)
    p4=read('p4_annual');p4['E_LLM']={'mode':mode,'days':365,'soc_violations':int(annual.soc_violations.sum()),
        'annual_saving_yuan':float((annual.E0_cost_yuan-annual.E_LLM_cost_yuan).sum()),
        'same_cohort_E2_annual_saving_yuan':float((annual.E0_cost_yuan-annual.E2_cost_yuan).sum()),
        'vs_E2':comparison(annual.E_LLM_cost_yuan,annual.E2_cost_yuan,ci_bootstrap),
        'scope':'independent synthetic cohort with causal 30-day warmup; not comparable to legacy annual total; no real investment claim'}
    write('p4_annual',p4)
    p0['E_LLM']['stages']={stage:{'summary':str(Path(folder)/'summary.json'),'status':'executed'} for stage,folder in
                          [('P0','p0_replay'),('P1','p1_demand_charge'),('P2','p2_spot_price'),('P3','p3_lp'),('P4','p4_annual')]}
    write('p0_replay',p0)
    meta=json.loads((out/'run_meta.json').read_text());meta['agent']={'arm':'E_LLM','mode':mode,'calls':len(archives),
        'fallback_calls':sum(r['diagnostics']['status']=='fallback' for r in archives),
        'safety_executor':'execute unchanged; SOC_MIN=.10 SOC_MAX=.95',
        'policy_archive':'agent_policy_archive.jsonl','env_secrets':'not archived'}
    meta['implementation_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'backend/agents/llm_energy_agent.py',ROOT/'backend/scripts/run_official_experiment.py',Path(__file__)]}
    meta['p2_prediction_sha256']=sha(ROOT/'reports/price_forecast/t3_price_benchmark/predictions.csv')
    meta['policy_archive_sha256']=sha(archive_path)
    (out/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


def extend_formulator_experiment(df, out: Path, advisor, *, event_mode="events_only"):
    """Run the isolated event-day formulator arm beside the sealed E2 cohort.

    The default event fixture is explicitly synthetic because the demo site has
    no dated Guangdong event feed. Event labels are assigned before settlement,
    and ordinary days never call the advisor.
    """
    from scripts.run_official_experiment import execute, actions_rule, ci_bootstrap
    baseline = pd.read_csv(out / "p0_replay/day_results.csv")
    dates = sorted(df.date.unique())[-30:]
    event_positions = {5: "maintenance", 12: "cold_wave", 19: "curtailment", 26: "negative_price"}
    rows, records = [], []
    soc = .5
    for index, day in enumerate(dates):
        actual = df[df.date == day].sort_values("slot")
        hist = df[df.date < day].sort_values("ts")
        pv = hist.groupby("slot").pv.mean().reindex(range(96)).fillna(0).to_numpy()
        load = hist.groupby("slot").load.mean().reindex(range(96)).fillna(0).to_numpy()
        prices = actual.price.to_numpy()
        kind = event_positions.get(index, "none")
        event = {"kind": kind, "severity": .8 if kind != "none" else 0.0,
                 "available_at": (actual.ts.iloc[0] - pd.Timedelta(minutes=15)).isoformat(),
                 "source": "synthetic_demo_fixture"}
        issue = event["available_at"]
        forecast = {"issued_at": issue, "target_timestamps": [t.isoformat() for t in actual.ts],
                    "pv_forecast_kw": list(map(float, pv)), "load_forecast_kw": list(map(float, load)),
                    "price_forecast_cny_per_kwh": list(map(float, prices)),
                    "initial_soc": soc, "resolution_minutes": 15}
        history = {"history_end": hist.ts.iloc[-1].isoformat(),
                   "history_days": int(hist.date.nunique()), "event_source": event["source"]}
        use_formulator = event_mode == "all_days" or is_event_day(event)
        policy = {"scenario": "none", "mode": "rule", "soc_min_safe": .10,
                  "peak_discharge_offset": 0, "reserve_weight": 0.0,
                  "discharge_window_start": 0, "discharge_window_end": 95,
                  "manual_review_required": False}
        if use_formulator:
            policy = advisor.formulate(forecast, history, event)
        diagnostics = dict(advisor.last_diagnostics) if use_formulator else {"mode": "rule", "status": "not_called"}
        actions = (actions_formulator_guided(pv, load, prices, policy, initial_soc=soc)
                   if use_formulator else actions_rule(actual.pv.to_numpy(), actual.load.to_numpy(), prices, pv, load))
        result = execute(actions, actual.pv.to_numpy(), actual.load.to_numpy(), prices, soc=soc)
        if result["soc_violations"] or not .10 - 1e-9 <= result["end_soc"] <= .95 + 1e-9:
            raise RuntimeError("formulator execution safety check failed")
        fallback = diagnostics.get("status") == "fallback"
        row = {"date": str(day), "arm": "E_LLM_FORMULATOR", "event_day": use_formulator,
               "event_kind": kind, "formulator_mode": policy.get("mode", "rule"),
               "manual_review_required": bool(policy.get("manual_review_required", False)),
               "fallback": fallback, "policy_changed_by_safety_layer": False, **result}
        rows.append(row)
        records.append({**row, "diagnostics": diagnostics})
        soc = result["end_soc"]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "p0_replay/formulator_day_results.csv", index=False)
    costs = frame.cost_yuan.to_numpy()
    e2 = baseline[baseline.arm == "E2"].set_index("date").loc[frame.date].cost_yuan.to_numpy()
    deltas = e2 - costs
    p0_path = out / "p0_replay/summary.json"
    p0 = json.loads(p0_path.read_text())
    p0.setdefault("totals_yuan", {})["E_LLM_FORMULATOR"] = float(costs.sum())
    p0.setdefault("saving_vs_e2_yuan", {})["E_LLM_FORMULATOR"] = float(deltas.sum())
    p0["E_LLM_FORMULATOR"] = {
        "arm": "E_LLM_FORMULATOR", "event_mode": event_mode, "n_days": len(frame),
        "event_day_count": int(frame.event_day.sum()), "event_source": "synthetic_demo_fixture",
        "total_cost_yuan": float(costs.sum()), "total_saving_vs_e2_yuan": float(deltas.sum()),
        "manual_intervention_count": count_manual_interventions(records),
        "soc_violations": int(frame.soc_violations.sum()),
        "worst_day_cost_yuan": tail_metrics(costs, .90)["worst_day_cost_yuan"],
        "cvar_90": tail_metrics(costs, .90), "cvar_95": tail_metrics(costs, .95),
        "event_day_cost_yuan": float(frame.loc[frame.event_day, "cost_yuan"].sum()),
        "event_day_increment_vs_e2_yuan": float(deltas[frame.event_day.to_numpy()].sum()),
        "vs_e2_daily_bootstrap_ci95_yuan": ci_bootstrap(deltas),
        "conclusion_boundary": "synthetic event fixture; protocol validation, not Guangdong event-effect evidence",
    }
    p0_path.write_text(json.dumps(p0, ensure_ascii=False, indent=2), encoding="utf-8")
    # Add the same arm's declared scope to each stage summary without rewriting prior values.
    for folder in ("p1_demand_charge", "p2_spot_price", "p3_lp", "p4_annual"):
        path = out / folder / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            summary["E_LLM_FORMULATOR"] = {"source": "P0 event-day cohort", "event_day_count": int(frame.event_day.sum()),
                                            "manual_intervention_count": count_manual_interventions(records),
                                            "soc_violations": int(frame.soc_violations.sum()),
                                            "scope": "summary linkage only; no stage-specific rerun"}
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "formulator_policy_archive.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in records) + "\n", encoding="utf-8")
    return p0["E_LLM_FORMULATOR"]
