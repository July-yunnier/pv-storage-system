#!/usr/bin/env python3
"""Execute the formal P0-P4 protocol on the sealed CPU scenario data.

This runner is deliberately offline and dependency-light (pandas/numpy/scipy).
It records the data provenance and clearly labels synthetic/demo inputs.

--agent llm --mock-offline runs a deterministic bounded policy experiment.
--agent llm --no-mock-offline calls the OpenAI-compatible endpoint from .env.
Agent results default to separate llm_mock/ or llm_api/ output directories.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linprog, brentq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
DEFAULT_DATA=ROOT/'reports/official_experiment/data/full_scene.csv'
OUT=ROOT/'reports/official_experiment'
PRICES={'sharp':1.25+0.21,'peak':1.05+0.21,'flat':0.65+0.21,'valley':0.28+0.21}
DT=.25
PCS=500.0; CAP=1000.0; SOC_MIN=.10; SOC_MAX=.95; ETA=math.sqrt(.88)
BOOT=2000; SEED=42

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def ci_bootstrap(x, seed=SEED, n=BOOT):
 x=np.asarray(x,float); rng=np.random.default_rng(seed)
 if len(x)==0:return [None,None]
 vals=np.mean(x[rng.integers(0,len(x),size=(n,len(x)))],axis=1)
 return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def execute(actions,pv,load,prices,soc=.5,deg=1.0):
 soc_e=soc*CAP*deg; lo=SOC_MIN*CAP*deg; hi=SOC_MAX*CAP*deg
 cost=0.; peak=0.; pvself=0.; pvtot=0.; dis=0.; cur=0.; viol=0.; trace=[]
 for (a,planned_kw),pv_kw,load_kw,price in zip(actions,pv,load,prices):
  c=d=0.
  if a=='charge': c=min(planned_kw,PCS,(hi-soc_e)/ETA/DT)
  elif a=='discharge': d=min(planned_kw,PCS,(soc_e-lo)*ETA/DT,max(load_kw-pv_kw,0))
  soc_e+=c*DT*ETA-d*DT/ETA
  if soc_e<lo-1e-6 or soc_e>hi+1e-6:viol+=1
  soc_e=min(max(soc_e,lo),hi)
  net=load_kw-pv_kw+c-d; grid=max(net,0); export=max(-net,0)
  cost+=grid*DT*price; peak=max(peak,grid); pvtot+=max(pv_kw,0)*DT
  pvself+=max(min(pv_kw,load_kw+c),0)*DT; dis+=d*DT; cur+=export*DT
  trace.append(soc_e/(CAP*deg))
 return {'cost_yuan':cost,'peak_import_kw':peak,'pv_self_use_ratio':pvself/pvtot if pvtot else 1.,'discharge_kwh':dis,'curtailment_kwh':cur,'soc_violations':viol,'end_soc':trace[-1] if trace else soc}

def actions_rule(pv,load,prices,forecast_pv,forecast_load):
 # forecast-driven rule: charge in valley or surplus PV; discharge in sharp/peak when net load exists.
 out=[]
 for p,l,pr,fp,fl in zip(pv,load,prices,forecast_pv,forecast_load):
  net=fl-fp
  if pr<=PRICES['valley']+1e-6 or net < -50: out.append(('charge',PCS))
  elif pr>=PRICES['peak']-1e-6 and net>0: out.append(('discharge',PCS))
  else: out.append(('idle',0.))
 return out

def fixed_actions(prices):
 return [('charge',PCS) if x<=PRICES['valley']+1e-6 else ('discharge',PCS) if x>=PRICES['sharp']-1e-6 else ('idle',0.) for x in prices]

def lp_actions(pv,load,prices,forecast_pv,forecast_load,initial_soc=.5):
 n=len(prices); eta=ETA; dt=DT
 # variables: grid[0:n], charge[0:n], discharge[0:n], energy[0:n+1]
 ng, nc, nd, ne = n, n, n, n+1
 off_c=ng; off_d=ng+nc; off_e=ng+nc+nd; total=ng+nc+nd+ne
 cobj=np.zeros(total); cobj[:n]=np.asarray(prices,float)*dt
 Aeq=[]; beq=[]
 for k in range(n):
  row=np.zeros(total); row[k]=1; row[off_c+k]=-1; row[off_d+k]=1
  Aeq.append(row); beq.append(float(forecast_load[k]-forecast_pv[k]))
 for k in range(n):
  row=np.zeros(total); row[off_e+k+1]=1; row[off_e+k]=-1; row[off_c+k]=-dt*eta; row[off_d+k]=dt/eta
  Aeq.append(row); beq.append(0.)
 row=np.zeros(total); row[off_e]=1; Aeq.append(row); beq.append(initial_soc*CAP)
 bounds=[(0,None)]*ng+[(0,PCS)]*nc+[(0,PCS)]*nd+[(SOC_MIN*CAP,SOC_MAX*CAP)]*ne
 res=linprog(cobj,A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,method='highs')
 if not res.success:return [('idle',0.)]*n
 return [('charge',float(res.x[off_c+i])) if res.x[off_c+i]>1e-6 else ('discharge',float(res.x[off_d+i])) if res.x[off_d+i]>1e-6 else ('idle',0.) for i in range(n)]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=DEFAULT_DATA)
 ap.add_argument('--agent',choices=['llm','llm-formulator'])
 ap.add_argument('--mock-offline',action=argparse.BooleanOptionalAction,default=True,help='Default: deterministic no-network mock; --no-mock-offline calls configured API')
 ap.add_argument('--event-mode',choices=['events-only','all-days'],default='events-only',help='Formulator invocation policy; applies to --agent llm-formulator')
 ap.add_argument('--output-dir',type=Path)
 args=ap.parse_args()
 out=args.output_dir or (OUT/('llm_formulator_mock' if args.agent=='llm-formulator' and args.mock_offline else 'llm_formulator_api' if args.agent=='llm-formulator' else 'llm_mock' if args.mock_offline else 'llm_api') if args.agent else OUT)
 out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(args.data); df['ts']=pd.to_datetime(df['ts']); df['date']=pd.to_datetime(df['date']).dt.date
 df['pv']=(df.pv_power_kw/50.0).clip(0,CAP); df['load']=df.load_power_kw.clip(0,800); df['price']=df.tariff_period.map(PRICES)
 dates=sorted(df.date.unique()); test=dates[-30:]; hist=set(dates[:-30]);
 daily=[]; forecast_rows=[]; p0_rows=[]; p1_rows=[]; p3_rows=[]
 soc1=soc2=soc3=socl=0.5
 for di,day in enumerate(test):
  d=df[df.date==day].sort_values('slot'); prev=df[df.date==dates[dates.index(day)-1]].sort_values('slot')
  history=df[df.date<day];
  slot_mean=history.groupby('slot').pv.mean().reindex(range(96)).fillna(0).to_numpy()
  load_mean=history.groupby('slot').load.mean().reindex(range(96)).fillna(0).to_numpy()
  dp=prev.pv.to_numpy(); dl=prev.load.to_numpy(); actual_pv=d.pv.to_numpy(); actual_load=d.load.to_numpy(); prices=d.price.to_numpy()
  for name,fc in [('slot_mean',slot_mean),('day_persistence',dp)]:
   forecast_rows.append({'date':str(day),'method':name,'mae_kw':float(np.mean(abs(fc-actual_pv))),'rmse_kw':float(np.sqrt(np.mean((fc-actual_pv)**2))),'bias_kw':float(np.mean(fc-actual_pv)),'skill_vs_day_persistence':None if name=='day_persistence' else float(1-np.sqrt(np.mean((fc-actual_pv)**2))/np.sqrt(np.mean((dp-actual_pv)**2)))})
  e0=execute([('idle',0)]*96,actual_pv,actual_load,prices)
  e1=execute(fixed_actions(prices),actual_pv,actual_load,prices,soc=soc1)
  e2=execute(actions_rule(actual_pv,actual_load,prices,slot_mean,load_mean),actual_pv,actual_load,prices,soc=soc2)
  e3=execute(lp_actions(actual_pv,actual_load,prices,actual_pv,actual_load,soc3),actual_pv,actual_load,prices,soc=soc3)
  lp=execute(lp_actions(actual_pv,actual_load,prices,slot_mean,load_mean,socl),actual_pv,actual_load,prices,soc=socl)
  soc1,soc2,soc3,socl=e1['end_soc'],e2['end_soc'],e3['end_soc'],lp['end_soc']
  for arm,e in [('E0',e0),('E1',e1),('E2',e2),('E3',e3)]:
   daily.append({'date':str(day),'arm':arm,**e,'saving_vs_e0_yuan':e0['cost_yuan']-e['cost_yuan']})
  p3_rows.append({'date':str(day),'E2_cost_yuan':e2['cost_yuan'],'LP_cost_yuan':lp['cost_yuan'],'E2_saving_yuan':e0['cost_yuan']-e2['cost_yuan'],'LP_saving_yuan':e0['cost_yuan']-lp['cost_yuan']})
  p1_rows.append({'date':str(day),'arm':'E0','energy_cost_yuan':e0['cost_yuan'],'demand_kw':e0['peak_import_kw']})
  for arm,e in [('E1',e1),('E2',e2),('E3',e3)]:p1_rows.append({'date':str(day),'arm':arm,'energy_cost_yuan':e['cost_yuan'],'demand_kw':e['peak_import_kw']})
 Path(out/'p0_replay').mkdir(exist_ok=True); Path(out/'p1_demand_charge').mkdir(exist_ok=True); Path(out/'p3_lp').mkdir(exist_ok=True); Path(out/'p4_annual').mkdir(exist_ok=True)
 pd.DataFrame(daily).to_csv(out/'p0_replay/day_results.csv',index=False); pd.DataFrame(forecast_rows).to_csv(out/'p0_replay/forecast_metrics.csv',index=False)
 pd.DataFrame(p1_rows).to_csv(out/'p1_demand_charge/day_results.csv',index=False); pd.DataFrame(p3_rows).to_csv(out/'p3_lp/day_results.csv',index=False)
 def arm_total(arm): return float(sum(x['cost_yuan'] for x in daily if x['arm']==arm))
 totals={a:arm_total(a) for a in ['E0','E1','E2','E3']}
 delta=np.array([next(x['saving_vs_e0_yuan'] for x in daily if x['date']==str(day) and x['arm']=='E2')-next(x['saving_vs_e0_yuan'] for x in daily if x['date']==str(day) and x['arm']=='E1') for day in test])
 p0_summary={'window':[str(test[0]),str(test[-1])],'n_days':30,'totals_yuan':totals,'saving_vs_e0_yuan':{a:totals['E0']-v for a,v in totals.items()},'E2_minus_E1_daily_mean_yuan':float(delta.mean()),'E2_minus_E1_bootstrap_ci95_yuan':ci_bootstrap(delta),'E2_vs_E3_upper_bound_attainment':(totals['E0']-totals['E2'])/(totals['E0']-totals['E3']) if totals['E0']!=totals['E3'] else None}
 (out/'p0_replay/summary.json').write_text(json.dumps(p0_summary,ensure_ascii=False,indent=2))
 # P1 demand charge; illustrative Guangdong demand tariff disclosed in metadata.
 demand_rate=35.0; p1=[]
 for arm in ['E0','E1','E2','E3']:
  rows=[x for x in p1_rows if x['arm']==arm]; energy=sum(x['energy_cost_yuan'] for x in rows); demand=max(x['demand_kw'] for x in rows)*demand_rate; p1.append({'arm':arm,'energy_cost_yuan':energy,'max_demand_kw':max(x['demand_kw'] for x in rows),'demand_charge_yuan':demand,'total_cost_yuan':energy+demand,'saving_vs_e0_yuan':None})
 e0=p1[0]['total_cost_yuan']
 for r in p1:r['saving_vs_e0_yuan']=e0-r['total_cost_yuan']
 (out/'p1_demand_charge/summary.json').write_text(json.dumps({'demand_rate_yuan_per_kw_month':demand_rate,'n_days':30,'arms':p1,'status':'protocol_validation_assumption'},ensure_ascii=False,indent=2))
 # P3 summary
 p3df=pd.DataFrame(p3_rows); p3summary={'n_days':30,'E2_saving_yuan':float(p3df.E2_saving_yuan.sum()),'LP_saving_yuan':float(p3df.LP_saving_yuan.sum()),'LP_increment_vs_E2_yuan':float((p3df.LP_saving_yuan-p3df.E2_saving_yuan).sum()),'LP_increment_bootstrap_ci95_yuan':ci_bootstrap((p3df.LP_saving_yuan-p3df.E2_saving_yuan).to_numpy())}; (out/'p3_lp/summary.json').write_text(json.dumps(p3summary,ensure_ascii=False,indent=2))
 # P4: 120-day synthetic annual expansion, cyclic replay with 2% annual degradation.
 full=[]
 for year_day in range(365):
  src=df[df.date==dates[year_day%len(dates)]].sort_values('slot'); pv=src.pv.to_numpy(); load=src.load.to_numpy(); pr=src.price.to_numpy();
  fc=df[df.date<dates[(year_day%len(dates))]].groupby('slot').pv.mean().reindex(range(96)).fillna(df.pv.mean()).to_numpy() if year_day%len(dates)>2 else pv
  fl=src.load.to_numpy(); e0=execute([('idle',0)]*96,pv,load,pr,deg=1-.02*year_day/365); e2=execute(actions_rule(pv,load,pr,fc,fl),pv,load,pr,deg=1-.02*year_day/365)
  full.append({'day':year_day+1,'E0_cost_yuan':e0['cost_yuan'],'E2_cost_yuan':e2['cost_yuan'],'saving_yuan':e0['cost_yuan']-e2['cost_yuan'],'degradation':1-.02*year_day/365})
 pd.DataFrame(full).to_csv(out/'p4_annual/day_results.csv',index=False); annual=float(sum(x['saving_yuan'] for x in full)); capex=1200000.; cash=[-capex]+[annual*.98**i-capex*.01 for i in range(10)]
 def npv(rate):return sum(v/(1+rate)**i for i,v in enumerate(cash))
 irr=None
 try: irr=brentq(npv,-.99,2)
 except ValueError: pass
 (out/'p4_annual/summary.json').write_text(json.dumps({'days_simulated':365,'source_days':120,'annual_saving_yuan':annual,'capex_yuan':capex,'annual_opex_yuan':capex*.01,'degradation_assumption':'2%/year','npv_10y_yuan':npv(.08),'irr':irr,'status':'synthetic_annual_protocol_validation'},ensure_ascii=False,indent=2))
 # P2 existing result copy and metadata
 p2src=ROOT/'reports/price_forecast/t3_price_benchmark/metrics.csv'; (out/'p2_spot_price').mkdir(exist_ok=True)
 if p2src.exists(): (out/'p2_spot_price/metrics.csv').write_bytes(p2src.read_bytes())
 meta={'generated_at':datetime.now(timezone.utc).isoformat(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'python':subprocess.check_output([__import__('sys').executable,'--version'],text=True).strip(),'data_sha256':sha(args.data),'data_window':[str(dates[0]),str(dates[-1])],'test_window':[str(test[0]),str(test[-1])],'site_id':'S-DONGGUAN-DEMO','units':'CNY/kWh, kW, kWh','timezone':'Asia/Shanghai','data_provenance':{'pv':'public/scenario observed proxy','load':'synthetic two-shift factory','tariff':'synthetic Guangdong policy template','p2':'GEFCom2014-P PJM Zone1 demo'},'bootstrap':{'n':BOOT,'seed':SEED}}
 (out/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 if args.agent=='llm':
  from agents.llm_energy_agent import LLMStrategicAdvisor
  from scripts.llm_experiment import extend_experiment
  extend_experiment(df,out,LLMStrategicAdvisor(mock_offline=args.mock_offline))
  p0_summary=json.loads((out/'p0_replay/summary.json').read_text())
  p1=json.loads((out/'p1_demand_charge/summary.json').read_text())
  p3summary=json.loads((out/'p3_lp/summary.json').read_text())
 if args.agent=='llm-formulator':
  from agents.llm_energy_agent import LLMStrategicAdvisor
  from scripts.llm_experiment import extend_formulator_experiment
  result=extend_formulator_experiment(df,out,LLMStrategicAdvisor(mock_offline=args.mock_offline),
                                      event_mode=args.event_mode)
  meta=json.loads((out/'run_meta.json').read_text())
  meta['agent']={'arm':'E_LLM_FORMULATOR','mode':'mock_offline' if args.mock_offline else 'llm_api',
                 'event_mode':args.event_mode,'event_source':'synthetic_demo_fixture',
                 'safety_executor':'execute unchanged; SOC_MIN=.10 SOC_MAX=.95',
                 'metrics':['cvar_90','cvar_95','worst_day_cost_yuan','manual_intervention_count']}
  (out/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
  p0_summary=json.loads((out/'p0_replay/summary.json').read_text())
  p1=json.loads((out/'p1_demand_charge/summary.json').read_text())
  p3summary=json.loads((out/'p3_lp/summary.json').read_text())
 print(json.dumps({'p0':p0_summary,'p1':p1,'p3':p3summary,'p4':json.loads((out/'p4_annual/summary.json').read_text())},ensure_ascii=False,indent=2))
from scripts.llm_experiment import actions_llm_guided

if __name__=='__main__':main()
