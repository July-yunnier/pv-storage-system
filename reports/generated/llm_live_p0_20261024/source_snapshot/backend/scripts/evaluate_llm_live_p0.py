"""Real local LLM P0 pilot in an isolated checkout."""
import argparse,collections,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'backend'))
from agents.llm_energy_agent import LLMStrategicAdvisor,compact_forecast
from scripts.run_official_experiment import execute,ci_bootstrap
from scripts.llm_experiment import actions_llm_guided,comparison

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
 out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'reports/official_experiment/llm_mock'
 records=[json.loads(line) for line in (source/'agent_policy_archive.jsonl').read_text().splitlines()]
 records=[r for r in records if r['stage']=='P0']
 data=ROOT/'reports/official_experiment/data/full_scene.csv';df=pd.read_csv(data)
 baseline=pd.read_csv(source/'p0_replay/day_results.csv')
 advisor=LLMStrategicAdvisor(mock_offline=False);soc=.5;rows=[];diagnostics=[]
 with (out/'policy_archive.jsonl').open('w') as archive:
  for i,r in enumerate(records):
   f=r['forecast'];f['initial_soc']=soc;t=time.monotonic();policy=advisor.advise(f,r['historical_summary']);elapsed=time.monotonic()-t
   audit={'forecast':f,'api_forecast':compact_forecast(f),'historical_summary':r['historical_summary'],'policy':policy,'diagnostics':dict(advisor.last_diagnostics),'elapsed_seconds':elapsed}
   archive.write(json.dumps(audit,ensure_ascii=False,allow_nan=False)+'\n');archive.flush()
   day=pd.Timestamp(f['target_timestamps'][0]).date().isoformat();actual=df[df.date==day].sort_values('slot')
   pv=(actual.pv_power_kw/50).clip(0,1000).to_numpy();load=actual.load_power_kw.clip(0,800).to_numpy();prices=f['price_forecast_cny_per_kwh']
   result=execute(actions_llm_guided(f['pv_forecast_kw'],f['load_forecast_kw'],prices,policy,initial_soc=soc),pv,load,prices,soc=soc)
   if result['soc_violations']:raise RuntimeError('safety violation')
   soc=result['end_soc'];rows.append({'date':day,'arm':'E_LLM',**result,**policy});diagnostics.append(advisor.last_diagnostics.copy())
   print(json.dumps({'day':i+1,'date':day,'policy':policy,'diagnostics':advisor.last_diagnostics,'seconds':round(elapsed,2)}),flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(out/'day_results.csv',index=False);costs=frame.cost_yuan.to_numpy()
 comparators={a:baseline[baseline.arm==a].set_index('date').loc[frame.date].cost_yuan.to_numpy() for a in ['E0','E1','E2','E3']}
 totals={a:float(v.sum()) for a,v in comparators.items()};totals['E_LLM']=float(costs.sum())
 summary={'mode':'llm_api','model':'qwen2.5:7b','prompt_version':'local-pilot-v1','window':[frame.date.iloc[0],frame.date.iloc[-1]],'n_days':len(frame),'totals_yuan':totals,'soc_violations':int(frame.soc_violations.sum()),'successful_calls':sum(d['status']=='ok' for d in diagnostics),'fallback_calls':sum(d['status']=='fallback' for d in diagnostics),'comparisons':{a:comparison(costs,v,ci_bootstrap) for a,v in comparators.items()},'policies':dict(collections.Counter(json.dumps({'soc_min_safe':r['soc_min_safe'],'peak_discharge_offset':r['peak_discharge_offset']},sort_keys=True) for r in rows)),'scope':'30-day demo P0 pilot only; revised prompt, no real Guangdong saving promise; planning confounds remain'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 paths=[Path(__file__),ROOT/'backend/agents/llm_energy_agent.py',ROOT/'backend/scripts/llm_experiment.py',ROOT/'backend/scripts/run_official_experiment.py']
 meta={'generated_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'bootstrap':{'n':2000,'seed':42},'max_tokens':96,'timeout_seconds':120,'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
 (out/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
