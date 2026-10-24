"""Add fixed-policy controls on the same planner, without more model calls."""
import hashlib,json,sys,urllib.request
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'backend'))
from scripts.run_official_experiment import execute,ci_bootstrap
from scripts.llm_experiment import actions_llm_guided,comparison
out=ROOT/'reports/generated/llm_live_p0_20261024'
summary=json.loads((out/'summary.json').read_text());data=pd.read_csv(ROOT/'reports/official_experiment/data/full_scene.csv')
records=[json.loads(line) for line in (out/'policy_archive.jsonl').read_text().splitlines()]
agent=pd.read_csv(out/'day_results.csv');controls=[]
for name,reserve in [('fixed_10pct',.10),('fixed_15pct',.15),('fixed_50pct',.50)]:
 soc=.5;cost=[]
 for row,record in zip(agent.to_dict('records'),records):
  forecast=record['forecast'];actual=data[data.date==row['date']].sort_values('slot');pr=forecast['price_forecast_cny_per_kwh']
  result=execute(actions_llm_guided(forecast['pv_forecast_kw'],forecast['load_forecast_kw'],pr,{'soc_min_safe':reserve,'peak_discharge_offset':0},initial_soc=soc),(actual.pv_power_kw/50).clip(0,1000),actual.load_power_kw.clip(0,800),pr,soc=soc)
  soc=result['end_soc'];cost.append(result['cost_yuan']);controls.append({'date':row['date'],'arm':name,**result})
 summary.setdefault('matched_planner_controls',{})[name]={'total_cost_yuan':sum(cost),'comparison':comparison(agent.cost_yuan,cost,ci_bootstrap)}
pd.DataFrame(controls).to_csv(out/'control_day_results.csv',index=False)
summary['latency_seconds']={'mean':sum(r['elapsed_seconds'] for r in records)/len(records),'max':max(r['elapsed_seconds'] for r in records)}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
meta=json.loads((out/'run_meta.json').read_text())
with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=3) as res:tags=json.load(res)
meta['ollama_model']=[m for m in tags['models'] if m['name']==summary['model']]
meta['policy_archive_sha256']=hashlib.sha256((out/'policy_archive.jsonl').read_bytes()).hexdigest()
meta['postprocessor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(out/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
e0=summary['totals_yuan']['E0'];llm=summary['totals_yuan']['E_LLM'];lines=['# 真实本地 LLM：P0 30 天试运行','',f"模型：{summary['model']}，提示词版本 local-pilot-v1，评估窗口 2019-04-01 至 2019-04-30（Asia/Shanghai）。本轮在独立副本运行，未修改原 PR。",'', '## 方法','', '真实调用本机 Ollama 的 OpenAI 兼容 API，温度 0，每日调用一次；只输入既有历史生成预测、已知政策电价、历史摘要和该臂继承 SOC。提示词冻结后未按收益调参。所有 96 槽预测保留，小数压缩为三位、规则时间网格编码为起止时间与分辨率。输入、输出和降级状态在结算前封存。电池参数与费用目标写入提示词；SOC 硬边界仍由原 execute 保证。', '', '接口预检曾因冷启动超时、字段缺失失败；这些预检不计入正式 30 天批次。预检仅用于接口兼容，正式批次使用修订提示词，不能与原提示词混为一轮。', '', '## 结果','', '| 对照 | 对照总电费（元） | LLM 多省金额（元；负数为更贵） | 配对日节省差 95% CI（元/日） |','|---|---:|---:|---|']
for name in ['E0','E1','E2','E3']:
 c=summary['comparisons'][name];ci=c['daily_increment_bootstrap_ci95_yuan'];lines.append(f"| {name} | {summary['totals_yuan'][name]:,.2f} | {c['total_increment_yuan']:,.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
for name,c in summary['matched_planner_controls'].items():
 d=c['comparison'];ci=d['daily_increment_bootstrap_ci95_yuan'];lines.append(f"| 同规划器 {name} | {c['total_cost_yuan']:,.2f} | {d['total_increment_yuan']:,.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
lines += ['',f"E_LLM 总电费 **{llm:,.2f} 元**，相对无储能节省 **{e0-llm:,.2f} 元（{100*(e0-llm)/e0:.2f}%）**。真实调用成功 {summary['successful_calls']}/30，降级 {summary['fallback_calls']}，SOC 违规 {summary['soc_violations']}。平均调用耗时 {summary['latency_seconds']['mean']:.2f} 秒。",'', '模型返回策略分布：','',*['- '+k+': '+str(v)+' 天。' for k,v in summary['policies'].items()], '', '## 解读与边界','', '是否省钱需看配对差与置信区间。E2 与新规划器存在计划功率计算差异，直接比较不能把全部差额归因于 LLM；同规划器固定 10%/15%/50% 对照用于分离这一混杂因素。CI 使用按日配对 bootstrap，2000 次、seed=42，连续 SOC 会带来日间相关性，因此本次 CI 仅作试运行描述，不作为现场收益承诺。', '', '本轮仅完成 P0 真实 LLM 试验，P1–P4 尚未运行真实模型；P0 数据仍为公开光伏代理、仿真负荷、政策价格演示场景。无真实广东现场收益或投资回报结论。温度 0 不保证跨硬件与服务版本严格确定，归档可复核本次策略。', '', '## 可复现','', '```bash','cd backend','python scripts/evaluate_llm_live_p0.py --output-dir ../reports/generated/llm_live_p0_new','```','', '同批次附加固定参数对照脚本为 scripts/summarize_llm_live_p0.py，当前绑定本轮目录。run_meta.json 保存脚本/数据/策略归档 SHA256 和 Ollama 模型 digest。']
(out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
