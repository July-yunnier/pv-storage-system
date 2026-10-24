"""Post-hoc settlement and reserve diagnostics; no API calls or policy changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scripts.run_official_experiment import (
    CAP, PCS, SOC_MIN, SOC_MAX, DT, ETA, PRICES,
    execute, fixed_actions, actions_rule, lp_actions, ci_bootstrap,
)
from scripts.llm_experiment import actions_llm_guided


def trace_execute(actions, pv, load, prices, timestamps, initial_soc):
    """Obtain diagnostics by calling the unchanged executor for each interval."""
    soc = initial_soc
    rows = []
    for action, p, l, price, ts in zip(actions, pv, load, prices, timestamps):
        result = execute([action], [p], [l], [price], soc=soc)
        change = (result['end_soc'] - soc) * CAP
        charge = max(change, 0.) / ETA / DT
        discharge = max(-change, 0.) * ETA / DT
        rows.append({
            'ts': ts, 'initial_soc': soc, 'soc': result['end_soc'],
            'action': action[0], 'planned_kw': action[1],
            'charge_kw': charge, 'discharge_kw': discharge,
            'grid_kw': result['peak_import_kw'], 'price_cny_per_kwh': price,
            'pv_kw': p, 'load_kw': l, 'cost_yuan': result['cost_yuan'],
            'soc_violations': result['soc_violations'],
        })
        soc = result['end_soc']
    trace = pd.DataFrame(rows)
    batch = execute(actions, pv, load, prices, soc=initial_soc)
    for got, expected in [
        (trace.cost_yuan.sum(), batch['cost_yuan']),
        (trace.grid_kw.max(), batch['peak_import_kw']),
        ((trace.discharge_kw * DT).sum(), batch['discharge_kwh']),
        (soc, batch['end_soc']),
    ]:
        if not np.isclose(got, expected, rtol=1e-9, atol=1e-7):
            raise AssertionError('interval diagnostics differ from original executor')
    return trace, batch


def paired_month_ci(agent, base, rate, *, block=1, n=2000, seed=42):
    """Resample paired days, recomputing monthly maximum (never amortize demand)."""
    rng = np.random.default_rng(seed)
    size = len(agent)
    starts = rng.integers(0, size, (n, int(np.ceil(size / block))))
    index = ((starts[:, :, None] + np.arange(block)) % size).reshape(n, -1)[:, :size]
    def bills(frame):
        return frame.cost_yuan.to_numpy()[index].sum(axis=1) + rate * frame.peak_import_kw.to_numpy()[index].max(axis=1)
    return np.quantile(bills(base) - bills(agent), [.025, .975]).tolist()


def block_daily_ci(delta, block=7):
    rng = np.random.default_rng(42)
    delta = np.asarray(delta)
    starts = rng.integers(0, len(delta), (2000, int(np.ceil(len(delta) / block))))
    ix = ((starts[:, :, None] + np.arange(block)) % len(delta)).reshape(2000, -1)[:, :len(delta)]
    return np.quantile(delta[ix].mean(axis=1), [.025, .975]).tolist()


def summarize_operation(trace):
    discharge = float(trace.discharge_kw.sum() * DT)
    return {
        'min_soc': float(min(trace.soc.min(), trace.initial_soc.min())),
        'hours_soc_at_or_below_15pct': float((trace.soc <= .15 + 1e-9).sum() * DT),
        'hours_soc_at_or_below_10pct': float((trace.soc <= .10 + 1e-9).sum() * DT),
        'soc_violations': int(trace.soc_violations.sum()),
        'power_violations': int(((trace.charge_kw > PCS + 1e-6) | (trace.discharge_kw > PCS + 1e-6)).sum()),
        'discharge_kwh': discharge,
        'equivalent_full_cycles_discharge_over_usable_capacity': discharge / (CAP * (SOC_MAX - SOC_MIN)),
        'pv_self_use_ratio': float(np.minimum(trace.pv_kw, trace.load_kw + trace.charge_kw).clip(lower=0).sum() / trace.pv_kw.clip(lower=0).sum()),
        'max_import_kw': float(trace.grid_kw.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'reports/generated/llm_live_p0_20261024')
    args = ap.parse_args()
    source = args.run_dir.resolve()
    out = source / 'p1_p3_diagnosis'
    out.mkdir(exist_ok=False)
    archive = source / 'policy_archive.jsonl'
    original_meta = json.loads((source / 'run_meta.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == original_meta['policy_archive_sha256']
    for name, digest in original_meta['implementation_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    records = [json.loads(line) for line in archive.read_text().splitlines()]
    sealed = pd.read_csv(source / 'day_results.csv').set_index('date')
    controls = pd.read_csv(source / 'control_day_results.csv')
    base = ROOT / 'reports/official_experiment/llm_mock'
    base_days = pd.read_csv(base / 'p0_replay/day_results.csv')
    lp_days = pd.read_csv(base / 'p3_lp/day_results.csv').set_index('date')
    rate = json.loads((base / 'p1_demand_charge/summary.json').read_text())['demand_rate_yuan_per_kw_month']
    data_path = ROOT / 'reports/official_experiment/data/full_scene.csv'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == original_meta['data_sha256']
    data = pd.read_csv(data_path)
    arms = ['E0', 'E1', 'E2', 'E3', 'LP', 'E_LLM', 'fixed_10pct', 'fixed_15pct', 'fixed_50pct']
    socs = dict.fromkeys(arms, .5)
    rows, traces, counterfactuals, features = [], [], [], []
    for record in records:
        forecast, policy = record['forecast'], record['policy']
        times = forecast['target_timestamps']
        day = pd.Timestamp(times[0]).date().isoformat()
        assert pd.Timestamp(forecast['issued_at']) < pd.Timestamp(times[0])
        assert pd.Timestamp(record['historical_summary']['history_end']) <= pd.Timestamp(forecast['issued_at'])
        actual = data[data.date == day].sort_values('slot')
        pv = (actual.pv_power_kw / 50).clip(0, 1000).to_numpy()
        load = actual.load_power_kw.clip(0, 800).to_numpy()
        fp = np.asarray(forecast['pv_forecast_kw'])
        fl = np.asarray(forecast['load_forecast_kw'])
        prices = np.asarray(forecast['price_forecast_cny_per_kwh'])
        assert len(actual) == len(times) == 96
        assert np.isclose(socs['E_LLM'], forecast['initial_soc'])
        daily = {}
        for arm in arms:
            start = socs[arm]
            if arm == 'E0': plan = [('idle', 0.)] * 96
            elif arm == 'E1': plan = fixed_actions(prices)
            elif arm == 'E2': plan = actions_rule(fp, fl, prices, fp, fl)
            elif arm in {'LP', 'E3'}: plan = lp_actions(fp, fl, prices, pv if arm == 'E3' else fp, load if arm == 'E3' else fl, start)
            else:
                selected = policy if arm == 'E_LLM' else {'soc_min_safe': float(arm.split('_')[1].removesuffix('pct')) / 100, 'peak_discharge_offset': 0}
                plan = actions_llm_guided(fp, fl, prices, selected, initial_soc=start)
            trace, result = trace_execute(plan, pv, load, prices, times, start)
            socs[arm] = result['end_soc']
            trace['date'] = day; trace['arm'] = arm
            traces.append(trace)
            daily[arm] = (trace, result)
            rows.append({'date': day, 'arm': arm, 'initial_soc': start, **result, **summarize_operation(trace)})
            if arm == 'E_LLM': expected = sealed.loc[day].cost_yuan
            elif arm == 'LP': expected = lp_days.loc[day].LP_cost_yuan
            elif arm.startswith('fixed_'): expected = controls[(controls.date == day) & (controls.arm == arm)].cost_yuan.iloc[0]
            else: expected = base_days[(base_days.date == day) & (base_days.arm == arm)].cost_yuan.iloc[0]
            assert np.isclose(result['cost_yuan'], expected, atol=1e-6)
        # One-day interventions hold the actual LLM start SOC, forecasts and offset fixed.
        # They are diagnostic counterfactuals, not a new continuous or adaptive policy.
        if policy['soc_min_safe'] == .5:
            for reserve in [.1, .15, .5]:
                plan = actions_llm_guided(fp, fl, prices, {**policy, 'soc_min_safe': reserve}, initial_soc=forecast['initial_soc'])
                trace, result = trace_execute(plan, pv, load, prices, times, forecast['initial_soc'])
                expected = execute(plan, fp, fl, prices, soc=forecast['initial_soc'])
                counterfactuals.append({'date': day, 'reserve': reserve, **result, **summarize_operation(trace),
                    'forecast_cost_yuan': expected['cost_yuan'], 'forecast_end_soc': expected['end_soc']})
        features.append({
            'date': day, 'reserve': policy['soc_min_safe'], 'initial_soc': forecast['initial_soc'],
            'forecast_pv_kwh': float(fp.sum() * DT), 'forecast_load_kwh': float(fl.sum() * DT),
            'forecast_net_energy_kwh': float((fl - fp).sum() * DT),
            'forecast_peak_net_kw': float((fl - fp)[prices >= PRICES['peak'] - 1e-6].max()),
            'actual_net_minus_forecast_net_mae_kw': float(np.abs((load - pv) - (fl - fp)).mean()),
            'history_days': record['historical_summary']['history_days'],
        })
    days = pd.DataFrame(rows)
    trace = pd.concat(traces, ignore_index=True)
    cf = pd.DataFrame(counterfactuals)
    features = pd.DataFrame(features)
    days.to_csv(out / 'daily_diagnostics.csv', index=False)
    trace.to_csv(out / 'interval_diagnostics.csv.gz', index=False)
    cf.to_csv(out / 'high_reserve_counterfactuals.csv', index=False)
    features.to_csv(out / 'policy_input_features.csv', index=False)
    byarm = {arm: days[days.arm == arm].set_index('date').sort_index() for arm in arms}
    bills = {}
    for arm, frame in byarm.items():
        peak = float(frame.peak_import_kw.max())
        bills[arm] = {'energy_cost_yuan': float(frame.cost_yuan.sum()), 'max_import_kw': peak,
                      'demand_charge_yuan': peak * rate, 'total_bill_yuan': float(frame.cost_yuan.sum() + peak * rate)}
    p1 = {'window': [days.date.min(), days.date.max()], 'n_days': 30,
          'demand_rate_cny_per_kw_month': rate, 'arms': bills,
          'scope': 'same P0 policies settled with assumed monthly demand rate; no demand-aware optimization'}
    p1['comparisons'] = {}
    llm = byarm['E_LLM']
    for arm in arms:
        if arm == 'E_LLM': continue
        p1['comparisons'][arm] = {
            'energy_increment_yuan': bills[arm]['energy_cost_yuan'] - bills['E_LLM']['energy_cost_yuan'],
            'demand_increment_yuan': bills[arm]['demand_charge_yuan'] - bills['E_LLM']['demand_charge_yuan'],
            'total_increment_yuan': bills[arm]['total_bill_yuan'] - bills['E_LLM']['total_bill_yuan'],
            'monthly_total_increment_bootstrap_ci95_yuan': paired_month_ci(llm, byarm[arm], rate),
            'monthly_total_increment_circular_block7_ci95_yuan': paired_month_ci(llm, byarm[arm], rate, block=7),
        }
    p3 = {'window': p1['window'], 'n_days': 30, 'scope': 'same historical forecasts, physical limits and continuous SOC; LP energy-only objective', 'comparisons': {}}
    for arm in ['E0', 'E1', 'E2', 'E3', 'LP', 'fixed_10pct', 'fixed_15pct']:
        delta = byarm[arm].cost_yuan.to_numpy() - llm.cost_yuan.to_numpy()
        p3['comparisons'][arm] = {'total_increment_yuan': float(delta.sum()), 'daily_mean_increment_yuan': float(delta.mean()),
            'daily_increment_bootstrap_ci95_yuan': ci_bootstrap(delta),
            'daily_increment_circular_block7_ci95_yuan': block_daily_ci(delta)}
    operations = {arm: summarize_operation(trace[trace.arm == arm]) for arm in arms}
    high = cf[cf.reserve == .5].set_index('date')
    low = cf[cf.reserve == .1].set_index('date').loc[high.index]
    delta = high.cost_yuan - low.cost_yuan
    reserve_diagnosis = {
        'n_high_reserve_days': len(high),
        'one_day_same_start_50_vs_10': {
            'extra_energy_cost_yuan_sum': float(delta.sum()),
            'extra_energy_cost_daily_mean_yuan': float(delta.mean()),
            'extra_energy_cost_daily_bootstrap_ci95_yuan': ci_bootstrap(delta),
            'days_50pct_more_expensive': int((delta > 1e-6).sum()),
            'days_lower_peak_import': int((high.peak_import_kw < low.peak_import_kw - 1e-6).sum()),
            'mean_peak_import_reduction_kw': float((low.peak_import_kw - high.peak_import_kw).mean()),
            'mean_min_soc_increase': float((high.min_soc - low.min_soc).mean()),
            'hours_at_or_below_15pct_10': float(low.hours_soc_at_or_below_15pct.sum()),
            'hours_at_or_below_15pct_50': float(high.hours_soc_at_or_below_15pct.sum()),
            'soc_violations_10': int(low.soc_violations.sum()), 'soc_violations_50': int(high.soc_violations.sum()),
            'discharge_reduction_kwh': float((low.discharge_kwh - high.discharge_kwh).sum()),
            'extra_forecast_cost_yuan_sum': float((high.forecast_cost_yuan - low.forecast_cost_yuan).sum()),
            'days_forecast_cost_50pct_higher': int((high.forecast_cost_yuan > low.forecast_cost_yuan + 1e-6).sum()),
            'scope': '16 post-hoc one-day interventions; each uses actual LLM start SOC, not a continuous counterfactual month',
        },
        'policy_feature_group_means': features.groupby('reserve').mean(numeric_only=True).reset_index().to_dict('records'),
        'motivation_boundary': 'no contemporaneous rationale or uncertainty/outage/wear objective archived; model reasoning cannot be identified from numeric policies',
    }
    diagnosis = {'p1': p1, 'p3': p3, 'operations': operations, 'reserve_diagnosis': reserve_diagnosis,
                 'reconciliation': {'sealed_costs_match': True, 'interval_and_batch_execution_match': True, 'new_llm_calls': 0},
                 'ci_scope': 'descriptive post-hoc: 30 dependent days, one synthetic/demo month; block7 sensitivity is exploratory'}
    diagnosis['break_even'] = {}
    for arm in ['fixed_10pct', 'fixed_15pct']:
        avoided_discharge = operations[arm]['discharge_kwh'] - operations['E_LLM']['discharge_kwh']
        peak_reduction = bills[arm]['max_import_kw'] - bills['E_LLM']['max_import_kw']
        extra_energy = bills['E_LLM']['energy_cost_yuan'] - bills[arm]['energy_cost_yuan']
        extra_bill = bills['E_LLM']['total_bill_yuan'] - bills[arm]['total_bill_yuan']
        diagnosis['break_even'][arm] = {
            'avoided_discharge_kwh': avoided_discharge,
            'additional_wear_value_cny_per_kwh_discharged_needed': extra_bill / avoided_discharge,
            'demand_rate_cny_per_kw_month_needed': extra_energy / peak_reduction,
            'same_terminal_soc': bool(np.isclose(llm.end_soc.iloc[-1], byarm[arm].end_soc.iloc[-1])),
        }
    for name, value in [('p1_summary.json', p1), ('p3_summary.json', p3), ('diagnosis.json', diagnosis)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    paths = [Path(__file__), archive, source / 'day_results.csv', source / 'control_day_results.csv', source / 'run_meta.json', data_path,
             ROOT / 'backend/scripts/run_official_experiment.py', ROOT / 'backend/scripts/llm_experiment.py',
             base / 'p0_replay/day_results.csv', base / 'p3_lp/day_results.csv', base / 'p1_demand_charge/summary.json']
    metadata = {'generated_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                'commit_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'evaluation_timezone': 'Asia/Shanghai',
                'input_and_script_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'bootstrap': {'n': 2000, 'seed': 42, 'block_sensitivity_days': 7},
                'new_llm_calls': 0, 'policy_tuning': False, 'units': 'CNY/kWh; demand CNY/kW/month; SOC fraction',
                'scope': 'post-hoc diagnostics only; no update to sealed model/prompt/results or original PR'}
    (out / 'run_meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    render_report(diagnosis, out)
    print(json.dumps({'p1': p1, 'p3': p3, 'reserve_diagnosis': reserve_diagnosis, 'operations': operations}, ensure_ascii=False, indent=2))


def render_report(d, out):
    p1, p3, ops = d['p1'], d['p3'], d['operations']
    cf = d['reserve_diagnosis']['one_day_same_start_50_vs_10']
    def money(x): return f'{x:,.2f}'
    def interval(x): return f'[{x[0]:,.2f}, {x[1]:,.2f}]'
    lines = [
        '# 真实 LLM 的 P1/P3 补充比较与 50% SOC 储备诊断', '',
        '## 1. 摘要', '',
        '**方法：**复用 qwen2.5:7b / local-pilot-v1 在 2019-04-01—2019-04-30（Asia/Shanghai）的封存策略，'
        '对同一演示场景做需量结算、预测 LP 比较和固定储备反事实；新增 LLM 调用 0 次，未改提示词、策略、预测或执行器。', '',
        '**结果：**含需量后，E_LLM 比 E2 多花 9,403.02 元、比同规划器固定 10% 多花 4,554.77 元；'
        '只计电量费时比预测 LP 多花 9,912.34 元。50% 储备在 16 个日期都增加了同起点单日费用，'
        '减少低 SOC 暴露与放电量，但未减少安全违规次数（两者均为 0）。', '',
        '**解读：**当前证据不支持“50% 储备经济上合理”或“LLM 已降低事故风险”。'
        '模型决策动机未记录，只能诊断输出与执行后果。', '',
        '## 2. 数据、对账与统计口径', '',
        '**方法：**站点 S-DONGGUAN-DEMO；公开光伏代理、仿真工厂负荷、广东政策价格模板，'
        '500 kW / 1000 kWh、往返效率 0.88、SOC 10%—95%。各臂分别跨日继承 SOC。'
        'LLM/固定储备共用 actions_llm_guided；E2 用原规则，LP 用相同历史预测，E3 用事后实测。'
        '逐槽诊断通过原 execute 逐次执行获得，并与批量执行和既有封存成本逐日对账。', '',
        '**结果：**9 臂 × 30 天 × 96 槽；原运行归档/实现哈希匹配，逐日费用全部匹配；'
        '30 次原 LLM 调用成功、原批次降级 0。本次只离线复算。', '',
        '**解读：**E2 与顾问规划器实现不同，比较两者不能隔离模型净贡献；固定 10%/15% 是更直接的同规划器对照。'
        '本次为看过结果后的诊断，非新样本外验收。', '',
        '统计：配对按日 bootstrap 2000 次、seed=42；另给 7 天循环块重采样作为相关性敏感性分析。'
        'P1 每个重采样月份重新计算 max(购电功率)，不把需量费摊到每日再求和。'
        '只有一个演示月份，最大值重采样不能覆盖未观察到的尖峰；CI 均为描述性。', '',
        '## 3. P1：加需量费用后的比较', '',
        '**方法：**沿用既有假设 35 元/kW·月，账单 = 电量费 + 月最大 15 分钟购电功率 × 单价；'
        '没有修改策略为需量优化，没有合同容量、计费最小值等完整现场规则。', '',
        '| 臂 | 电量费（元） | 月最大购电功率（kW） | 需量费（元） | 总账单（元） |',
        '|---|---:|---:|---:|---:|',
    ]
    for arm, b in p1['arms'].items():
        lines.append(f"| {arm} | {money(b['energy_cost_yuan'])} | {b['max_import_kw']:.3f} | {money(b['demand_charge_yuan'])} | {money(b['total_bill_yuan'])} |")
    lines += ['', '| 对照 | LLM 多省电量费（元） | 多省需量费（元） | 多省总额（元） | 月总差配对 95% CI（元） | 7天块 CI（元） |',
              '|---|---:|---:|---:|---|---|']
    for arm in ['E0', 'E1', 'E2', 'LP', 'fixed_10pct', 'fixed_15pct']:
        c = p1['comparisons'][arm]
        lines.append(f"| {arm} | {money(c['energy_increment_yuan'])} | {money(c['demand_increment_yuan'])} | {money(c['total_increment_yuan'])} | {interval(c['monthly_total_increment_bootstrap_ci95_yuan'])} | {interval(c['monthly_total_increment_circular_block7_ci95_yuan'])} |")
    lines += ['', '正数表示 E_LLM 更省。', '',
        '**解读：**相对 E2，4.4 kW 的月峰值下降仅省 154 元，无法抵消 9,557.02 元额外电量费。'
        'E_LLM 仍高于无储能的 800 kW 购电峰值，不能称已经实现现场削峰。', '',
        '对预测 LP，E_LLM 含需量账单低 2,384.42 元，但按日重采样 CI 包含 0；'
        '7 天块 CI 不含 0，说明判断对方法敏感，不宣称稳健显著更省。'
        '原因是既有 LP 只优化电量费，充电形成 1261.984 kW 尖峰；应诊断 LP 目标函数缺少需量项，不能外推 LLM 优于含需量 LP。'
        '同理 E3 只在电量目标下作为事后参照，含需量后不是总账单最优上界。', '',
        '峰值轨迹：E_LLM 的 910.648 kW 出现在 2019-04-25T23:15:00+08:00（410.648 kW 负荷 + 500 kW 充电）；'
        'E2/固定10% 的 915.048 kW 在 2019-04-11T23:45:00+08:00（415.048 + 500）；'
        'LP 的 1261.984 kW 在 2019-04-25T06:45:00+08:00。下降来自充电时序和电量变化，不能归因为高峰主动削减。', '',
        '## 4. P3：与预测 LP 的电量费比较', '',
        '**方法：**同一 30 天窗口、相同预测数组、价格、容量和效率，SOC 各臂连续继承；'
        '本节只计电量费，无终端 SOC 等值约束或残值结算。', '',
        '| 对照 | LLM 多省总额（元） | 日均多省（元/日） | 日配对 95% CI（元/日） | 7天块 CI（元/日） |',
        '|---|---:|---:|---|---|']
    for arm in ['E2', 'LP', 'E3', 'fixed_10pct', 'fixed_15pct']:
        c = p3['comparisons'][arm]
        lines.append(f"| {arm} | {money(c['total_increment_yuan'])} | {money(c['daily_mean_increment_yuan'])} | {interval(c['daily_increment_bootstrap_ci95_yuan'])} | {interval(c['daily_increment_circular_block7_ci95_yuan'])} |")
    lines += ['', '**结果与解读：**预测 LP 电量费 196,566.94 元，E_LLM 206,479.28 元；'
        '本窗口 LLM 落后，两个描述性 CI 均为负。终端 SOC：LLM 86.17%、LP 17.29%、E2 56.90%；'
        '储备未结算残值会影响跨臂比较，因此不把全部差额解释为模型损失。'
        'LLM 与固定10% 的期末 SOC 同为 86.17%，其 4,708.77 元电量费差没有终端电量不等这一混杂。', '',
        '## 5. 50% SOC：同起点反事实与运行风险', '',
        '**方法：**只对原模型选择 50% 的 16 天，将储备改为 10%/15%/50%，固定当天原 LLM 期初 SOC、'
        '预测、偏移和实测，调用原执行器；不把单日末 SOC 接到另一反事实日。'
        '这能隔离当天储备参数的作用，不能当成新的连续月策略。', '',
        f"**结果：**50% 相对 10%：16/16 天电量费更高，合计增加 {money(cf['extra_energy_cost_yuan_sum'])} 元，"
        f"日均 {money(cf['extra_energy_cost_daily_mean_yuan'])} 元，描述性 95% CI {interval(cf['extra_energy_cost_daily_bootstrap_ci95_yuan'])} 元/日；"
        f"少放电 {money(cf['discharge_reduction_kwh'])} kWh。9/16 天的日峰值降低，平均下降 {cf['mean_peak_import_reduction_kw']:.2f} kW。", '',
        f"预测本身下（不使用当天实测）50% 储备也在 {cf['days_forecast_cost_50pct_higher']}/16 天增加电量费，合计 {money(cf['extra_forecast_cost_yuan_sum'])} 元。"
        '在没有额外备用价值和终端残值目标的前提下，不能用“预测误差造成的偶然损失”充分解释该选择。', '',
        '| 连续30天臂 | SOC最低值 | SOC≤15%时长（h） | SOC违规/功率越限 | 放电量（kWh） | 等效循环数* | PV自用率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['E_LLM', 'fixed_10pct', 'fixed_15pct', 'fixed_50pct', 'LP']:
        o = ops[arm]
        lines.append(f"| {arm} | {100*o['min_soc']:.2f}% | {o['hours_soc_at_or_below_15pct']:.2f} | {o['soc_violations']}/{o['power_violations']} | {money(o['discharge_kwh'])} | {o['equivalent_full_cycles_discharge_over_usable_capacity']:.2f} | {100*o['pv_self_use_ratio']:.2f}% |")
    lines += ['', '*协议口径：放电总量 / 可用容量850 kWh。SOC时长按各15分钟末状态计数，为暴露代理指标。', '',
        '**风险解读：**LLM 相对固定10% 将 SOC≤15% 时长从 52.5 h 降到21 h，减少放电 5,217.92 kWh；'
        '两臂最低 SOC 约10.8%/10.7%，硬约束违规均0，不能称违规风险已经降低。'
        '固定15%在本窗口低 SOC 时长0、最低15.72%、账单更低，说明目前并无证据证明50%是必要储备。'
        '少放电可能减少循环老化，但没有电芯寿命模型；高SOC停留还可能影响日历老化，不能只凭循环数宣称延寿。'
        '没有停电、负荷切除、备电服务、故障率或极端误差样本，无法计算事故概率、供电可靠性或备用收益。', '',
        '## 6. 保守储备是否值得：盈亏平衡', '',
        '**方法：**在当前月实际差额上，计算需量单价或额外循环损耗价值达到多少才可抵消成本；'
        '这是假设敏感性计算，不是电池寿命估计或建议电价。', '',
        '| 同规划器对照 | LLM多花含需量账单（元） | 需量单价盈亏平衡（元/kW·月） | 所需额外避免损耗价值（元/kWh放电） | 期末SOC相同 |',
        '|---|---:|---:|---:|---|']
    for arm, b in d['break_even'].items():
        c = p1['comparisons'][arm]
        lines.append(f"| {arm} | {money(-c['total_increment_yuan'])} | {money(b['demand_rate_cny_per_kw_month_needed'])} | {b['additional_wear_value_cny_per_kwh_discharged_needed']:.3f} | {'是' if b['same_terminal_soc'] else '否，需残值校正'} |")
    lines += ['', '**解读：**相对固定10%，当前需量单价35元/kW·月远低于约1070.18元的盈亏平衡值；'
        '若只依赖少放电补偿成本，需要额外避免损耗价值约0.873元/kWh放电。'
        '没有寿命、备电损失或用户风险偏好证据，不能认定已覆盖该成本。', '',
        '## 7. 为什么模型经常返回50%：证据与未知', '',
        '**可确认：**16天返回50%、14天10%，全部放电偏移0，均为成功调用而非降级；'
        '提示词只要求最小化购电成本，未给需量单价、风险预算、停电概率、损耗价格或预测区间。'
        '归档只有策略字段，没有同时记录的理由。', '',
        '**输入对比（探索性）：**50%组平均预测净电量8550.24 kWh，10%组8521.69 kWh；'
        '预测峰段净负荷686.81与686.13 kW，均接近。50%组实际净负荷预测MAE为105.30 kW，'
        '反而低于10%组134.55 kW；这个误差是事后结果，模型发报时不可见，不能当作其决策理由。'
        '50%组平均期初SOC92.19%、10%组77.98%，且日期分布不同；期初SOC受前日策略影响，存在内生性。', '',
        '**不能确认：**“模型担心云遮”“学会了预测误差风险”“主动优化需量”“50%是最优安全值”等解释都缺乏证据。'
        '通用保守偏好、数字锚定或成本推理能力不足均只是待检验假设，不能从策略输出还原内部推理。', '',
        '## 8. 结论与后续诊断', '',
        '1. 已证实于本演示窗口：需量费小幅改善不足以抵消电量损失；预测LP电量费用更低；全部臂安全违规0。',
        '2. 仅可作代理解释：高储备减少低SOC时长与循环吞吐量，不等于降低实际事故/停电风险。',
        '3. 尚未覆盖：含需量LP、磨损/备电价值、极端扰动、跨季节和现场收益；不能作投资或广东真实省钱承诺。',
        '4. 下一轮先固定成本目标、终端价值和可量化风险，再冻结提示词与验收标准，在未用于本轮诊断的新窗口验证。'
        '可以增加简短结构化“依据/约束”审计字段，但事后追问模型的理由不能替代本轮缺失的证据。'
        '对SOC/备用约束做预先约定的误差与停电压力测试，与固定15%和含需量LP对照；本轮不调参重报。', '',
        '## 9. 复现与文件', '',
        '```bash', 'cd backend',
        'python scripts/analyze_llm_live_p1_p3.py --run-dir ../reports/generated/llm_live_p0_20261024',
        '```', '', '目标子目录必须尚不存在；重复分析请先复制原封存运行目录到新位置。'
        '输出 p1_summary.json / p3_summary.json / diagnosis.json、逐槽轨迹、同起点反事实、输入特征和 run_meta.json。'
        '金额正方向统一为“对照账单−LLM账单”，正值更省；输入和脚本哈希可核对，所有完整时间戳带时区。', '',
        '![P1/P3与储备诊断](diagnostic.png)',
    ]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    selected = ['E2', 'LP', 'E_LLM', 'fixed_10pct', 'fixed_15pct']
    x = np.arange(len(selected))
    energy = np.array([p1['arms'][a]['energy_cost_yuan'] for a in selected]) / 1000
    demand = np.array([p1['arms'][a]['demand_charge_yuan'] for a in selected]) / 1000
    axes[0].bar(x, energy, label='Energy', color='#3576a8')
    axes[0].bar(x, demand, bottom=energy, label='Demand', color='#e7a647')
    axes[0].set_xticks(x, [a.replace('fixed_', 'Fixed ') for a in selected], rotation=35, ha='right')
    axes[0].set(title='P1: monthly bill', ylabel='Thousand CNY'); axes[0].legend(fontsize=8)
    axes[1].bar(x, [ops[a]['hours_soc_at_or_below_15pct'] for a in selected], color='#5d9d83')
    axes[1].set_xticks(x, [a.replace('fixed_', 'Fixed ') for a in selected], rotation=35, ha='right')
    axes[1].set(title='Low-SOC exposure (proxy)', ylabel='Hours at SOC <= 15%')
    cfs = pd.read_csv(out / 'high_reserve_counterfactuals.csv')
    hi = cfs[cfs.reserve == .5].set_index('date'); lo = cfs[cfs.reserve == .1].set_index('date').loc[hi.index]
    axes[2].scatter(lo.peak_import_kw - hi.peak_import_kw, hi.cost_yuan - lo.cost_yuan, color='#bb534e')
    axes[2].axvline(0, color='gray', lw=.8); axes[2].axhline(0, color='gray', lw=.8)
    axes[2].set(title='16 matched one-day interventions', xlabel='Peak reduction: 50% vs 10% (kW)', ylabel='Extra energy cost at 50% (CNY)')
    fig.suptitle('Real Qwen 7B P0 policies: post-hoc diagnostics | Demo data, April 2019')
    fig.savefig(out / 'diagnostic.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
