"""LLM API failures, deterministic offline mode, physical safety and causal archives."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from agents.llm_energy_agent import LLMStrategicAdvisor, clamp_formulator_policy
from scripts.run_official_experiment import execute, actions_llm_guided
from scripts.llm_experiment import tail_metrics, count_manual_interventions


def reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_offline_deterministic_no_network():
    client=Mock(); advisor=LLMStrategicAdvisor(client=client)
    assert advisor.advise('{"pv":[1,2]}','{}')==advisor.advise({'pv':[1,2]}, {})
    assert client.mock_calls==[]
    assert advisor.last_diagnostics['mode']=='mock_offline'


def test_env_configuration_clamping(monkeypatch):
    monkeypatch.setenv('LLM_BASE_URL','https://provider.example/v1')
    monkeypatch.setenv('LLM_MODEL','from-env');monkeypatch.setenv('LLM_API_KEY','test-key')
    client=Mock();client.chat.completions.create.return_value=reply('{"soc_min_safe":2,"peak_discharge_offset":-300}')
    advisor=LLMStrategicAdvisor(mock_offline=False,client=client)
    assert advisor.advise({}, {})=={'soc_min_safe':.95,'peak_discharge_offset':-4}
    assert client.chat.completions.create.call_args.kwargs['model']=='from-env'


@pytest.mark.parametrize('content',['bad json','{}','[]','{"soc_min_safe":NaN,"peak_discharge_offset":0}',
                                     '{"soc_min_safe":true,"peak_discharge_offset":0}',
                                     '{"soc_min_safe":0.2,"peak_discharge_offset":"oops"}'])
def test_bad_response_fallback(content):
    client=Mock();client.chat.completions.create.return_value=reply(content)
    advisor=LLMStrategicAdvisor(mock_offline=False,client=client)
    assert advisor.advise({}, {})=={'soc_min_safe':.1,'peak_discharge_offset':0}
    assert advisor.last_diagnostics['status']=='fallback'


def test_formulator_high_level_policy_is_clamped_and_has_no_actions():
    policy = clamp_formulator_policy({'scenario': 'cold_wave', 'soc_min_safe': 2,
        'peak_discharge_offset': -99, 'reserve_weight': 9,
        'discharge_window_start': -4, 'discharge_window_end': 120,
        'manual_review_required': True})
    assert policy['soc_min_safe'] == .95
    assert policy['reserve_weight'] == 1.5
    assert policy['discharge_window_start'] == 0
    assert policy['discharge_window_end'] == 95
    assert 'actions' not in policy


def test_formulator_mock_event_policy_and_invalid_event_fallback():
    advisor = LLMStrategicAdvisor()
    result = advisor.formulate({}, {}, {'kind': 'cold_wave', 'severity': .8})
    assert result['scenario'] == 'cold_wave'
    assert result['mode'] == 'event_formulator'
    assert advisor.formulate({}, {}, {'kind': 'unknown'})['scenario'] == 'none'
    assert advisor.last_diagnostics['status'] == 'fallback'


def test_tail_metrics_and_manual_intervention_definition():
    result = tail_metrics([1., 2., 3., 4.], .75)
    assert result['var_cost_yuan'] == 3.25
    assert result['cvar_cost_yuan'] == 4.
    assert result['worst_day_cost_yuan'] == 4.
    assert count_manual_interventions([{'manual_review_required': False, 'fallback': False},
                                       {'manual_review_required': True},
                                       {'policy_changed_by_safety_layer': True}]) == 2


def test_network_failure_and_remote_missing_key(monkeypatch):
    monkeypatch.setenv('LLM_BASE_URL','https://provider.example/v1')
    monkeypatch.delenv('LLM_API_KEY',raising=False)
    client=Mock();advisor=LLMStrategicAdvisor(mock_offline=False,client=client)
    assert advisor.advise({}, {})['soc_min_safe']==.1
    assert client.mock_calls==[]
    monkeypatch.setenv('LLM_API_KEY','test-key')
    client.chat.completions.create.side_effect=TimeoutError('secret message')
    assert advisor.advise({}, {})['soc_min_safe']==.1
    assert 'secret' not in json.dumps(advisor.last_diagnostics)


def test_dotenv_process_env_precedence(monkeypatch,tmp_path):
    for name in ['LLM_MODEL','LLM_BASE_URL','LLM_API_KEY']:monkeypatch.delenv(name,raising=False)
    env=tmp_path/'.env';env.write_text('LLM_MODEL=from-file\nLLM_BASE_URL=http://localhost:12345/v1\n')
    monkeypatch.setenv('LLM_MODEL','from-process')
    # load_dotenv mutations are scoped to this test.
    monkeypatch.setenv('LLM_BASE_URL','http://localhost:12345/v1')
    client=Mock();client.chat.completions.create.return_value=reply('{"soc_min_safe":0.1,"peak_discharge_offset":0}')
    advisor=LLMStrategicAdvisor(mock_offline=False,client=client,env_file=env)
    advisor.advise({}, {})
    assert client.chat.completions.create.call_args.kwargs['model']=='from-process'


@pytest.mark.parametrize('reserve',[.1,.4,.95,5,-5])
@pytest.mark.parametrize('offset',[-400,0,400])
def test_extreme_policy_forecast_error_soc_safe(reserve,offset):
    rng=np.random.default_rng(42);pv=rng.uniform(0,1000,96);load=rng.uniform(0,800,96)
    prices=np.r_[np.full(32,.49),np.full(64,1.46)]
    plan=actions_llm_guided(pv,load,prices,{'soc_min_safe':reserve,'peak_discharge_offset':offset})
    assert all(0<=kw<=500 for _,kw in plan)
    soc=.5
    for i in range(96):
        result=execute([plan[i]],[0.],[800.],[prices[i]],soc=soc)
        soc=result['end_soc']
        assert result['soc_violations']==0
        assert .1-1e-12<=soc<=.95+1e-12


def test_reserve_and_offset_no_midnight_wrap():
    prices=np.array([1.46]+[.86]*95);pv=np.zeros(96);load=np.full(96,800.)
    held=actions_llm_guided(pv,load,prices,{'soc_min_safe':.95,'peak_discharge_offset':0})
    assert all(a!='discharge' or kw==0 for a,kw in held)
    shifted=actions_llm_guided(pv,load,prices,{'soc_min_safe':.1,'peak_discharge_offset':-1})
    assert not any(a=='discharge' for a,_ in shifted)
    delayed=actions_llm_guided(pv,load,prices,{'soc_min_safe':.1,'peak_discharge_offset':1})
    assert delayed[0][0]=='idle' and delayed[1][0]=='discharge'


def test_experiment_arm_causal_archive_and_zero_violations(tmp_path,monkeypatch):
    import sys
    from scripts import run_official_experiment as runner
    monkeypatch.setattr(sys,'argv',['run_official_experiment.py','--agent','llm','--mock-offline','--output-dir',str(tmp_path)])
    runner.main()
    summary=json.loads((tmp_path/'p0_replay/summary.json').read_text())
    assert 'E_LLM' in summary['totals_yuan']
    assert summary['E_LLM']['soc_violations']==0
    records=[json.loads(line) for line in (tmp_path/'agent_policy_archive.jsonl').read_text().splitlines()]
    assert {r['stage'] for r in records}=={'P0','P2','P4'}
    assert len(records)==30+90+365
    for r in records:
        f=r['forecast'];issue=pd.Timestamp(f['issued_at'])
        assert issue.tzinfo is not None
        assert all(pd.Timestamp(t)>issue for t in f['target_timestamps'])
        assert 'actual' not in json.dumps(f)
        assert r['diagnostics']['mode']=='mock_offline'
        if r['stage'] in {'P0','P2'}: assert pd.Timestamp(r['historical_summary']['history_end'])<=issue
    for folder in ['p1_demand_charge','p2_spot_price','p3_lp','p4_annual']:
        assert 'E_LLM' in json.loads((tmp_path/folder/'summary.json').read_text())
    for folder,file in [('p0_replay','day_results.csv'),('p2_spot_price','agent_day_results.csv'),('p4_annual','agent_day_results.csv')]:
        frame=pd.read_csv(tmp_path/folder/file)
        assert frame.soc_violations.sum()==0
    annual=pd.read_csv(tmp_path/'p4_annual/agent_day_results.csv')
    assert annual.E_LLM_end_soc.between(.1-1e-9,.95+1e-9).all()


def test_client_configuration_and_local_defaults(monkeypatch,tmp_path):
    import openai
    for name in ['LLM_MODEL','LLM_BASE_URL','LLM_API_KEY']:monkeypatch.delenv(name,raising=False)
    client=Mock();client.chat.completions.create.return_value=reply('{"soc_min_safe":0.1,"peak_discharge_offset":0}')
    constructor=Mock(return_value=client);monkeypatch.setattr(openai,'OpenAI',constructor)
    advisor=LLMStrategicAdvisor(mock_offline=False,env_file=tmp_path/'absent')
    assert advisor.advise({}, {})['soc_min_safe']==.1
    assert constructor.call_args.kwargs['base_url']=='http://localhost:11434/v1'
    assert constructor.call_args.kwargs['api_key']=='ollama-local-placeholder'
    assert client.chat.completions.create.call_args.kwargs['model']=='qwen2.5:7b'


def test_dotenv_loads_file_values(monkeypatch,tmp_path):
    for name in ['LLM_MODEL','LLM_BASE_URL','LLM_API_KEY']:
        monkeypatch.delenv(name,raising=False)
    env=tmp_path/'.env'
    env.write_text('LLM_MODEL=file-model\nLLM_BASE_URL=http://localhost:12345/v1\nLLM_API_KEY=file-placeholder\n')
    client=Mock();client.chat.completions.create.return_value=reply('{"soc_min_safe":0.1,"peak_discharge_offset":0}')
    advisor=LLMStrategicAdvisor(mock_offline=False,client=client,env_file=env)
    advisor.advise({}, {})
    assert client.chat.completions.create.call_args.kwargs['model']=='file-model'
    # monkeypatch.delenv above restores the pre-test environment even though dotenv wrote it.


def test_executor_body_is_unchanged():
    import hashlib,inspect
    from scripts import run_official_experiment as runner
    # Frozen source digest of execute at 6486d75. AST dumps vary across Python
    # versions; source hashing keeps this guard portable and requires no .git.
    current=inspect.getsource(runner.execute).strip()
    digest=hashlib.sha256(current.encode()).hexdigest()
    assert digest=='f8f3afa1bb7c1bd1892677718e61029abede572b992e36a0de1da0e255d29cca'
