import json
import pytest
from agents.llm_energy_agent import compact_forecast,parse_response

@pytest.mark.parametrize('content',['{"soc_min_safe":0.2,"peak_discharge_offset":1}', '```json\n{"soc_min_safe":0.2,"peak_discharge_offset":1}\n```'])
def test_policy_response_plain_and_fenced(content):
 assert parse_response(content)=={'soc_min_safe':.2,'peak_discharge_offset':1}

def test_response_explanation_remains_invalid():
 with pytest.raises(json.JSONDecodeError):parse_response('Here is a policy: {"soc_min_safe":0.2}')

def test_compact_forecast_preserves_slots_and_original():
 original={'target_timestamps':['2020-01-01T00:00:00+08:00','2020-01-01T00:15:00+08:00'],'pv_forecast_kw':[1.234567,3.33333],'resolution_minutes':15}
 result=compact_forecast(original)
 assert result['n_slots']==2 and result['target_start']==original['target_timestamps'][0]
 assert result['target_end']==original['target_timestamps'][-1]
 assert result['pv_forecast_kw']==[1.235,3.333]
 assert original['pv_forecast_kw']==[1.234567,3.33333]
 assert 'target_timestamps' in original
