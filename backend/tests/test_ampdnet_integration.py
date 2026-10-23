"""Actual delivered graph/API tests; no retraining and no upstream import."""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import torch

from app.domain.forecast import pv_ampdnet as ampdnet
from app.domain.models import AmpdnetInput, ForecastRequest
from tests.conftest import SITE_PAYLOAD

ROOT=Path(__file__).resolve().parents[2]
RELEASE=Path(__file__).resolve().parents[1] / "data/model_releases/ampdnet_delivery20"


def request_payload(station='Solar1',task='15min',site_id='S-AMPD'):
    length,steps=ampdnet.TASKS[task]
    now=datetime(2020,10,1,10,0,tzinfo=ZoneInfo('Asia/Shanghai'))
    times=[now]
    while len(times)<length:
        prev=times[-1]-timedelta(minutes=15)
        if prev.hour<6:prev=prev.replace(hour=18,minute=45)-timedelta(days=1)
        times.append(prev)
    observations=[dict(ts=t.isoformat(),tsi=400.,dni=100.,ghi=150.,temperature_c=20.) for t in reversed(times)]
    return dict(site_id=site_id,targets=['pv_power'],horizon_hours=steps*.25,resolution_minutes=15,issued_at=now.isoformat(),options=dict(pv_model='ampdnet',intraday=True,ampdnet_input=dict(station=station,task=task,observations=observations)))


def site(client,cap=50000):
    payload=json.loads(json.dumps(SITE_PAYLOAD));payload['site_id']='S-AMPD';payload['pv']['capacity_kwp']=cap
    assert client.post('/api/v1/sites',json=payload).status_code==201


@pytest.mark.parametrize('station,cap',[('Solar1',50000),('Solar2',130000),('Solar6',35000),('Solar7',30000),('Solar8',30000)])
@pytest.mark.parametrize('task',['15min','1h','4h'])
def test_delivered_graph_inference_and_provenance(client,station,cap,task):
    site(client,cap);p=request_payload(station,task)
    response=client.post('/api/v1/forecast',json=p)
    assert response.status_code==200,response.text
    out=response.json()['results'][0]
    length,steps=ampdnet.TASKS[task]
    assert len(out['points'])==steps
    assert out['model']['version']=='delivery20-full-s42'
    assert out['accuracy'] is None and out['p10'] is None and out['p90'] is None
    assert '20 epochs completed' in out['provenance']['training_status']
    ts=datetime.fromisoformat(p['issued_at'])
    for i,row in enumerate(out['points']):
        assert datetime.fromisoformat(row['ts'])-ts==timedelta(minutes=15*(i+1))
        assert 0<=row['value']<=cap
    meta=json.loads((RELEASE/f'{task}_{station}_processed_3_full_tr0.6_val0.2_s42.json').read_text())
    graph=torch.jit.load(str(RELEASE/meta['artifact_file']),map_location='cpu').eval()
    raw=np.array([[r[k] for k in ['tsi','dni','ghi','temperature_c']] for r in p['options']['ampdnet_input']['observations']],dtype=np.float32)
    low=np.array(meta['input_min'],dtype=np.float32);high=np.array(meta['input_max'],dtype=np.float32)
    with torch.inference_mode():norm=graph(torch.from_numpy(((raw-low)/(high-low))[None])).numpy()[0]
    expected=(norm*np.float32(meta['output_max_mw']-meta['output_min_mw'])+np.float32(meta['output_min_mw']))*1000
    assert np.allclose(out['components']['raw_prediction_kw'],expected,atol=.02,rtol=0)


@pytest.mark.parametrize('kind',['gap','future','wrong_length','wrong_cap','cross_night','missing_issue','wrong_horizon','legacy_power'])
def test_reject_invalid_or_unsupported_inputs(client,kind):
    site(client,1000 if kind=='wrong_cap' else 50000)
    p=request_payload(task='4h');obs=p['options']['ampdnet_input']['observations']
    if kind=='gap':obs[3]['ts']=obs[2]['ts']
    elif kind=='future':obs[-1]['ts']=(datetime.fromisoformat(p['issued_at'])+timedelta(minutes=15)).isoformat()
    elif kind=='wrong_length':obs.pop()
    elif kind=='cross_night':
        p['issued_at']='2020-10-01T18:45:00+08:00'
        # the issue change itself also invalidates last input; still must not infer.
        obs[-1]['ts']=p['issued_at']
    elif kind=='missing_issue':p.pop('issued_at')
    elif kind=='wrong_horizon':p['horizon_hours']=24
    elif kind=='legacy_power':obs[0]['power_kw']=42
    response=client.post('/api/v1/forecast',json=p)
    assert response.status_code==422,response.text


def test_reject_tampered_artifact(client,tmp_path,monkeypatch):
    site(client);p=request_payload()
    tag='15min_Solar1_processed_3_full_tr0.6_val0.2_s42'
    meta=json.loads((RELEASE/f'{tag}.json').read_text())
    (tmp_path/f'{tag}.json').write_text(json.dumps(meta));(tmp_path/meta['artifact_file']).write_bytes(b'not the published graph')
    monkeypatch.setenv("AMPDNET_MODEL_DIR", str(tmp_path))
    from app.infrastructure import model_store
    model_store.load_ampdnet_artifact.cache_clear()
    response=client.post('/api/v1/forecast',json=p)
    assert response.status_code==422 and 'SHA256' in response.text


def test_cross_night_output_is_rejected_with_valid_input():
    p=request_payload(task='4h')
    shift=timedelta(hours=5)
    p['issued_at']=(datetime.fromisoformat(p['issued_at'])+shift).isoformat()
    # Regenerate the exact compressed-day history ending at 15:00.
    current=datetime.fromisoformat(p['issued_at']);times=[current]
    while len(times)<192:
        prev=times[-1]-timedelta(minutes=15)
        if prev.hour<6:prev=prev.replace(hour=18,minute=45)-timedelta(days=1)
        times.append(prev)
    for row,ts in zip(p['options']['ampdnet_input']['observations'],reversed(times)):row['ts']=ts.isoformat()
    req=ForecastRequest.model_validate(p)
    with pytest.raises(ValueError,match='daylight boundary'):
        ampdnet.predict_pv_ampdnet(
            req.options.ampdnet_input, req.issued_at, 50000, 'Asia/Shanghai',
            metadata={'cap_kw': 50000}, infer=lambda _: [0] * 16,
            normalize=lambda raw, meta: raw, denormalize=lambda output, meta: output,
        )
