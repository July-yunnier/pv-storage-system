"""T3：真值隔离、风险量纲、能量加权与输入校验。"""
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from app.domain.forecast.price_features import PriceSeries, build_samples
from scripts import run_interval_dispatch_ablation as a
from scripts.run_price_benchmark import _forecast_window, _record, _rolling_days


def predictions(days=2):
    rows = []
    for day in pd.date_range('2013-08-01', periods=days, freq='D', tz='America/New_York'):
        actual = np.linspace(-0.05, 0.8, 24)
        _record(rows, 1, day, 'test', actual, actual - .1, actual, actual + .1)
    return pd.DataFrame(rows)


def response(discharge=40.):
    return SimpleNamespace(schedule=[SimpleNamespace(
        grid_import_kw=100., charge_from_grid_kw=1., charge_from_pv_kw=0.,
        power_kw=discharge, action='discharge', pv_curtailment_kw=0., soc_end=.1,
    ) for _ in range(96)], equivalent_full_cycles=.4)


def settlement():
    return a._settle(response(), np.repeat([.2]*20+[1.]*4, 4),
                     np.ones(96)*10, np.ones(96)*100, a._site(), 2.)


def test_peak_shortfall_units_and_penalty():
    result = settlement()
    # (100 kW target - 40 kW actual) * 4 peak hours = 240 kWh.
    assert result['supply_shortfall_kwh'] == 240
    assert result['shortfall_penalty_yuan'] == 480
    assert result['cost_yuan'] == 800
    assert result['total_cost_yuan'] == 1280
    assert result['reserve_deficit_max_kwh'] == 50


def test_shortfall_pcs_cap_positive_net_and_zero_coefficient():
    result = a._settle(response(), np.ones(96), np.zeros(96),
                       np.repeat([1000., -100.], 48), a._site(), 0)
    assert result['supply_shortfall_kwh'] == (250-40)*12
    assert result['total_cost_yuan'] == result['cost_yuan']


def test_negative_peak_never_creates_negative_penalty():
    result = a._settle(response(), np.ones(96)*-.1, np.zeros(96), np.ones(96)*100, a._site(), 2)
    assert result['cost_yuan'] < 0
    assert result['shortfall_penalty_yuan'] == 0


def test_settlement_alignment():
    with pytest.raises(ValueError, match='align'):
        a._settle(response(), np.ones(95), np.zeros(96), np.ones(96), a._site(), 2)


def test_summary_energy_weights_and_separate_premiums():
    first = dict(day='2013-08-01', policy='p50', stress=.1, margin=0., **settlement())
    second = dict(first, day='2013-08-02')
    first.update(mis_charge_kwh=24., charge_kwh=24.)
    second.update(mis_charge_kwh=0., charge_kwh=216.)
    interval = [dict(row, policy='interval', margin=.02, cost_yuan=810., total_cost_yuan=1300.)
                for row in (first, second)]
    summary = a._summarize(pd.DataFrame([first, second, *interval]))
    base = summary[summary.policy == 'p50'].iloc[0]
    robust = summary[summary.policy == 'interval'].iloc[0]
    assert base.mis_charge_ratio == pytest.approx(.1)  # Daily mean would be .5.
    assert robust.cost_delta_pct == pytest.approx(20/1600*100)
    assert robust.total_cost_delta_pct == pytest.approx(40/2560*100)
    assert base.days == 2


@pytest.mark.parametrize('defect', ['missing','duplicate','nan','unordered','naive','date','minute','gap'])
def test_validation_rejects_bad_days(defect):
    frame = predictions()
    if defect == 'missing':
        frame = frame.iloc[1:]
    elif defect == 'duplicate':
        frame.loc[1, 'ts'] = frame.loc[0, 'ts']
    elif defect == 'nan':
        frame.loc[0, 'actual'] = np.nan
    elif defect == 'unordered':
        frame.loc[0, 'p10'] = 10.
    elif defect == 'naive':
        frame.loc[0, 'ts'] = '2013-08-01T00:00:00'
    elif defect == 'date':
        frame.loc[0, 'ts'] = '2013-08-03T00:00:00-04:00'
    elif defect == 'minute':
        frame.loc[0, 'ts'] = '2013-08-01T00:15:00-04:00'
    else:
        frame['day'] = frame.day.str.replace('08-02','08-03')
        frame['ts'] = frame.ts.str.replace('08-02','08-03')
    with pytest.raises(ValueError):
        a._validate_predictions(frame, 'test')


def test_min_days_and_negative_prices_and_timezone():
    frame = predictions()
    assert len(a._validate_predictions(frame, 'test', 2)) == 48
    assert all(datetime.fromisoformat(value).utcoffset() is not None for value in frame.ts)
    assert (frame.actual < 0).any()
    with pytest.raises(ValueError, match='at least 60'):
        a._validate_predictions(frame, 'test', 60)


def series():
    n = 100*24
    return PriceSeries(ts=pd.date_range('2013-01-01', periods=n, freq='h', tz='Asia/Shanghai'),
                       price=np.arange(n, dtype=float)/10000, load=np.ones(n), load_total=np.ones(n)*2)


def test_rolling_days_missing_actuals_and_out_of_range():
    s = series()
    assert len(_rolling_days(s, '2013-02-01', 60)) == 60
    s.price[31*24] = np.nan
    with pytest.raises(ValueError, match='missing evaluation actuals'):
        _rolling_days(s, '2013-02-01', 60)
    with pytest.raises(ValueError):
        _rolling_days(s, '2014-01-01', 1)


def test_window_masks_future_labels_and_preserves_known_loads():
    s = series()
    origin = 80*24-1
    window, issue = _forecast_window(s, origin, 60)
    assert issue == 60*24-1
    assert np.isfinite(window.price[:issue+1]).all()
    assert np.isnan(window.price[issue+1:]).all()
    assert np.isfinite(window.load[issue+1:]).all()
    s.price[origin+1:] = 1e8
    changed, _ = _forecast_window(s, origin, 60)
    pd.testing.assert_frame_equal(
        build_samples(window, origins=np.array([issue]), require_target=False)[0],
        build_samples(changed, origins=np.array([issue]), require_target=False)[0])
    origins = np.arange(0, issue-24+1, 3)
    _, targets, _, _ = build_samples(window, origins=origins, require_target=True)
    assert np.isfinite(targets).all()


@pytest.mark.parametrize('day', ['2013-03-10', '2013-11-03'])
def test_rolling_rejects_dst_even_when_source_uses_naive_clock(day):
    s = PriceSeries(ts=pd.date_range('2013-01-01', periods=365*24, freq='h'),
                    price=np.ones(365*24))
    with pytest.raises(ValueError, match='24 distinct hours'):
        _rolling_days(s, day, 1)


def test_bias_independent_of_order_preserves_negative_quantiles():
    forward = {s:a._bias_for('2013-08-01',s) for s in (0.,.1,.3)}
    reverse = {s:a._bias_for('2013-08-01',s) for s in (.3,.1,0.)}
    for stress, factor in forward.items():
        np.testing.assert_array_equal(factor, reverse[stress])
        bands = np.array([-.3,-.2,-.1])[:,None]*factor
        assert (np.diff(bands, axis=0) >= 0).all()
        assert (bands < 0).all()


def test_cli_60_days_and_saved_summary(tmp_path, monkeypatch):
    path = tmp_path/'predictions.csv'
    predictions(60).to_csv(path,index=False)
    out = tmp_path/'output'
    monkeypatch.setattr('sys.argv', ['ablation','--predictions',str(path),'--model','test',
        '--output',str(out),'--price-scale','1','--min-days','60','--stress','.1,.2,.3'])
    assert a.main() == 0
    daily = pd.read_csv(out/'by_day.csv')
    summary = pd.read_csv(out/'summary.csv')
    assert len(daily) == 60*3*2
    assert set(summary.days) == {60}
    assert len(summary) == 6
    assert summary.supply_shortfall_kwh.ge(0).all()
    np.testing.assert_allclose(summary.total_cost_yuan, summary.cost_yuan+summary.shortfall_penalty_yuan)
