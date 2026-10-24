"""Guard monthly maximum settlement and unchanged physical execution."""
import pandas as pd
import pytest

from scripts.analyze_llm_live_p1_p3 import paired_month_ci, trace_execute
from scripts.run_official_experiment import CAP, DT, ETA


def test_demand_is_one_monthly_maximum_not_sum_of_daily_peaks():
    agent = pd.DataFrame({'cost_yuan': [2., 2.], 'peak_import_kw': [4., 4.]})
    base = pd.DataFrame({'cost_yuan': [3., 3.], 'peak_import_kw': [10., 10.]})
    # Two daily energy differences of 1 plus ONE demand difference of 6 * 35.
    assert paired_month_ci(agent, base, 35.) == pytest.approx([212., 212.])


def test_monthly_maximum_is_recomputed_in_each_paired_resample():
    agent = pd.DataFrame({'cost_yuan': [0., 0.], 'peak_import_kw': [4., 4.]})
    base = pd.DataFrame({'cost_yuan': [0., 0.], 'peak_import_kw': [10., 0.]})
    # Resampling the low-peak day twice removes the observed 10 kW peak.
    assert paired_month_ci(agent, base, 35.) == pytest.approx([-140., 210.])


def test_interval_diagnostics_preserve_executor_clipping_and_energy_balance():
    trace, batch = trace_execute(
        [('charge', 500.), ('discharge', 500.), ('discharge', 500.)],
        [0., 0., 0.], [100., 100., 0.], [.5, 1., 1.],
        pd.date_range('2019-04-01', periods=3, freq='15min', tz='Asia/Shanghai'),
        initial_soc=.94,
    )
    # First slot clips at SOC 95%, second at net load 100 kW, last at zero load.
    charge = .01 * CAP / ETA / DT
    assert trace.charge_kw.tolist() == pytest.approx([charge, 0., 0.])
    assert trace.discharge_kw.tolist() == pytest.approx([0., 100., 0.])
    assert batch['cost_yuan'] == pytest.approx((100. + charge) * DT * .5)
    assert batch['end_soc'] == pytest.approx(.95 - 100. * DT / ETA / CAP)
    assert trace.soc_violations.sum() == 0
