from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.domain.forecast.pv import predict_pv_intraday


def history():
    tz = ZoneInfo("Asia/Shanghai")
    start = datetime(2026, 9, 1, tzinfo=tz)
    return tz, [(start + timedelta(minutes=15 * i), 20.) for i in range(96 * 4)]


@pytest.mark.parametrize("hour,minute,count,first_hour,first_minute", [(10, 45, 52, 11, 0), (10, 59, 52, 11, 0), (23, 45, 0, 0, 0)])
def test_minute_slot_and_future_only(hour, minute, count, first_hour, first_minute):
    tz, rows = history()
    issued = datetime(2026, 9, 5, hour, minute, tzinfo=tz)
    result = predict_pv_intraday(rows, issued, 100., tz)
    assert len(result.points) == count
    assert all(p.ts > issued for p in result.points)
    if count:
        assert (result.points[0].ts.hour, result.points[0].ts.minute) == (first_hour, first_minute)


def test_later_measurements_do_not_change_earlier_prediction():
    tz, rows = history()
    issued = datetime(2026, 9, 5, 10, 45, tzinfo=tz)
    rows.append((issued, 40.))
    before = predict_pv_intraday(rows, issued, 100., tz)
    rows.append((issued + timedelta(minutes=1), 999.))
    after = predict_pv_intraday(rows, issued, 100., tz)
    assert before.model_dump() == after.model_dump()


def test_non_aligned_issue_uses_actual_elapsed_decay():
    tz, rows = history()
    issued = datetime(2026, 9, 5, 10, 59, tzinfo=tz)
    rows.append((issued.replace(minute=45), 40.))
    result = predict_pv_intraday(rows, issued, 100., tz, w_dev=1)
    import math
    assert result.points[0].value == pytest.approx(20 + .8 * 20 * math.exp(-1/90), abs=1e-6)
