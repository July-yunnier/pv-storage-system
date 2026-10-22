"""合成日前区间与实测对账，不代表真实市场效果。"""
from datetime import date, datetime, time, timedelta, timezone
import json

import pytest

from app.application import observability_service as service
from tests.conftest import CN_TZ

METHOD = "price_diurnal_quantile"


def seed(client, offset=1, covered=12, resolution=60, method=METHOD, interval=True, missing=False, late=False):
    db = client.app.state.db
    day = datetime.now(CN_TZ).date() - timedelta(days=offset)
    start = datetime.combine(day, time.min, tzinfo=CN_TZ)
    points = [{"ts": (start + timedelta(hours=h)).isoformat(),
               "price_cny_per_kwh": 0.5 if h < covered else 0.8, "quality": "measured"} for h in range(24)]
    r = client.post('/api/v1/sites/S-API/price/spot-prices', json={"source": "pytest_synthetic", "points": points})
    assert r.status_code == 200, r.text
    forecast = []
    for i in range(1440 // resolution):
        p = {"ts": (start + timedelta(minutes=i * resolution)).astimezone(timezone.utc).isoformat(), "value": 0.5}
        if interval:
            p.update(p10=0.4, p90=0.6)
        forecast.append(p)
    if missing:
        forecast.pop()
    db.insert_forecast_archive(site_id='S-API', target='price', method=method, target_date=day,
        issued_at=start + timedelta(hours=1) if late else start - timedelta(hours=1), points=forecast, provenance=None)
    return day


@pytest.mark.parametrize('resolution', [15, 30, 60])
def test_hourly_alignment_and_idempotency(site_client, resolution):
    seed(site_client, resolution=resolution)
    db = site_client.app.state.db
    assert service.reconcile_site(db, 'S-API') == 1
    m = db.list_forecast_accuracy('S-API')[0]['metrics']
    assert m['interval_coverage'] == m['picp80'] == 0.5
    assert m['n_points'] == m['interval_n_points'] == 24
    assert m['mae'] == pytest.approx(0.15)
    assert m['width80'] == pytest.approx(0.2)
    assert service.reconcile_site(db, 'S-API') == 0


@pytest.mark.parametrize('count,warning', [(2, None), (3, 'price_interval_undercoverage')])
def test_warning_in_both_apis(site_client, count, warning):
    for offset in range(1, count + 1):
        seed(site_client, offset=offset)
    r = site_client.get('/api/v1/sites/S-API/price/forecast', params={'model': 'statistical'})
    assert r.status_code == 200, r.text
    assert r.json()['interval_coverage'] == 0.5
    assert r.json()['warning'] == warning
    r = site_client.get('/api/v1/sites/S-API/accuracy', params={'target': 'price', 'days': 1})
    assert r.status_code == 200, r.text
    summary = r.json()['summary'][f'price:{METHOD}']
    assert summary['interval_coverage'] == 0.5
    assert summary['warning'] == warning
    assert summary['consecutive_low_days'] == count


@pytest.mark.parametrize('kind', ['threshold', 'gap', 'method', 'unknown', 'stale', 'recovered'])
def test_warning_break_conditions(site_client, kind):
    for offset in (1, 2, 3):
        if kind == 'gap' and offset == 2:
            continue
        seed(site_client, offset=offset + (1 if kind == 'stale' else 0),
             covered=18 if kind == 'threshold' else (24 if kind == 'recovered' and offset == 1 else 12),
             method='another_model' if kind == 'method' and offset == 2 else METHOD,
             interval=not (kind == 'unknown' and offset == 2))
    db = site_client.app.state.db
    service.reconcile_site(db, 'S-API')
    assert service.price_interval_status(db, 'S-API', METHOD)['warning'] is None


@pytest.mark.parametrize('interval,expected', [(False, None), (True, 0.5)])
def test_old_metrics_backfilled_from_original_archive(site_client, interval, expected):
    day = seed(site_client, interval=interval)
    db = site_client.app.state.db
    db.upsert_forecast_accuracy(site_id='S-API', target='price', method=METHOD, target_date=day,
                               metrics={'mae': 0.1, 'picp80': 0.0})
    assert service.reconcile_site(db, 'S-API') == 1
    r = site_client.get('/api/v1/sites/S-API/accuracy', params={'target': 'price'})
    assert r.status_code == 200, r.text
    assert r.json()['rows'][0]['metrics']['picp80'] == expected
    summary = r.json()['summary'][f'price:{METHOD}']
    assert summary['interval_coverage'] == summary['picp80'] == expected
    assert summary['interval_days'] == int(interval)
    assert service.reconcile_site(db, 'S-API') == 0


@pytest.mark.parametrize('kind', ['missing', 'late', 'naive', 'duplicate'])
def test_invalid_or_incomplete_archive_not_reconciled(site_client, kind):
    seed(site_client, missing=kind == 'missing', late=kind == 'late')
    db = site_client.app.state.db
    if kind in ('naive', 'duplicate'):
        with db.connection() as conn:
            row = conn.execute('SELECT id, points_json FROM forecast_archive').fetchone()
            points = json.loads(row['points_json'])
            if kind == 'naive':
                points[0]['ts'] = '2026-01-01T00:00:00'
            else:
                points[-1] = points[0]
            conn.execute('UPDATE forecast_archive SET points_json=? WHERE id=?', (json.dumps(points), row['id']))
    assert service.reconcile_site(db, 'S-API') == 0


def test_multiday_price_api_archives_intervals(site_client):
    r = site_client.get('/api/v1/sites/S-API/price/forecast', params={'days': 3})
    assert r.status_code == 200, r.text
    assert r.json()['interval_coverage'] is None
    rows = site_client.app.state.db.get_unreconciled_forecasts('S-API', date(2100, 1, 1))
    assert len(rows) == 3
    assert all(len(row['points']) == 96 for row in rows)
    for row in rows:
        assert len({datetime.fromisoformat(p['ts']).astimezone(CN_TZ).date() for p in row['points']}) == 1
        assert all('p10' in p and 'p90' in p for p in row['points'])


def test_hourly_means_and_inclusive_boundaries(site_client):
    seed(site_client, resolution=15, covered=24)
    db = site_client.app.state.db
    with db.connection() as conn:
        row = conn.execute('SELECT id, points_json FROM forecast_archive').fetchone()
        points = json.loads(row['points_json'])
        for i, p in enumerate(points):
            p.update(p10=0.4 if i % 2 == 0 else 0.6, value=0.7, p90=0.9)
        conn.execute('UPDATE forecast_archive SET points_json=? WHERE id=?', (json.dumps(points), row['id']))
    service.reconcile_site(db, 'S-API')
    assert db.list_forecast_accuracy('S-API')[0]['metrics']['interval_coverage'] == 1.0
