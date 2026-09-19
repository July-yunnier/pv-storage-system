import unittest
from datetime import datetime, timedelta, timezone

from app.domain.dispatch import DispatchInputs, RuleDispatchEngine
from app.domain.models import (
    LoadConfig,
    MeasurementRead,
    PVConfig,
    SiteLocation,
    SiteRead,
    StorageConfig,
    TariffCreate,
    TariffRead,
)
from app.domain.tariff import TouProvider


class DispatchEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        now = datetime.now(timezone.utc)
        self.site = SiteRead(
            site_id="S-TEST",
            name="测试站点",
            location=SiteLocation(lat=32.2, lon=118.7),
            pv=PVConfig(capacity_kwp=1000),
            load=LoadConfig(peak_kw=800),
            storage=StorageConfig(
                pcs_kw=500,
                capacity_kwh=1000,
                round_trip_eff=0.88,
                soc_min=0.10,
                soc_max=0.95,
            ),
            tariff_id="T-TEST",
            created_at=now,
            updated_at=now,
        )
        tariff = TariffRead(
            **TariffCreate(
                tariff_id="T-TEST",
                name="测试电价",
                transmission_and_fund=0.21,
                tiers=[
                    {"name": "peak", "price": 1.0, "hours": ["08:00-20:00"]},
                    {"name": "valley", "price": 0.3, "hours": ["20:00-08:00"]},
                ],
            ).model_dump(),
            created_at=now,
            updated_at=now,
        )
        start = datetime(2026, 9, 20, tzinfo=timezone(timedelta(hours=8)))
        self.pv = [
            MeasurementRead(
                data_type="pv_power",
                ts=start + timedelta(minutes=15 * index),
                value=0,
                unit="kW",
                source="test",
            )
            for index in range(96)
        ]
        self.load = [
            MeasurementRead(
                data_type="load_power",
                ts=start + timedelta(minutes=15 * index),
                value=800,
                unit="kW",
                source="test",
            )
            for index in range(96)
        ]
        self.prices = TouProvider(tariff).points(start, 96)

    def test_rule_dispatch_respects_soc_and_reduces_cost(self) -> None:
        result = RuleDispatchEngine().run(
            DispatchInputs(
                site=self.site,
                target_date=self.pv[0].ts.date(),
                initial_soc=0.5,
                pv_points=self.pv,
                load_points=self.load,
                price_points=self.prices,
            )
        )
        self.assertEqual(len(result.schedule), 96)
        self.assertGreater(result.expected_benefit_yuan, 0)
        self.assertTrue(any(point.action == "charge" for point in result.schedule))
        self.assertTrue(any(point.action == "discharge" for point in result.schedule))
        self.assertTrue(all(0.1 <= point.soc_end <= 0.95 for point in result.schedule))
        self.assertTrue(all(point.grid_export_kw == 0 for point in result.schedule))


if __name__ == "__main__":
    unittest.main()
