import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test.db")
        self.client = TestClient(create_app(self.db_path))
        site_payload = {
            "site_id": "S-API",
            "name": "API 测试站点",
            "location": {"lat": 32.2, "lon": 118.7, "timezone": "Asia/Shanghai"},
            "pv": {"capacity_kwp": 1000},
            "load": {"peak_kw": 800},
            "storage": {
                "pcs_kw": 500,
                "capacity_kwh": 1000,
                "round_trip_eff": 0.88,
                "soc_min": 0.10,
                "soc_max": 0.95,
            },
        }
        response = self.client.post("/api/v1/sites", json=site_payload)
        self.assertEqual(response.status_code, 201, response.text)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_health_and_schedule_flow(self) -> None:
        self.assertEqual(self.client.get("/api/v1/health").json()["status"], "ok")
        start = datetime(2026, 9, 20, tzinfo=timezone(timedelta(hours=8)))
        for data_type, value in [("pv_power", 0), ("load_power", 800)]:
            points = [
                {
                    "ts": (start + timedelta(minutes=15 * index)).isoformat(),
                    "value": value,
                }
                for index in range(96)
            ]
            response = self.client.post(
                "/api/v1/sites/S-API/measurements",
                json={
                    "data_type": data_type,
                    "unit": "kW",
                    "resolution_minutes": 15,
                    "points": points,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["accepted"], 96)

        response = self.client.post(
            "/api/v1/storage/schedule",
            json={"site_id": "S-API", "date": "2026-09-20", "initial_soc": 0.5},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(body["schedule"]), 96)
        self.assertGreater(body["expected_benefit_yuan"], 0)


if __name__ == "__main__":
    unittest.main()
