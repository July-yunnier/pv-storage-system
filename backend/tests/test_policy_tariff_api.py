from fastapi.testclient import TestClient


VALID_CANDIDATE = {
    "policy_name": "测试政策",
    "base_price": 0.48,
    "transmission_and_fund": 0.21,
    "rules": [
        {"start_time": "00:00", "end_time": "08:00", "type": "valley", "multiplier": 0.5},
        {"start_time": "08:00", "end_time": "11:00", "type": "peak", "multiplier": 1.7},
        {"start_time": "11:00", "end_time": "12:00", "type": "sharp", "multiplier": 2.2},
        {"start_time": "12:00", "end_time": "17:00", "type": "flat", "multiplier": 1.0},
        {"start_time": "17:00", "end_time": "22:00", "type": "peak", "multiplier": 1.7},
        {"start_time": "22:00", "end_time": "24:00", "type": "valley", "multiplier": 0.5},
    ],
}


def test_policy_confirmation_requires_human_and_creates_tariff(client: TestClient) -> None:
    response = client.post(
        "/api/v1/policy-tariffs/confirm",
        json={
            "candidate": VALID_CANDIDATE,
            "tariff_id": "T-POLICY-API",
            "target_date": "2026-10-04",
            "confirmed_by": "test-reviewer",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["tariff_id"] == "T-POLICY-API"
    assert {tier["name"] for tier in body["tiers"]} == {"valley", "peak", "sharp", "flat"}


def test_policy_confirmation_rejects_gap(client: TestClient) -> None:
    candidate = {**VALID_CANDIDATE, "rules": [dict(rule) for rule in VALID_CANDIDATE["rules"]]}
    candidate["rules"][1]["start_time"] = "08:15"
    response = client.post(
        "/api/v1/policy-tariffs/confirm",
        json={"candidate": candidate, "confirmed_by": "test-reviewer"},
    )
    assert response.status_code == 422
    assert "08:00" in response.json()["error"]["message"]


def test_parse_text_without_api_config_returns_provider_error(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    response = client.post(
        "/api/v1/policy-tariffs/parse-text",
        json={"policy_name": "测试", "policy_text": "测试政策原文"},
    )
    assert response.status_code == 502
    assert "LLM_API_KEY" in response.json()["error"]["message"]
