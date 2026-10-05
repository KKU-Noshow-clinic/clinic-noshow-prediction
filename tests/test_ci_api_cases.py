"""The presentation cases (tests/fixtures/api_cases.json) must give the expected status in CI."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from noshow.serving import app as app_module

CASES = json.loads((Path(__file__).parent / "fixtures" / "api_cases.json").read_text("utf-8"))
client = TestClient(app_module.app)


@pytest.fixture
def loaded(bundle, monkeypatch):
    monkeypatch.setattr(app_module.holder, "bundle", bundle)
    monkeypatch.setattr(app_module.holder, "last_error", None)


def test_cases_cover_normal_and_abnormal():
    statuses = [case["expected_status"] for case in CASES]
    assert statuses.count(200) >= 5
    assert statuses.count(422) >= 5


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_demo_case(loaded, case):
    resp = client.post("/predict", json=case["request"])
    assert resp.status_code == case["expected_status"], resp.text
    if resp.status_code == 200:
        assert 0.0 <= resp.json()["noshow_score"] <= 1.0
