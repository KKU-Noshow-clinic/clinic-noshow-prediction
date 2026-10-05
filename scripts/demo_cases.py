"""Presentation demo: send normal and abnormal requests to a running API, then validate CSVs.

    make up                                   # or: make api
    uv run python scripts/demo_cases.py [--host http://localhost:8000]

API cases are in tests/fixtures/api_cases.json (the same cases run in CI against a test model:
tests/test_demo_cases.py). Normal cases must return 200, abnormal ones 422.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

from noshow.ci.data_validation import problems

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "fixtures" / "api_cases.json"
CSV_CASES = [
    (ROOT / "tests" / "fixtures" / "good_data.csv", True),
    (ROOT / "tests" / "fixtures" / "bad_data.csv", False),
    (ROOT / "tests" / "fixtures" / "bad_data_cases.csv", False),
]


def run_api_cases(host: str) -> bool:
    print(f"API cases -> {host}/predict")
    all_ok = True
    for case in json.loads(CASES.read_text(encoding="utf-8")):
        resp = requests.post(f"{host}/predict", json=case["request"], timeout=10)
        ok = resp.status_code == case["expected_status"]
        all_ok &= ok
        if resp.status_code == 200:
            body = resp.json()
            detail = f"score={body['noshow_score']:.3f} alert={body['alert']}"
        else:
            detail = "; ".join(e["msg"] for e in resp.json().get("detail", []))[:80]
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {case['name']:<30} {resp.status_code} {detail}  ({case['why']})")
    return all_ok


def run_csv_cases() -> bool:
    print("\nData validation (RAW_SCHEMA)")
    all_ok = True
    for path, should_pass in CSV_CASES:
        found = problems(path)
        ok = (not found) == should_pass
        all_ok &= ok
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {path.name}: {'VALID' if not found else f'{len(found)} problem(s)'}")
        for line in found:
            print(f"         - {line}")
    return all_ok


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # case descriptions are in Thai (Windows console)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    args = parser.parse_args()
    ok = run_api_cases(args.host.rstrip("/"))
    ok &= run_csv_cases()
    print("\nAll cases behaved as expected" if ok else "\nSome cases did NOT behave as expected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
