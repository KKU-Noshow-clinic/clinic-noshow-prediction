"""CI data validation: check raw CSV files against RAW_SCHEMA (the data "blueprint").

    uv run python -m noshow.ci.data_validation tests/fixtures/good_data.csv
    uv run python -m noshow.ci.data_validation tests/fixtures/bad_data.csv --expect-fail

Exit code is 0 when every file gives the expected result (valid, or invalid with --expect-fail).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from noshow.data.schema import RAW_SCHEMA


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Check columns, types, allowed values and date strings. Raises on any problem."""
    if frame.empty:
        raise ValueError("Dataset is empty")

    validated = RAW_SCHEMA.validate(frame, lazy=True)
    for column in ("ScheduledDay", "AppointmentDay"):
        pd.to_datetime(validated[column], utc=True, errors="raise")
    return validated


def problems(path: Path) -> list[str]:
    """Return a readable list of validation problems for one CSV (empty list = valid)."""
    try:
        validate_raw(pd.read_csv(path))
    except SchemaErrors as error:
        cases = error.failure_cases
        return [
            f"column={row.column} check={row.check} value={row.failure_case!r}"
            for row in cases.itertuples()
        ]
    except (SchemaError, ValueError, pd.errors.ParserError) as error:
        return [f"{type(error).__name__}: {error}"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--expect-fail", action="store_true", help="succeed only if every file is rejected"
    )
    args = parser.parse_args(argv)

    ok = True
    for path in args.files:
        found = problems(path)
        status = "INVALID" if found else "VALID"
        wanted = "INVALID" if args.expect_fail else "VALID"
        expected = status == wanted
        ok &= expected
        print(f"{status}: {path} ({len(found)} problem(s), expected {wanted})")
        for line in found:
            print(f"  - {line}")
        if not expected:
            print(f"  !! unexpected result for {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
