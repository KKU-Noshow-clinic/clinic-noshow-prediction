"""CI data validation: the raw schema accepts good data and rejects every kind of bad value."""

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from noshow.ci.data_validation import main, problems, validate_raw
from noshow.data.schema import RAW_SCHEMA

GOOD = "tests/fixtures/good_data.csv"
BAD = "tests/fixtures/bad_data.csv"
BAD_CASES = "tests/fixtures/bad_data_cases.csv"


def good_row() -> pd.DataFrame:
    return pd.read_csv(GOOD).iloc[[0]].reset_index(drop=True)


@pytest.mark.parametrize(
    "column, value",
    [
        ("Gender", "U"),
        ("Age", 121),
        ("Age", -2),
        ("Scholarship", 2),
        ("Hipertension", -1),
        ("Diabetes", 5),
        ("Alcoholism", 3),
        ("Handcap", 5),
        ("SMS_received", 2),
        ("No-show", "Maybe"),
        ("PatientId", 0),
    ],
)
def test_one_bad_value_is_rejected(column, value):
    df = good_row()
    df[column] = df[column].astype(object)
    df.loc[0, column] = value
    with pytest.raises(SchemaError):
        RAW_SCHEMA.validate(df)


def test_missing_column_is_rejected():
    with pytest.raises(SchemaError):
        RAW_SCHEMA.validate(good_row().drop(columns="SMS_received"))


def test_extra_column_is_rejected():
    # strict=True reports an unknown column as SchemaErrors
    with pytest.raises((SchemaError, SchemaErrors)):
        RAW_SCHEMA.validate(good_row().assign(Notes="x"))


def test_empty_data_and_bad_dates_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_raw(pd.DataFrame())
    with pytest.raises(ValueError):
        validate_raw(good_row().assign(AppointmentDay="not a date"))


def test_bad_cases_fixture_reports_every_broken_column():
    columns = {line.split()[0] for line in problems(BAD_CASES)}
    expected = ["Age", "Scholarship", "Handcap", "AppointmentID", "PatientId", "SMS_received"]
    assert {f"column={name}" for name in expected} <= columns


def test_cli_exit_codes():
    assert main([GOOD]) == 0
    assert main([BAD, BAD_CASES, "--expect-fail"]) == 0
    assert main([BAD]) == 1  # bad data must not pass silently
    assert main([GOOD, "--expect-fail"]) == 1
