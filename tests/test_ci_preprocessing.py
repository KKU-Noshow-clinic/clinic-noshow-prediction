"""CI unit tests for preprocessing: clean() and the shared feature pipeline."""

import numpy as np
import pandas as pd

from noshow.data.clean import clean
from noshow.features import AppointmentFeatureBuilder, build_preprocessor

GOOD = "tests/fixtures/good_data.csv"


def appointments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientId": [1.0, 1.0, 2.0],
            "Gender": ["F", "F", "M"],
            "ScheduledDay": [
                "2016-05-01T08:00:00Z",
                "2016-05-04T08:00:00Z",
                "2016-05-01T08:00:00Z",
            ],
            "AppointmentDay": ["2016-05-03", "2016-05-05", "2016-05-04"],
            "Age": [0, 18, 80],
            "Neighbourhood": ["A", "A", "B"],
            "Scholarship": [0, 0, 1],
            "Hipertension": [0, 0, 1],
            "Diabetes": [0, 0, 0],
            "Alcoholism": [0, 0, 0],
            "Handcap": [0, 0, 0],
            "SMS_received": [1, 1, 0],
        }
    )


def test_clean_drops_booking_after_appointment_but_keeps_same_day():
    rows = pd.concat([pd.read_csv(GOOD).iloc[[0]]] * 2, ignore_index=True)
    rows.loc[0, ["ScheduledDay", "AppointmentDay"]] = [
        "2016-05-10T15:00:00Z",
        "2016-05-10T00:00:00Z",
    ]
    rows.loc[1, ["ScheduledDay", "AppointmentDay"]] = [
        "2016-05-11T08:00:00Z",
        "2016-05-10T00:00:00Z",
    ]
    cleaned = clean(rows)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "ScheduledDay"].day == 10


def test_clean_does_not_change_its_input():
    raw = pd.read_csv(GOOD)
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_age_groups_and_weekday():
    X = appointments()
    features = AppointmentFeatureBuilder().fit(X).transform(X)
    assert features["age_group"].astype(str).tolist() == ["child", "teen", "elderly"]
    assert features["appointment_weekday"].tolist() == ["Tuesday", "Thursday", "Wednesday"]


def test_negative_lead_time_is_clipped_to_zero():
    X = appointments().iloc[[0]].assign(ScheduledDay="2016-05-05T08:00:00Z")
    assert AppointmentFeatureBuilder().fit(X).transform(X)["lead_time_days"].iloc[0] == 0


def test_new_patient_has_no_history():
    X = appointments()
    builder = AppointmentFeatureBuilder().fit(X, pd.Series([1, 1, 1]))
    features = builder.transform(X.iloc[[0]].assign(PatientId=999.0))
    assert features["previous_appointments"].iloc[0] == 0
    assert features["previous_noshow_rate"].iloc[0] == 0


def test_preprocessor_output_is_numeric_and_finite():
    X = appointments()
    matrix = build_preprocessor().fit(X, pd.Series([1, 0, 0])).transform(X)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    assert dense.shape[0] == len(X)
    assert np.isfinite(dense.astype(float)).all()
