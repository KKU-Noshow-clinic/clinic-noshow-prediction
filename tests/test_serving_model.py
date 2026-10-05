"""Serving logic that does not need FastAPI: schema rules, scoring, threshold, model holder."""

import pandas as pd
import pytest
from pydantic import ValidationError

from noshow.serving.model import ModelHolder
from noshow.serving.schemas import Appointment


def test_record_matches_training_columns(valid):
    record = Appointment(**valid).to_record()
    assert set(record) == set(valid)
    assert record["ScheduledDay"].endswith("Z")


def test_handcap_is_collapsed_like_cleaning(valid):
    record = Appointment(**{**valid, "Handcap": 3}).to_record()
    assert record["Handcap"] == 1


def test_date_only_and_naive_timestamps_are_utc(valid):
    appt = Appointment(
        **{**valid, "ScheduledDay": "2016-05-04 08:00", "AppointmentDay": "2016-05-10"}
    )
    assert appt.ScheduledDay.utcoffset().total_seconds() == 0
    assert appt.AppointmentDay.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    "change",
    [
        {"Gender": "X"},
        {"Age": -1},
        {"Age": 150},
        {"SMS_received": 2},
        {"Handcap": 5},
        {"PatientId": 0},
        {"ScheduledDay": "2016-05-11T08:00:00Z"},  # booked after the appointment
        {"No-show": "Yes"},  # label must not be sent
    ],
)
def test_invalid_input_rejected(change, valid):
    with pytest.raises(ValidationError):
        Appointment(**{**valid, **change})


def test_missing_field_rejected(valid):
    data = dict(valid)
    data.pop("Age")
    with pytest.raises(ValidationError):
        Appointment(**data)


def test_bundle_matches_local_pipeline(bundle, fitted_pipeline, valid):
    """API path (schema -> record -> bundle) gives the same score as calling the pipeline."""
    record = Appointment(**valid).to_record()
    served = bundle.predict([record])[0]
    local = fitted_pipeline.predict_proba(pd.DataFrame([valid]))[0, 1]
    assert served["noshow_score"] == pytest.approx(local, abs=1e-6)
    assert served["alert"] == (served["noshow_score"] >= bundle.threshold)


def test_threshold_controls_alert(bundle, valid):
    record = Appointment(**valid).to_record()
    score = bundle.predict([record])[0]["noshow_score"]
    low = type(bundle)(bundle.model, threshold=0.0, version="t", uri="t")
    high = type(bundle)(bundle.model, threshold=1.0, version="t", uri="t")
    assert low.predict([record])[0]["alert"] is True
    assert high.predict([record])[0]["alert"] is (score >= 1.0)


def test_unseen_neighbourhood_and_new_patient_still_score(bundle, valid):
    record = Appointment(
        **{**valid, "Neighbourhood": "NEW PLACE", "PatientId": 999999.0}
    ).to_record()
    score = bundle.predict([record])[0]["noshow_score"]
    assert 0.0 <= score <= 1.0


def test_holder_keeps_old_model_when_reload_fails(bundle):
    calls = iter([bundle, RuntimeError("mlflow down")])

    def loader():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    holder = ModelHolder(loader=loader)
    assert holder.reload() is bundle
    assert holder.reload() is bundle  # failure keeps serving the previous model
    assert "mlflow down" in holder.last_error


def test_holder_starts_empty_when_registry_unreachable():
    def loader():
        raise ConnectionError("no champion yet")

    holder = ModelHolder(loader=loader)
    assert holder.reload() is None
    assert holder.get() is None
    assert "no champion yet" in holder.last_error
