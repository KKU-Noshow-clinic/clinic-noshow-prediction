"""Shared fixtures: a tiny model trained on synthetic data with the real preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from noshow.features import build_preprocessor
from noshow.serving.model import ModelBundle

VALID_APPOINTMENT = {
    "PatientId": 1001.0,
    "Gender": "F",
    "ScheduledDay": "2016-05-04T08:00:00Z",
    "AppointmentDay": "2016-05-10T00:00:00Z",
    "Age": 45,
    "Neighbourhood": "CENTRO",
    "Scholarship": 0,
    "Hipertension": 1,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handcap": 0,
    "SMS_received": 1,
}


def synthetic_appointments(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    scheduled = pd.Timestamp("2016-04-01", tz="UTC") + pd.to_timedelta(
        rng.integers(0, 40, n), unit="D"
    )
    lead = rng.integers(0, 30, n)
    X = pd.DataFrame(
        {
            "PatientId": rng.integers(1, 120, n).astype(float),
            "Gender": rng.choice(["F", "M"], n),
            "ScheduledDay": scheduled.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "AppointmentDay": (scheduled + pd.to_timedelta(lead, unit="D"))
            .normalize()
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Age": rng.integers(0, 95, n),
            "Neighbourhood": rng.choice(["CENTRO", "JARDIM DA PENHA", "MARIA ORTIZ"], n),
            "Scholarship": rng.integers(0, 2, n),
            "Hipertension": rng.integers(0, 2, n),
            "Diabetes": rng.integers(0, 2, n),
            "Alcoholism": rng.integers(0, 2, n),
            "Handcap": rng.integers(0, 2, n),
            "SMS_received": rng.integers(0, 2, n),
        }
    )
    # Longer lead time -> more no-shows, so the model learns something non-trivial.
    y = pd.Series(np.where(rng.random(n) < 0.1 + lead / 40, "Yes", "No"))
    return X, y


@pytest.fixture
def valid() -> dict:
    return dict(VALID_APPOINTMENT)


@pytest.fixture(scope="session")
def fitted_pipeline() -> Pipeline:
    X, y = synthetic_appointments()
    model = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=500))]
    )
    return model.fit(X, (y == "Yes").astype(int))


@pytest.fixture
def bundle(fitted_pipeline) -> ModelBundle:
    return ModelBundle(model=fitted_pipeline, threshold=0.5, version="test", uri="memory://test")
