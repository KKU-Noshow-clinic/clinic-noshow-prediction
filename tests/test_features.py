import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from noshow.features import AppointmentFeatureBuilder, build_preprocessor


def appointments():
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
            "Age": [20, 20, 70],
            "Neighbourhood": ["A", "A", "B"],
            "Scholarship": [0, 0, 1],
            "Hipertension": [0, 0, 1],
            "Diabetes": [0, 0, 0],
            "Alcoholism": [0, 0, 0],
            "Handcap": [0, 0, 0],
            "SMS_received": [1, 1, 0],
        }
    )


def test_feature_values_and_history_are_leakage_safe():
    X = appointments()
    y = pd.Series([1, 0, 0])
    features = AppointmentFeatureBuilder().fit_transform(X, y)
    assert features["lead_time_days"].tolist() == [2, 1, 3]
    assert features["previous_appointments"].tolist() == [0.0, 1.0, 0.0]
    assert features.loc[1, "previous_noshow_rate"] == 1.0


def test_future_outcome_is_not_used_at_scheduling_time():
    X = appointments()
    X.loc[1, "ScheduledDay"] = "2016-05-02T08:00:00Z"
    features = AppointmentFeatureBuilder().fit_transform(X, pd.Series([1, 0, 0]))
    assert features.loc[1, "previous_appointments"] == 0
    assert features.loc[1, "previous_noshow_rate"] == 0


def test_serving_history_is_as_of_scheduling_day():
    X = appointments()
    builder = AppointmentFeatureBuilder().fit(X.iloc[[0]], pd.Series([1]))
    early = X.iloc[[1]].assign(ScheduledDay="2016-05-03T12:00:00Z")
    later = X.iloc[[1]].assign(ScheduledDay="2016-05-04T08:00:00Z")
    assert builder.transform(early)["previous_appointments"].iloc[0] == 0
    assert builder.transform(later)["previous_noshow_rate"].iloc[0] == 1


def test_shared_pipeline_handles_unseen_categories():
    X = appointments()
    y = pd.Series([1, 0, 0])
    model = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=200))]
    )
    model.fit(X, y)
    unseen = X.iloc[[0]].assign(Neighbourhood="NEW")
    assert model.predict_proba(unseen).shape == (1, 2)
