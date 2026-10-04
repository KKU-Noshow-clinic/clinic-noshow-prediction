from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL_FEATURES = ["Gender", "Neighbourhood", "appointment_weekday", "age_group"]
NUMERIC_FEATURES = [
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "lead_time_days",
    "previous_appointments",
    "previous_noshow_rate",
]


def _target_to_int(y: pd.Series) -> pd.Series:
    values = pd.Series(y, index=y.index)
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(int)
    return values.map({"No": 0, "Yes": 1}).astype(int)


class AppointmentFeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.patient_history_ = {}
        if y is not None:
            history = pd.DataFrame(
                {
                    "patient": X["PatientId"].to_numpy(),
                    "appointment_day": self._day_number(X["AppointmentDay"]),
                    "target": _target_to_int(pd.Series(y, index=X.index)).to_numpy(),
                }
            )
            for patient, rows in history.groupby("patient", sort=False):
                ordered = rows.sort_values("appointment_day")
                days = ordered["appointment_day"].to_numpy()
                outcomes = ordered["target"].to_numpy()
                self.patient_history_[patient] = (days, np.r_[0, np.cumsum(outcomes)])
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params):
        return self.fit(X, y).transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scheduled_days = self._day_number(X["ScheduledDay"])
        counts = np.zeros(len(X), dtype=float)
        rates = np.zeros(len(X), dtype=float)
        for position, (patient, scheduled_day) in enumerate(
            zip(X["PatientId"].to_numpy(), scheduled_days, strict=True)
        ):
            history = self.patient_history_.get(patient)
            if history is None:
                continue
            appointment_days, cumulative_noshows = history

            count = int(np.searchsorted(appointment_days, scheduled_day, side="left"))
            counts[position] = count
            if count:
                rates[position] = cumulative_noshows[count] / count
        return self._base_features(X).assign(
            previous_appointments=counts,
            previous_noshow_rate=rates,
        )

    @staticmethod
    def _day_number(values: pd.Series) -> np.ndarray:
        return pd.to_datetime(values, utc=True).dt.normalize().astype("int64").to_numpy()

    @staticmethod
    def _base_features(X: pd.DataFrame) -> pd.DataFrame:
        scheduled = pd.to_datetime(X["ScheduledDay"], utc=True)
        appointment = pd.to_datetime(X["AppointmentDay"], utc=True)
        lead_time = (appointment.dt.normalize() - scheduled.dt.normalize()).dt.days.clip(lower=0)
        age = pd.to_numeric(X["Age"])
        age_group = pd.cut(
            age,
            bins=[-np.inf, 12, 18, 35, 55, 75, np.inf],
            labels=["child", "teen", "young_adult", "adult", "senior", "elderly"],
        )
        result = X[
            [
                "Gender",
                "Neighbourhood",
                "Age",
                "Scholarship",
                "Hipertension",
                "Diabetes",
                "Alcoholism",
                "Handcap",
                "SMS_received",
            ]
        ].copy()
        result["lead_time_days"] = lead_time
        result["appointment_weekday"] = appointment.dt.day_name()
        result["age_group"] = age_group
        return result
