"""Request/response schemas for the serving API.

Field names match the raw Kaggle columns so the request can be passed straight into the
sklearn Pipeline logged by the modeling workstream (no renaming = no training-serving skew).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Binary = Literal[0, 1]


class Appointment(BaseModel):
    """One appointment to score. `No-show` is NOT sent: it is the label we predict."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "PatientId": 29872499824296.0,
                "Gender": "F",
                "ScheduledDay": "2016-05-04T08:00:00Z",
                "AppointmentDay": "2016-05-10T00:00:00Z",
                "Age": 45,
                "Neighbourhood": "JARDIM DA PENHA",
                "Scholarship": 0,
                "Hipertension": 1,
                "Diabetes": 0,
                "Alcoholism": 0,
                "Handcap": 0,
                "SMS_received": 1,
            }
        },
    )

    PatientId: float = Field(gt=0)
    Gender: Literal["F", "M"]
    ScheduledDay: datetime
    AppointmentDay: datetime
    Age: int = Field(ge=0, le=120)
    Neighbourhood: str = Field(min_length=1, max_length=100)
    Scholarship: Binary
    Hipertension: Binary
    Diabetes: Binary
    Alcoholism: Binary
    Handcap: int = Field(ge=0, le=4)
    SMS_received: Binary

    @field_validator("ScheduledDay", "AppointmentDay")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        # Training data is UTC ("...Z"); treat naive timestamps/dates as UTC too.
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("Neighbourhood")
    @classmethod
    def _upper(cls, value: str) -> str:
        # Raw data stores neighbourhoods in upper case.
        return value.strip().upper()

    @model_validator(mode="after")
    def _booked_before_appointment(self) -> Appointment:
        # Same rule as data/clean.py: the booking must not be after the appointment day.
        if self.ScheduledDay.date() > self.AppointmentDay.date():
            raise ValueError("ScheduledDay must be on or before AppointmentDay")
        return self

    def to_record(self) -> dict:
        record = self.model_dump(mode="json")
        # Same rule as data/clean.py: Handcap 2-4 is collapsed to 1.
        record["Handcap"] = min(record["Handcap"], 1)
        return record


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appointments: list[Appointment] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    PatientId: float
    noshow_score: float = Field(
        description="Model score for the no-show class (class-weighted, NOT a calibrated "
        "probability: use it for ranking/alerting, not as an exact percentage)."
    )
    alert: bool = Field(description="True when noshow_score >= threshold -> send SMS/call")
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    n_alerts: int
