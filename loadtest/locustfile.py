"""Locust load test for the serving API.

Run (API must be up, e.g. `make up` or `make api`):
    make loadtest
or by hand:
    uv run locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 2m --csv loadtest/results/run
"""

import random
from datetime import UTC, datetime, timedelta

from locust import HttpUser, between, task

NEIGHBOURHOODS = [
    "JARDIM CAMBURI",
    "MARIA ORTIZ",
    "RESISTÊNCIA",
    "JARDIM DA PENHA",
    "ITARARÉ",
    "CENTRO",
    "TABUAZEIRO",
    "SANTA MARTHA",
]


def random_appointment() -> dict:
    scheduled = datetime(2016, 4, 25, tzinfo=UTC) + timedelta(
        days=random.randint(0, 40), hours=random.randint(7, 17)
    )
    appointment = scheduled.replace(hour=0) + timedelta(days=random.choice([0, 1, 3, 7, 14, 30]))
    return {
        "PatientId": float(random.randint(1, 50_000)),
        "Gender": random.choice(["F", "M"]),
        "ScheduledDay": scheduled.isoformat().replace("+00:00", "Z"),
        "AppointmentDay": appointment.isoformat().replace("+00:00", "Z"),
        "Age": random.randint(0, 95),
        "Neighbourhood": random.choice(NEIGHBOURHOODS),
        "Scholarship": int(random.random() < 0.1),
        "Hipertension": int(random.random() < 0.2),
        "Diabetes": int(random.random() < 0.07),
        "Alcoholism": int(random.random() < 0.03),
        "Handcap": int(random.random() < 0.02),
        "SMS_received": int(random.random() < 0.32),
    }


class ClinicUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=random_appointment())

    @task(1)
    def predict_batch(self):
        batch = [random_appointment() for _ in range(20)]
        self.client.post("/predict_batch", json={"appointments": batch})

    @task(1)
    def health(self):
        self.client.get("/health")
