"""CI infrastructure compatibility (Lecture 10): the selected model can be saved and loaded back."""

from pathlib import Path

import mlflow.sklearn
import numpy as np
import yaml
from conftest import synthetic_appointments
from sklearn.pipeline import Pipeline

from noshow.features import build_preprocessor
from noshow.models.train import experiments, load_config


def test_selected_model_artifact_round_trip(tmp_path):
    # Train the selected experiment on small synthetic data (CI has no real dataset).
    X, y = synthetic_appointments()
    selected = load_config()["model"]["selected_experiment"]
    estimator = next(model for name, model, _ in experiments(seed=42) if name == selected)
    model = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
    model.fit(X, (y == "Yes").astype(int))

    # Save the same way as train.py, load the same way as serving.
    path = tmp_path / "model"
    mlflow.sklearn.save_model(
        model, str(path), serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    loaded = mlflow.sklearn.load_model(str(path))

    np.testing.assert_allclose(loaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])
    size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_000_000
    gate = yaml.safe_load(Path("configs/slo.yaml").read_text(encoding="utf-8"))["gate"]
    assert size_mb < gate["max_model_size_mb"]
