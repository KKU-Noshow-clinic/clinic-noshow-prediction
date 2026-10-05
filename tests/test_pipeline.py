from unittest.mock import Mock

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

import noshow.pipeline.flow as pipeline_module


def test_validate_rejects_empty_data():
    with pytest.raises(ValueError, match="Dataset is empty"):
        pipeline_module.validate.fn(pd.DataFrame())


def test_validate_rejects_missing_columns():
    invalid = pd.DataFrame({"Age": [25]})
    with pytest.raises(SchemaErrors):
        pipeline_module.validate.fn(invalid)


def test_invalid_data_stops_before_split_and_training(monkeypatch):
    split_mock = Mock()
    train_mock = Mock()

    monkeypatch.setattr(pipeline_module, "ingest", lambda path: pd.DataFrame())
    monkeypatch.setattr(pipeline_module, "validate", pipeline_module.validate.fn)
    monkeypatch.setattr(pipeline_module, "split_data", split_mock)
    monkeypatch.setattr(pipeline_module, "train_models", train_mock)

    with pytest.raises(ValueError, match="Dataset is empty"):
        pipeline_module.pipeline.fn()

    split_mock.assert_not_called()
    train_mock.assert_not_called()
