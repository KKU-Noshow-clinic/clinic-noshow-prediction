from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import noshow.registry.rollback as rollback_module


def test_rollback_restores_previous_champion(monkeypatch):
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(
        aliases={"champion": "2", "previous_champion": "1"}
    )
    client.get_model_version.return_value = SimpleNamespace(
        status="READY", tags={"threshold": "0.5"}
    )
    monkeypatch.setattr(rollback_module, "MlflowClient", lambda **kwargs: client)

    result = rollback_module.rollback("http://localhost:5001")

    client.get_model_version.assert_called_once_with("clinic-noshow", "1")
    client.set_registered_model_alias.assert_called_once_with("clinic-noshow", "champion", "1")
    assert result["rolled_back_from"] == "2"
    assert result["champion_version"] == "1"
    assert result["threshold"] == 0.5


def test_rollback_without_previous_version_changes_nothing(monkeypatch):
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"champion": "1"})
    monkeypatch.setattr(rollback_module, "MlflowClient", lambda **kwargs: client)

    with pytest.raises(ValueError, match="No previous champion"):
        rollback_module.rollback("http://localhost:5001")

    client.set_registered_model_alias.assert_not_called()
