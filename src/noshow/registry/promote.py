from __future__ import annotations

import argparse
import json

from mlflow import MlflowClient

from noshow.registry.gate import check_gate


def promote(tracking_uri: str, model_name: str | None = None) -> dict:
    result = (
        check_gate(tracking_uri)
        if model_name is None
        else check_gate(tracking_uri, model_name=model_name)
    )
    if not result["passed"]:
        raise ValueError(f"Gate failed: {result['failures']}")

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )
    name = result["model_name"]
    version = result["version"]

    challenger = client.get_model_version_by_alias(name, "challenger")
    if challenger.version != version:
        raise RuntimeError("Challenger changed during gate checking; retry")

    registered = client.get_registered_model(name)
    previous = registered.aliases.get("champion")

    if previous is not None and previous != version:
        client.set_registered_model_alias(name, "previous_champion", previous)

    client.set_registered_model_alias(name, "champion", version)

    return {
        "model_name": name,
        "champion_version": version,
        "previous_champion_version": previous,
        "model_uri": f"models:/{name}@champion",
        "threshold": result["threshold"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://localhost:5001")
    args = parser.parse_args()
    print(json.dumps(promote(args.tracking_uri), indent=2))


if __name__ == "__main__":
    main()
