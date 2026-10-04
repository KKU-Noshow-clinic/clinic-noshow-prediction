from __future__ import annotations

import argparse
import json
import math
from urllib.request import Request, urlopen

from mlflow import MlflowClient


def rollback(tracking_uri: str, model_name: str = "clinic-noshow") -> dict:
    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )
    name = model_name
    registered = client.get_registered_model(name)
    current = registered.aliases.get("champion")
    previous = registered.aliases.get("previous_champion")

    if previous is None:
        raise ValueError("No previous champion is available for rollback")
    if current == previous:
        raise ValueError("Champion already points to the rollback version")

    target = client.get_model_version(name, previous)
    if target.status != "READY":
        raise ValueError("Previous champion is not READY")

    threshold = float(target.tags["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Previous champion has an invalid threshold")

    client.set_registered_model_alias(name, "champion", previous)

    return {
        "model_name": name,
        "rolled_back_from": current,
        "champion_version": previous,
        "model_uri": f"models:/{name}@champion",
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://localhost:5001")
    parser.add_argument("--model-name", default="clinic-noshow")
    parser.add_argument("--api-url", help="Reload the serving API after rollback")
    args = parser.parse_args()

    if args.api_url and args.model_name != "clinic-noshow":
        parser.error("--api-url supports only the clinic-noshow serving model")

    result = rollback(args.tracking_uri, args.model_name)
    print(json.dumps(result, indent=2), flush=True)

    if args.api_url:
        request = Request(
            f"{args.api_url.rstrip('/')}/reload",
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            deployment = json.load(response)

        expected = str(result["champion_version"])
        loaded = str(deployment.get("model_version"))
        if loaded != expected:
            raise RuntimeError(f"API loaded version {loaded}; expected rollback version {expected}")

        print(json.dumps({"deployment": deployment}, indent=2))


if __name__ == "__main__":
    main()
