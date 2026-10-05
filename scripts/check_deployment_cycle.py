import argparse
import json
import os
from urllib.request import urlopen

from mlflow import MlflowClient
from prefect import flow

from noshow.pipeline.flow import deploy_model
from noshow.registry.promote import promote
from noshow.registry.rollback import rollback


def health(api_url: str) -> dict:
    with urlopen(f"{api_url.rstrip('/')}/health", timeout=30) as response:
        return json.load(response)


@flow(name="deployment-cycle-demo", log_prints=True)
def demo(model_name: str, api_url: str) -> None:
    if not model_name.startswith("clinic-noshow-integration-demo-"):
        raise ValueError("This script supports integration demo models only")

    tracking_uri = "http://localhost:5001"
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    main_before = client.get_registered_model("clinic-noshow").aliases.get("champion")
    aliases = client.get_registered_model(model_name).aliases
    original = aliases["champion"]
    candidate = aliases["challenger"]
    expected_uri = f"models:/{model_name}@champion"

    before = health(api_url)
    if (
        not before["model_loaded"]
        or before["model_uri"] != expected_uri
        or str(before["model_version"]) != str(original)
    ):
        raise ValueError("Demo API must load the original demo champion")
    if original == candidate:
        raise ValueError("Demo champion and challenger must differ")

    os.environ["NOSHOW_API_URL"] = api_url

    promotion = promote(tracking_uri, model_name=model_name)
    deployed = deploy_model(promotion)
    after_promote = health(api_url)
    if after_promote["model_uri"] != expected_uri or str(after_promote["model_version"]) != str(
        candidate
    ):
        raise RuntimeError("API did not load the promoted demo model")
    print("PROMOTE + RELOAD VERIFIED", deployed)

    reverted = rollback(tracking_uri, model_name=model_name)
    restored = deploy_model(reverted)
    after_rollback = health(api_url)
    if after_rollback["model_uri"] != expected_uri or str(after_rollback["model_version"]) != str(
        original
    ):
        raise RuntimeError("API did not load the rollback demo model")
    print("ROLLBACK + RELOAD VERIFIED", restored)

    main_after = client.get_registered_model("clinic-noshow").aliases.get("champion")
    if main_after != main_before:
        raise RuntimeError("Main champion changed during the demo")
    print("DEPLOYMENT CYCLE VERIFIED; main champion unchanged")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    demo(args.model_name, args.api_url)


if __name__ == "__main__":
    main()
