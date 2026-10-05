import argparse
import json
import uuid

import mlflow
from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-uri", required=True)
    parser.add_argument("--candidate-uri", required=True)
    args = parser.parse_args()

    uri = "http://localhost:5001"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    name = f"clinic-noshow-integration-demo-{uuid.uuid4().hex[:8]}"
    baseline = mlflow.register_model(args.baseline_uri, name)
    candidate = client.copy_model_version(args.candidate_uri, name)

    for version in (baseline, candidate):
        client.set_model_version_tag(name, version.version, "threshold", "0.5")

    client.set_registered_model_alias(name, "champion", baseline.version)
    client.set_registered_model_alias(name, "challenger", candidate.version)

    print(
        json.dumps(
            {
                "model_name": name,
                "champion_version": baseline.version,
                "challenger_version": candidate.version,
                "model_uri": f"models:/{name}@champion",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
