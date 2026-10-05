from __future__ import annotations

import json
from uuid import uuid4

from mlflow import MlflowClient

from noshow.registry.rollback import rollback


def main() -> None:
    tracking_uri = "http://localhost:5001"
    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )
    source_name = "clinic-noshow"
    demo_name = f"clinic-noshow-rollback-demo-{uuid4().hex[:8]}"

    versions = []
    for alias in ("champion", "challenger"):
        source = client.get_model_version_by_alias(source_name, alias)
        copied = client.copy_model_version(
            src_model_uri=f"models:/{source_name}/{source.version}",
            dst_name=demo_name,
        )
        client.set_model_version_tag(
            demo_name, copied.version, "threshold", source.tags["threshold"]
        )
        client.set_model_version_tag(demo_name, copied.version, "demo", "true")
        versions.append(copied.version)

    previous, current = versions
    client.set_registered_model_alias(demo_name, "previous_champion", previous)
    client.set_registered_model_alias(demo_name, "champion", current)

    print(f"Before rollback: {demo_name}, champion={current}")
    result = rollback(tracking_uri, demo_name)

    restored = client.get_model_version_by_alias(demo_name, "champion")
    if restored.version != previous:
        raise RuntimeError("Rollback did not restore the previous version")

    print(json.dumps(result, indent=2))
    print("ROLLBACK VERIFIED")


if __name__ == "__main__":
    main()
