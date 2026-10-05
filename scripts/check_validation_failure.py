import pandas as pd
from prefect import flow

from noshow.pipeline.flow import notify_failure, validate


@flow(
    name="validation-failure-demo",
    log_prints=True,
    on_failure=[notify_failure],
)
def demo():
    validate(pd.DataFrame())


if __name__ == "__main__":
    demo()
