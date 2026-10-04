from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshow.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    AppointmentFeatureBuilder,
)


def build_preprocessor() -> Pipeline:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    columns = ColumnTransformer(
        [("categorical", categorical, CATEGORICAL_FEATURES), ("numeric", numeric, NUMERIC_FEATURES)]
    )
    return Pipeline([("features", AppointmentFeatureBuilder()), ("columns", columns)])
