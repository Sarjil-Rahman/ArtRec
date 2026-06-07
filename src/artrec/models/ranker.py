from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    # Only use features available before a recommendation is shown. Post-event
    # signals such as dwell time are kept in raw data for analysis, but excluded
    # here to avoid target leakage and train/serve skew.
    "position",
    "popularity_score_pre",
    "affinity_score",
    "price",
    "margin",
    "freshness",
    "budget_mean",
    "price_sensitivity",
    "novelty_preference",
    "repeat_tolerance",
    "activity_level",
    "conversion_propensity",
    "save_propensity",
    "price_gap_ratio",
    "price_above_budget",
    "margin_to_price",
    "top_3_position",
    "top_5_position",
    "user_item_seen_before",
    "user_artist_seen_before",
    "user_style_seen_before",
    "user_item_not_interested_before",
    "user_artist_not_interested_before",
    "user_style_not_interested_before",
    "session_rank_inverse",
    "tag_count",
    "has_blue_tag",
    "has_nature_tag",
    "hour",
    "day_of_week",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["surface", "device", "country", "segment", "style"]
PINNED_ARTIFACT_ENVIRONMENT = (
    "Python >=3.10,<3.11 with numpy==2.2.6, pandas==2.3.3, "
    "scikit-learn==1.7.2, joblib==1.5.3, and lightgbm==4.6.0"
)


def make_ranking_label(df: pd.DataFrame) -> pd.Series:
    not_interested = df["not_interested"] if "not_interested" in df.columns else 0
    return (
        df["clicked"].astype(float) * 1.0
        + df["saved"].astype(float) * 1.2
        + df["add_to_cart"].astype(float) * 1.8
        + df["purchased_immediate"].astype(float) * 3.0
        - pd.Series(not_interested, index=df.index).astype(float) * 2.0
    )


@dataclass
class ArtRanker:
    pipeline: Pipeline

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "ArtRanker":
        X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        y = (make_ranking_label(train_df) >= 1.5).astype(int)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        clf = LogisticRegression(max_iter=400, class_weight="balanced")
        pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
        pipeline.fit(X, y)
        return cls(pipeline=pipeline)

    def score(self, df: pd.DataFrame):
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        try:
            return self.pipeline.predict_proba(X)[:, 1]
        except AttributeError as exc:
            raise RuntimeError(
                "The ranking artifact loaded, but scoring failed in scikit-learn. "
                f"The bundled joblib artifacts are intended for {PINNED_ARTIFACT_ENVIRONMENT}. "
                "Install with `python -m pip install -r requirements.txt` or rebuild "
                "artifacts with `python scripts/build_all.py` in the environment you "
                "plan to serve."
            ) from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ArtRanker":
        return joblib.load(path)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def _session_group_sizes(df: pd.DataFrame) -> list[int]:
    if "session_id" not in df.columns:
        return [len(df)]
    return df.groupby("session_id", sort=False).size().astype(int).tolist()


@dataclass
class LightGBMLearningToRanker:
    pipeline: Pipeline

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame | None = None,
    ) -> "LightGBMLearningToRanker":
        del validation_df
        train_df = train_df.sort_values(["session_id", "timestamp"]).reset_index(
            drop=True
        )
        y = np.rint(make_ranking_label(train_df).clip(lower=0)).astype(int)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        pipeline = Pipeline([("preprocessor", preprocessor), ("ranker", ranker)])
        pipeline.fit(
            _feature_frame(train_df),
            y,
            ranker__group=_session_group_sizes(train_df),
        )
        return cls(pipeline=pipeline)

    def score(self, df: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return self.pipeline.predict(_feature_frame(df))
