from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from artrec.utils.common import ensure_dir
from artrec.data.io import write_csv


def _cosine_sim(a, b) -> float:
    left = np.array(a, dtype=float)
    right = np.array(b, dtype=float)
    return float(np.dot(left, right) / ((np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9))


def _historical_count_before_timestamp(
    df: pd.DataFrame,
    keys: list[str],
    value_col: str | None = None,
) -> pd.Series:
    work = df[keys + ["timestamp"] + ([value_col] if value_col else [])].copy()
    work["_row_id"] = np.arange(len(work))
    work["_event_time"] = pd.to_datetime(work["timestamp"])
    group_cols = keys + ["_event_time"]
    if value_col is None:
        by_time = work.groupby(group_cols, dropna=False).size().rename("_count").reset_index()
    else:
        by_time = (
            work.groupby(group_cols, dropna=False)[value_col]
            .sum()
            .rename("_count")
            .reset_index()
        )
    by_time = by_time.sort_values(keys + ["_event_time"])
    by_time["_prior_count"] = by_time.groupby(keys, dropna=False)["_count"].cumsum() - by_time["_count"]
    work = work.merge(by_time[group_cols + ["_prior_count"]], on=group_cols, how="left")
    return work.sort_values("_row_id")["_prior_count"].fillna(0).astype(int)


def build_training_frame(
    users_df: pd.DataFrame, catalog_df: pd.DataFrame, impressions_df: pd.DataFrame
) -> pd.DataFrame:
    user_cols = [
        "user_id",
        "segment",
        "budget_mean",
        "price_sensitivity",
        "novelty_preference",
        "repeat_tolerance",
        "activity_level",
        "conversion_propensity",
        "save_propensity",
        "preferred_styles",
        "taste_vector",
    ]
    user_base = users_df[user_cols].copy()
    user_base["preferred_style_count"] = (
        user_base["preferred_styles"]
        .fillna("")
        .apply(lambda x: len([t for t in str(x).split("|") if t]))
    )
    item_base = catalog_df.copy()
    item_base["tag_count"] = (
        item_base["tags"]
        .fillna("")
        .apply(lambda x: len([t for t in str(x).split("|") if t]))
    )
    item_base["has_blue_tag"] = (
        item_base["tags"].fillna("").apply(lambda x: int("blue" in str(x).split("|")))
    )
    item_base["has_nature_tag"] = (
        item_base["tags"].fillna("").apply(lambda x: int("nature" in str(x).split("|")))
    )
    item_base = item_base[
        ["item_id", "tag_count", "has_blue_tag", "has_nature_tag", "embedding"]
    ]

    impressions = impressions_df.copy()
    if "not_interested" not in impressions.columns:
        impressions["not_interested"] = 0
    df = impressions.merge(user_base, on="user_id", how="left").merge(
        item_base, on="item_id", how="left"
    )
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["top_3_position"] = (df["position"] <= 2).astype(int)
    df["top_5_position"] = (df["position"] <= 4).astype(int)
    df["dwell_log1p"] = (
        df["dwell_seconds"].fillna(0).apply(lambda x: 0 if x <= 0 else math.log1p(x))
    )
    df["price_gap_ratio"] = (df["price"] - df["budget_mean"]) / df["budget_mean"].clip(
        lower=1.0
    )
    df["price_above_budget"] = (df["price"] > df["budget_mean"]).astype(int)
    df["margin_to_price"] = df["margin"] / df["price"].clip(lower=1.0)
    if "retrieval_similarity_pre" not in df.columns:
        df["retrieval_similarity_pre"] = df.apply(
            lambda row: _cosine_sim(row["taste_vector"], row["embedding"]),
            axis=1,
        )
    df["user_item_seen_before"] = _historical_count_before_timestamp(
        df, ["user_id", "item_id"]
    )
    df["user_artist_seen_before"] = _historical_count_before_timestamp(
        df, ["user_id", "artist_id"]
    )
    df["user_style_seen_before"] = _historical_count_before_timestamp(
        df, ["user_id", "style"]
    )
    df["not_interested"] = df["not_interested"].fillna(0).astype(int)
    for keys, out_col in [
        (["user_id", "item_id"], "user_item_not_interested_before"),
        (["user_id", "artist_id"], "user_artist_not_interested_before"),
        (["user_id", "style"], "user_style_not_interested_before"),
    ]:
        df[out_col] = _historical_count_before_timestamp(
            df, keys, value_col="not_interested"
        )
    df["session_rank_inverse"] = 1.0 / (df["position"] + 1)
    df = df.drop(columns=["embedding", "taste_vector"], errors="ignore")
    return df


def train_test_split_by_time(feature_df: pd.DataFrame, test_ratio: float = 0.2):
    feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(feature_df) * (1 - test_ratio))
    return feature_df.iloc[:split_idx].copy(), feature_df.iloc[split_idx:].copy()


def train_validation_test_split_by_time(
    feature_df: pd.DataFrame,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
):
    if validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("validation_ratio and test_ratio must be positive")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1")

    feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)
    train_end = int(len(feature_df) * (1 - validation_ratio - test_ratio))
    validation_end = int(len(feature_df) * (1 - test_ratio))
    return (
        feature_df.iloc[:train_end].copy(),
        feature_df.iloc[train_end:validation_end].copy(),
        feature_df.iloc[validation_end:].copy(),
    )


def save_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    validation_df: pd.DataFrame | None = None,
):
    ensure_dir(output_dir)
    write_csv(train_df, output_dir / "train_features.csv")
    if validation_df is not None:
        write_csv(validation_df, output_dir / "validation_features.csv")
    write_csv(test_df, output_dir / "test_features.csv")
