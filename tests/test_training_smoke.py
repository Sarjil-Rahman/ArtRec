from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artrec.data.catalog import CatalogConfig, generate_catalog
from artrec.simulation.behavior import BehaviorSimulator, SimulationConfig
from artrec.features.build import build_training_frame
from artrec.models.ranker import ArtRanker
from artrec.models import ranker as ranker_module


def test_training_smoke():
    catalog_df = generate_catalog(
        CatalogConfig(n_items=60, n_artists=12, random_seed=2)
    )
    simulator = BehaviorSimulator(
        catalog_df, SimulationConfig(n_users=30, n_days=4, random_seed=2)
    )
    users_df, catalog_df, impressions_df, *_ = simulator.run()
    feature_df = build_training_frame(users_df, catalog_df, impressions_df)
    ranker = ArtRanker.fit(feature_df.head(500))
    scores = ranker.score(feature_df.head(10))
    assert len(scores) == 10


def test_catalog_generation_supports_more_artists_than_base_name_pairs():
    catalog_df = generate_catalog(
        CatalogConfig(n_items=160, n_artists=150, random_seed=42)
    )

    assert len(catalog_df) == 160
    artists = catalog_df[["artist_id", "artist_display_name"]].drop_duplicates()
    assert catalog_df["artist_id"].nunique() == 150
    assert artists["artist_display_name"].nunique() == len(artists)
    assert any(" Studio 2" in name for name in artists["artist_display_name"])


def test_ranker_excludes_post_event_leakage_features():
    forbidden = {
        "affinity_score",
        "dwell_seconds",
        "dwell_log1p",
        "clicked",
        "saved",
        "add_to_cart",
        "purchased_immediate",
        "not_interested",
        "position",
        "top_3_position",
        "top_5_position",
        "session_rank_inverse",
    }
    assert forbidden.isdisjoint(set(ranker_module.NUMERIC_FEATURES))
    assert "retrieval_similarity_pre" in ranker_module.NUMERIC_FEATURES


def test_negative_feedback_features_are_historical_only():
    catalog_df = generate_catalog(CatalogConfig(n_items=20, n_artists=5, random_seed=9))
    users_df = BehaviorSimulator(
        catalog_df, SimulationConfig(n_users=1, n_days=1, random_seed=9)
    ).users_df
    item = catalog_df.iloc[0]
    impressions_df = pd.DataFrame(
        [
            {
                "session_id": "s1",
                "timestamp": "2026-01-01 10:00:00",
                "user_id": "user_0000",
                "surface": "home_feed",
                "device": "web",
                "country": "GB",
                "position": 0,
                "item_id": item["item_id"],
                "artist_id": item["artist_id"],
                "style": item["style"],
                "price": item["price"],
                "margin": item["margin"],
                "freshness": item["freshness"],
                "availability": 1,
                "logging_policy_score": 1.0,
                "popularity_score_pre": 0.5,
                "affinity_score": 0.2,
                "examined": 1,
                "clicked": 0,
                "dwell_seconds": 0,
                "saved": 0,
                "add_to_cart": 0,
                "purchased_immediate": 0,
                "not_interested": 1,
                "not_interested_reason": "poor_match",
            },
            {
                "session_id": "s2",
                "timestamp": "2026-01-01 11:00:00",
                "user_id": "user_0000",
                "surface": "home_feed",
                "device": "web",
                "country": "GB",
                "position": 0,
                "item_id": item["item_id"],
                "artist_id": item["artist_id"],
                "style": item["style"],
                "price": item["price"],
                "margin": item["margin"],
                "freshness": item["freshness"],
                "availability": 1,
                "logging_policy_score": 1.0,
                "popularity_score_pre": 0.5,
                "affinity_score": 0.2,
                "examined": 1,
                "clicked": 0,
                "dwell_seconds": 0,
                "saved": 0,
                "add_to_cart": 0,
                "purchased_immediate": 0,
                "not_interested": 0,
                "not_interested_reason": "none",
            },
        ]
    )
    feature_df = build_training_frame(users_df, catalog_df, impressions_df)
    assert "user_item_not_interested_before" in feature_df.columns
    assert "user_artist_not_interested_before" in feature_df.columns
    assert "user_style_not_interested_before" in feature_df.columns
    assert feature_df.loc[0, "user_item_not_interested_before"] == 0
    assert feature_df.loc[1, "user_item_not_interested_before"] == 1


def test_historical_counts_exclude_same_timestamp_slate_rows():
    catalog_df = generate_catalog(CatalogConfig(n_items=20, n_artists=5, random_seed=10))
    users_df = BehaviorSimulator(
        catalog_df, SimulationConfig(n_users=1, n_days=1, random_seed=10)
    ).users_df
    item = catalog_df.iloc[0]
    base = {
        "timestamp": "2026-01-01 10:00:00",
        "user_id": "user_0000",
        "surface": "home_feed",
        "device": "web",
        "country": "GB",
        "item_id": item["item_id"],
        "artist_id": item["artist_id"],
        "style": item["style"],
        "price": item["price"],
        "margin": item["margin"],
        "freshness": item["freshness"],
        "availability": 1,
        "logging_policy_score": 1.0,
        "popularity_score_pre": 0.5,
        "retrieval_similarity_pre": 0.2,
        "affinity_score": 0.9,
        "examined": 1,
        "clicked": 0,
        "dwell_seconds": 0,
        "saved": 0,
        "add_to_cart": 0,
        "purchased_immediate": 0,
        "not_interested_reason": "none",
    }
    impressions_df = pd.DataFrame(
        [
            {**base, "session_id": "s1", "position": 0, "not_interested": 1},
            {**base, "session_id": "s1", "position": 1, "not_interested": 0},
            {
                **base,
                "session_id": "s2",
                "timestamp": "2026-01-01 11:00:00",
                "position": 0,
                "not_interested": 0,
            },
        ]
    )
    feature_df = build_training_frame(users_df, catalog_df, impressions_df)

    assert feature_df.loc[0, "user_item_seen_before"] == 0
    assert feature_df.loc[1, "user_item_seen_before"] == 0
    assert feature_df.loc[1, "user_item_not_interested_before"] == 0
    assert feature_df.loc[2, "user_item_seen_before"] == 2
    assert feature_df.loc[2, "user_item_not_interested_before"] == 1
