from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.evaluation.decision_intelligence import (
    append_lift_columns,
    assign_experiment_group,
    build_cohort_retention,
    build_strategy_recommendations,
    build_uplift_by_segment,
    build_user_segments,
    evaluate_sequential_baseline,
    run_ab_simulation,
    summarize_strategy_metrics,
    write_business_tradeoff_report,
)


def _toy_frame() -> pd.DataFrame:
    rows = []
    for user_idx, user_id in enumerate(["user_a", "user_b", "user_c", "user_d"]):
        for session_idx in range(2):
            for pos in range(4):
                item_num = (user_idx * 3 + session_idx + pos) % 8
                rows.append(
                    {
                        "session_id": f"s_{user_idx}_{session_idx}",
                        "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=session_idx + user_idx),
                        "user_id": user_id,
                        "position": pos,
                        "item_id": f"item_{item_num}",
                        "artist_id": f"artist_{item_num % 3}",
                        "style": ["abstract", "modern", "landscape"][item_num % 3],
                        "medium": ["Oil", "Print"][item_num % 2],
                        "price": 100.0 + item_num * 50.0,
                        "margin": 30.0 + item_num * 12.0,
                        "freshness": 0.2 + pos * 0.1,
                        "popularity_score_pre": 1.0 - pos * 0.1,
                        "logging_policy_score": 0.8 - pos * 0.1,
                        "model_score": 0.1 + item_num * 0.03 + (3 - pos) * 0.02,
                        "price_above_budget": int(item_num > 4),
                        "margin_to_price": (30.0 + item_num * 12.0) / (100.0 + item_num * 50.0),
                        "clicked": int(pos == 0 or item_num % 4 == 0),
                        "saved": int(pos == 1),
                        "add_to_cart": int(item_num % 5 == 0),
                        "purchased_immediate": int(item_num % 7 == 0),
                    }
                )
    return pd.DataFrame(rows)


def _toy_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["item_id", "artist_id", "style", "medium", "price"]].drop_duplicates("item_id")


def _toy_users() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_a", "user_b", "user_c", "user_d"],
            "budget_mean": [180.0, 250.0, 400.0, 600.0],
        }
    )


def test_business_metric_calculation_and_report_generation(tmp_path):
    frame = _toy_frame()
    recs = build_strategy_recommendations(frame, top_k=3)
    metrics = summarize_strategy_metrics(recs, catalog_size=8)

    assert set(metrics["strategy"]) == {"popularity", "model", "business", "diversity"}
    assert metrics["ctr_proxy"].between(0, 1).all()
    assert metrics["catalog_coverage"].between(0, 1).all()

    report_path = tmp_path / "business_tradeoff_report.md"
    write_business_tradeoff_report(metrics, report_path)
    text = report_path.read_text(encoding="utf-8")
    assert "synthetic offline data" in text
    assert "not real commercial impact" in text


def test_ab_assignment_is_deterministic_and_exclusive():
    first = [assign_experiment_group(user_id) for user_id in ["u1", "u2", "u3"]]
    second = [assign_experiment_group(user_id) for user_id in ["u1", "u2", "u3"]]
    assert first == second
    assert set(first).issubset({"control", "treatment"})

    assigned, metrics = run_ab_simulation(_toy_frame(), top_k=3)
    assert (assigned["ab_group"] == assigned["assigned_group"]).all()
    assert assigned.groupby("user_id")["ab_group"].nunique().max() == 1
    assert {"control", "treatment"}.issubset(set(metrics["group"]))


def test_uplift_calculation_is_valid():
    metrics = pd.DataFrame(
        {
            "group": ["control", "treatment"],
            "ctr": [0.10, 0.12],
            "save_rate": [0.05, 0.05],
            "cart_rate": [0.02, 0.03],
            "purchase_rate": [0.01, 0.015],
            "revenue_proxy": [100.0, 120.0],
            "profit_proxy": [40.0, 50.0],
            "diversity": [0.7, 0.8],
            "catalog_coverage": [0.3, 0.33],
        }
    )
    lifted = append_lift_columns(metrics)
    assert lifted["ctr_lift_pct"].iloc[0] == pytest.approx(20.0)
    assert lifted["profit_proxy_lift_pct"].iloc[0] == pytest.approx(25.0)


def test_user_segments_and_segment_uplift():
    frame = _toy_frame()
    segments = build_user_segments(frame, _toy_catalog(frame), _toy_users())

    assert {"click_rate", "preferred_style", "preferred_medium", "segment_label"}.issubset(segments.columns)
    assert segments["segment_label"].notna().all()

    assigned, _metrics = run_ab_simulation(frame, top_k=3)
    uplift = build_uplift_by_segment(assigned, segments, _toy_catalog(frame))
    assert {"segment_label", "price_band", "style", "artist_popularity_tier"}.issubset(set(uplift["dimension"]))
    assert "profit_lift_pct" in uplift.columns


def test_cohort_retention_matrix_logic():
    frame = _toy_frame()
    retention = build_cohort_retention(frame)

    assert {"first_seen_week", "week_number", "weekly_retention"}.issubset(retention.columns)
    assert retention["weekly_retention"].between(0, 1).all()
    assert (retention[retention["week_number"] == 0]["weekly_retention"] == 1.0).all()


def test_sequential_baseline_uses_time_based_cutoff_without_future_leakage():
    frame = _toy_frame()
    result = evaluate_sequential_baseline(frame, top_k=3)

    assert result.metrics["baseline"].iloc[0] == "recent_style_popularity"
    cutoff = pd.Timestamp(result.metrics["train_cutoff"].iloc[0])
    assert (result.recommendations["timestamp"] > cutoff).all()
    assert result.metrics["recall_at_k"].between(0, 1).all()
