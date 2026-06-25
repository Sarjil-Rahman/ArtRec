from pathlib import Path
import json
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.models.ranker import ArtRanker
from artrec.pipeline import build_end_to_end


def test_build_outputs_artifacts_and_ranking_metrics(tmp_path):
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    build_end_to_end(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        n_items=70,
        n_artists=15,
        n_users=30,
        n_days=5,
        seed=7,
    )

    expected = [
        data_dir / "raw" / "catalog.csv",
        data_dir / "raw" / "users.csv",
        data_dir / "processed" / "train_features.csv",
        data_dir / "processed" / "test_features.csv",
        artifact_dir / "ranker.joblib",
        artifact_dir / "retriever.joblib",
        artifact_dir / "semantic_retriever.joblib",
        artifact_dir / "model_metrics.json",
        artifact_dir / "model_comparison.json",
        artifact_dir / "model_comparison.md",
        artifact_dir / "business_metrics.json",
        artifact_dir / "business_report.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    train = pd.read_csv(data_dir / "processed" / "train_features.csv")
    validation = pd.read_csv(data_dir / "processed" / "validation_features.csv")
    test = pd.read_csv(data_dir / "processed" / "test_features.csv")
    key = ["session_id", "timestamp", "user_id", "position", "item_id"]
    train_keys = set(map(tuple, train[key].astype(str).values.tolist()))
    validation_keys = set(map(tuple, validation[key].astype(str).values.tolist()))
    test_keys = set(map(tuple, test[key].astype(str).values.tolist()))
    assert not train_keys & validation_keys
    assert not train_keys & test_keys
    assert not validation_keys & test_keys

    ranker = ArtRanker.load(artifact_dir / "ranker.joblib")
    assert ranker is not None

    metrics = json.loads(
        (artifact_dir / "model_metrics.json").read_text(encoding="utf-8")
    )
    ranking = metrics["ranking"]
    for key in [
        "precision_at_5",
        "recall_at_5",
        "hit_rate_at_5",
        "ndcg_at_5",
        "map_at_5",
        "catalog_coverage",
    ]:
        assert key in ranking
        assert 0.0 <= ranking[key] <= 1.0

    intent_feedback = metrics["intent_feedback"]
    for key in [
        "semantic_search_precision_at_10",
        "not_interested_suppression_rate_at_10",
    ]:
        assert key in intent_feedback
        assert 0.0 <= intent_feedback[key] <= 1.0
    for key in [
        "hybrid_vs_personalized_ndcg_delta_at_10",
        "post_feedback_rank_delta_at_10",
    ]:
        assert key in intent_feedback
        assert isinstance(intent_feedback[key], float)
    assert intent_feedback["semantic_search_cases"] > 0
    assert intent_feedback["feedback_eval_cases"] > 0

    comparison = json.loads(
        (artifact_dir / "model_comparison.json").read_text(encoding="utf-8")
    )
    assert {"train", "validation", "test"}.issubset(comparison["splits"])
    assert "lightgbm_lambdarank" in comparison["splits"]["validation"]
