from __future__ import annotations
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from artrec.data.io import read_csv_with_types
from artrec.features.build import (
    build_training_frame,
    train_test_split_by_time,
    train_validation_test_split_by_time,
    save_feature_frames,
)
from artrec.models.ranker import ArtRanker, LightGBMLearningToRanker
from artrec.retrieval.index import CatalogRetriever
from artrec.retrieval.semantic import SemanticCatalogRetriever
from artrec.evaluation.offline import (
    compare_rankers_on_splits,
    evaluate_ranker,
    save_metrics,
    write_model_comparison_report,
)
from artrec.utils.common import dump_json
from artrec.mlops.manifest import write_manifest
from artrec.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    data_dir, artifact_dir = Path(args.data_dir), Path(args.artifact_dir)
    raw_dir, processed_dir = data_dir / "raw", data_dir / "processed"
    users_df = read_csv_with_types(raw_dir / "users.csv")
    catalog_df = read_csv_with_types(raw_dir / "catalog.csv")
    impressions_df = read_csv_with_types(raw_dir / "impressions.csv")
    feature_df = build_training_frame(users_df, catalog_df, impressions_df)
    train_core_df, validation_df, test_df = train_validation_test_split_by_time(
        feature_df, validation_ratio=0.2, test_ratio=0.2
    )
    train_df, _ = train_test_split_by_time(feature_df, test_ratio=0.2)
    save_feature_frames(train_df, test_df, processed_dir, validation_df=validation_df)
    ranker = ArtRanker.fit(train_df)
    ranker.save(artifact_dir / "ranker.joblib")
    logistic_baseline = ArtRanker.fit(train_core_df)
    lightgbm_baseline = LightGBMLearningToRanker.fit(
        train_core_df, validation_df=validation_df
    )
    retriever = CatalogRetriever.fit(catalog_df, n_neighbors=50)
    retriever.save(artifact_dir / "retriever.joblib")
    semantic_retriever = SemanticCatalogRetriever.fit(catalog_df)
    semantic_retriever.save(artifact_dir / "semantic_retriever.joblib")
    metrics = evaluate_ranker(test_df, ranker.score(test_df))
    save_metrics(metrics, artifact_dir / "model_metrics.json")
    model_comparison = compare_rankers_on_splits(
        {
            "train": train_core_df,
            "validation": validation_df,
            "test": test_df,
        },
        {
            "logistic_ranker": logistic_baseline,
            "lightgbm_lambdarank": lightgbm_baseline,
        },
    )
    dump_json(artifact_dir / "model_comparison.json", model_comparison)
    write_model_comparison_report(
        model_comparison, artifact_dir / "model_comparison.md"
    )
    write_manifest(
        artifact_dir / "manifest.json",
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        metrics_path=artifact_dir / "model_metrics.json",
        seed=42,
    )
    logger.info("Saved ranker, retrievers, and metrics")


if __name__ == "__main__":
    main()
