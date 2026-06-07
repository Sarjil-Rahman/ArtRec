from __future__ import annotations
from pathlib import Path
from artrec.data.catalog import CatalogConfig, generate_catalog, save_catalog
from artrec.simulation.behavior import BehaviorSimulator, SimulationConfig
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
    evaluate_intent_feedback_metrics,
    evaluate_ranker,
    save_metrics,
    write_model_comparison_report,
)
from artrec.business.reporting import summarise_business_metrics, write_business_report
from artrec.mlops.manifest import write_manifest
from artrec.serve.recommender import RecommendationService, ServiceAssets
from artrec.utils.common import dump_json
from artrec.utils.logging import get_logger

logger = get_logger(__name__)


def build_end_to_end(
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    n_items: int = 800,
    n_artists: int = 150,
    n_users: int = 500,
    n_days: int = 45,
    seed: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    artifact_dir = Path(artifact_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic catalog")
    catalog_df = generate_catalog(
        CatalogConfig(n_items=n_items, n_artists=n_artists, random_seed=seed)
    )
    save_catalog(catalog_df, raw_dir)

    logger.info("Simulating user behavior")
    simulator = BehaviorSimulator(
        catalog_df, SimulationConfig(n_users=n_users, n_days=n_days, random_seed=seed)
    )
    users_df, catalog_df, impressions_df, interactions_df, conversions_df = (
        simulator.run()
    )
    simulator.save_outputs(
        users_df, catalog_df, impressions_df, interactions_df, conversions_df, raw_dir
    )

    logger.info("Building feature set")
    feature_df = build_training_frame(users_df, catalog_df, impressions_df)
    train_core_df, validation_df, test_df = train_validation_test_split_by_time(
        feature_df, validation_ratio=0.2, test_ratio=0.2
    )
    train_df, _ = train_test_split_by_time(feature_df, test_ratio=0.2)
    save_feature_frames(train_df, test_df, processed_dir, validation_df=validation_df)

    logger.info("Training retrieval and ranking layers")
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

    logger.info("Evaluating offline")
    test_scores = ranker.score(test_df)
    model_metrics = evaluate_ranker(test_df, test_scores)
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
    eval_service = RecommendationService(
        ServiceAssets(
            catalog_df=catalog_df,
            users_df=users_df,
            impressions_df=impressions_df,
            retriever=retriever,
            ranker=ranker,
            semantic_retriever=semantic_retriever,
        )
    )
    model_metrics["intent_feedback"] = evaluate_intent_feedback_metrics(
        eval_service,
        test_df=test_df,
        catalog_df=catalog_df,
    )
    save_metrics(model_metrics, artifact_dir / "model_metrics.json")

    logger.info("Writing business metrics")
    business_metrics = summarise_business_metrics(impressions_df, conversions_df)
    dump_json(artifact_dir / "business_metrics.json", business_metrics)
    write_business_report(
        business_metrics, model_metrics, artifact_dir / "business_report.md"
    )
    write_manifest(
        artifact_dir / "manifest.json",
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        metrics_path=artifact_dir / "model_metrics.json",
        seed=seed,
    )

    return {
        "catalog_rows": len(catalog_df),
        "user_rows": len(users_df),
        "impressions_rows": len(impressions_df),
        "conversions_rows": len(conversions_df),
        "roc_auc": model_metrics.get("roc_auc"),
        "average_precision": model_metrics.get("average_precision"),
        "revenue": business_metrics.get("revenue"),
        "margin": business_metrics.get("margin"),
        "semantic_retriever_built": True,
        "intent_feedback_metrics_built": True,
        "model_comparison_built": True,
    }
