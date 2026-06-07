from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from artrec.models.ranker import make_ranking_label
from artrec.utils.common import dump_json

POSITION_WEIGHTS = {
    0: 1.00,
    1: 0.88,
    2: 0.79,
    3: 0.71,
    4: 0.64,
    5: 0.58,
    6: 0.53,
    7: 0.49,
    8: 0.45,
    9: 0.42,
}
DEFAULT_K_VALUES = (5, 10, 12)


def _attention_weight(pos: int) -> float:
    return POSITION_WEIGHTS.get(int(pos), 0.35)


def _dcg(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
    return float(np.sum(labels * discounts))


def _average_precision_at_k(labels: np.ndarray, k: int) -> float:
    top = labels[:k]
    relevant = int(top.sum())
    if relevant == 0:
        return 0.0
    precisions = []
    hits = 0
    for idx, label in enumerate(top, start=1):
        if label:
            hits += 1
            precisions.append(hits / idx)
    return float(np.mean(precisions)) if precisions else 0.0


def _ranking_metrics(
    eval_df: pd.DataFrame, y_binary: pd.Series, k_values=DEFAULT_K_VALUES
) -> dict:
    work = eval_df.copy()
    work["is_relevant"] = y_binary.astype(int).values
    out = {}
    all_recommended = set()
    diversity_rows = []
    novelty_rows = []
    total_items = int(work["item_id"].nunique()) if "item_id" in work.columns else 0

    for k in k_values:
        precision, recall, hit_rate, ndcg, map_scores = [], [], [], [], []
        recommended_at_k = set()
        for _, sess in work.groupby("session_id"):
            ranked = sess.sort_values("model_score", ascending=False)
            labels = ranked["is_relevant"].to_numpy(dtype=int)
            top = ranked.head(k)
            top_labels = top["is_relevant"].to_numpy(dtype=int)
            relevant_total = int(labels.sum())
            hits = int(top_labels.sum())
            precision.append(hits / max(1, min(k, len(top))))
            recall.append(hits / relevant_total if relevant_total else 0.0)
            hit_rate.append(1.0 if hits > 0 else 0.0)
            ideal = np.sort(labels)[::-1][:k]
            ideal_dcg = _dcg(ideal)
            ndcg.append(_dcg(top_labels) / ideal_dcg if ideal_dcg > 0 else 0.0)
            map_scores.append(_average_precision_at_k(labels, k))
            recommended_at_k.update(top["item_id"].tolist())
            if "artist_id" in top.columns and len(top) > 1:
                diversity_rows.append(
                    1.0 - (top["artist_id"].value_counts(normalize=True) ** 2).sum()
                )
            if "popularity_score_pre" in top.columns and len(top) > 0:
                novelty_rows.append(
                    float((1.0 - top["popularity_score_pre"].clip(0, 1)).mean())
                )
        all_recommended.update(recommended_at_k)
        out[f"precision_at_{k}"] = float(np.mean(precision)) if precision else 0.0
        out[f"recall_at_{k}"] = float(np.mean(recall)) if recall else 0.0
        out[f"hit_rate_at_{k}"] = float(np.mean(hit_rate)) if hit_rate else 0.0
        out[f"ndcg_at_{k}"] = float(np.mean(ndcg)) if ndcg else 0.0
        out[f"map_at_{k}"] = float(np.mean(map_scores)) if map_scores else 0.0
        out[f"catalog_coverage_at_{k}"] = (
            len(recommended_at_k) / total_items if total_items else 0.0
        )

    out["catalog_coverage"] = len(all_recommended) / total_items if total_items else 0.0
    out["artist_diversity_gini_simpson"] = (
        float(np.mean(diversity_rows)) if diversity_rows else 0.0
    )
    out["novelty_inverse_popularity"] = (
        float(np.mean(novelty_rows)) if novelty_rows else 0.0
    )
    return out


def evaluate_ranker(test_df: pd.DataFrame, scores: np.ndarray) -> dict:
    y_binary = (make_ranking_label(test_df) >= 1.5).astype(int)
    metrics = {
        "rows": int(len(test_df)),
        "positive_rate": float(y_binary.mean()),
        "roc_auc": (
            float(roc_auc_score(y_binary, scores)) if y_binary.nunique() > 1 else None
        ),
        "average_precision": (
            float(average_precision_score(y_binary, scores))
            if y_binary.nunique() > 1
            else None
        ),
    }
    eval_df = test_df.copy()
    eval_df["model_score"] = scores
    eval_df["attention_weight"] = eval_df["position"].apply(_attention_weight)
    metrics["ranking"] = _ranking_metrics(eval_df, y_binary)
    baseline_rows, reranked_rows = [], []
    for _, sess in eval_df.groupby("session_id"):
        baseline = sess.sort_values("position").head(5)
        reranked = sess.sort_values("model_score", ascending=False).head(5)
        baseline_rows.append(
            {
                "weighted_clicks": float(
                    (baseline["clicked"] * baseline["attention_weight"]).sum()
                ),
                "weighted_carts": float(
                    (baseline["add_to_cart"] * baseline["attention_weight"]).sum()
                ),
                "weighted_purchases": float(
                    (
                        baseline["purchased_immediate"] * baseline["attention_weight"]
                    ).sum()
                ),
                "weighted_margin": float(
                    (
                        baseline["margin"]
                        * baseline["purchased_immediate"]
                        * baseline["attention_weight"]
                    ).sum()
                ),
            }
        )
        reranked_rows.append(
            {
                "weighted_clicks": float(
                    (reranked["clicked"] * reranked["attention_weight"]).sum()
                ),
                "weighted_carts": float(
                    (reranked["add_to_cart"] * reranked["attention_weight"]).sum()
                ),
                "weighted_purchases": float(
                    (
                        reranked["purchased_immediate"] * reranked["attention_weight"]
                    ).sum()
                ),
                "weighted_margin": float(
                    (
                        reranked["margin"]
                        * reranked["purchased_immediate"]
                        * reranked["attention_weight"]
                    ).sum()
                ),
            }
        )
    base_df, rer_df = pd.DataFrame(baseline_rows), pd.DataFrame(reranked_rows)

    def uplift(metric: str) -> dict:
        base = float(base_df[metric].mean())
        new = float(rer_df[metric].mean())
        pct = ((new - base) / base * 100.0) if base > 0 else None
        return {"baseline": base, "reranked": new, "uplift_pct": pct}

    metrics["proxy_uplift"] = {
        "weighted_clicks": uplift("weighted_clicks"),
        "weighted_carts": uplift("weighted_carts"),
        "weighted_purchases": uplift("weighted_purchases"),
        "weighted_margin": uplift("weighted_margin"),
    }
    return metrics


def compact_ranker_metrics(eval_df: pd.DataFrame, scores: np.ndarray) -> dict:
    metrics = evaluate_ranker(eval_df, scores)
    ranking = metrics["ranking"]
    return {
        "rows": metrics["rows"],
        "positive_rate": metrics["positive_rate"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "ndcg_at_5": ranking["ndcg_at_5"],
        "map_at_5": ranking["map_at_5"],
        "recall_at_5": ranking["recall_at_5"],
        "catalog_coverage": ranking["catalog_coverage"],
    }


def compare_rankers_on_splits(
    split_frames: dict[str, pd.DataFrame],
    rankers: dict[str, object],
) -> dict:
    comparison = {
        "notes": (
            "Offline comparison on synthetic time-based splits. These metrics are "
            "diagnostics for model selection, not production uplift or A/B-test proof."
        ),
        "splits": {},
    }
    for split_name, frame in split_frames.items():
        comparison["splits"][split_name] = {}
        for model_name, ranker in rankers.items():
            comparison["splits"][split_name][model_name] = compact_ranker_metrics(
                frame, ranker.score(frame)
            )
    return comparison


def write_model_comparison_report(comparison: dict, output_path: Path) -> Path:
    metric_keys = [
        "rows",
        "positive_rate",
        "roc_auc",
        "average_precision",
        "ndcg_at_5",
        "map_at_5",
        "recall_at_5",
        "catalog_coverage",
    ]
    lines = [
        "# Model Comparison",
        "",
        comparison["notes"],
        "",
    ]
    for split_name, models in comparison["splits"].items():
        lines.extend(
            [
                f"## {split_name.title()} Split",
                "",
                "| model | " + " | ".join(metric_keys) + " |",
                "|---|" + "|".join(["---:"] * len(metric_keys)) + "|",
            ]
        )
        for model_name, metrics in models.items():
            values = []
            for key in metric_keys:
                value = metrics[key]
                if value is None:
                    values.append("n/a")
                elif isinstance(value, int):
                    values.append(str(value))
                else:
                    values.append(f"{float(value):.4f}")
            lines.append(f"| {model_name} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Prefer the validation and test splits over the training split when discussing generalisation.",
            "- Treat wins as offline evidence only; real impact still requires production logging and an online experiment.",
            "- The LightGBM baseline is a stronger learning-to-rank comparison, but it is not promoted to the serving artifact by default.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _query_from_item(item: pd.Series) -> str:
    parts = [
        item.get("dominant_hue"),
        item.get("style"),
        item.get("object_name", "artwork"),
    ]
    return " ".join(str(part) for part in parts if pd.notna(part) and str(part))


def _semantic_relevance(results: pd.DataFrame, source_item: pd.Series) -> pd.Series:
    style = str(source_item.get("style", "")).lower()
    hue = str(source_item.get("dominant_hue", "")).lower()
    same_style = results["style"].fillna("").astype(str).str.lower() == style
    hue_match = results["dominant_hue"].fillna("").astype(str).str.lower() == hue
    tag_match = results["tags"].fillna("").astype(str).str.lower().str.contains(hue)
    return (same_style & (hue_match | tag_match)).astype(int)


def _precision_at_k_from_labels(labels: pd.Series, k: int) -> float:
    if labels.empty:
        return 0.0
    top = labels.head(k)
    return float(top.sum() / max(1, len(top)))


def _ndcg_at_k_for_items(
    recommended_items: list[str], relevant_items: set[str], k: int
) -> float:
    labels = np.array(
        [1 if item_id in relevant_items else 0 for item_id in recommended_items[:k]],
        dtype=int,
    )
    if labels.size == 0:
        return 0.0
    ideal_relevant = min(len(relevant_items), k)
    if ideal_relevant == 0:
        return 0.0
    ideal = np.ones(ideal_relevant, dtype=int)
    return _dcg(labels) / _dcg(ideal)


def evaluate_intent_feedback_metrics(
    service,
    test_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    k: int = 10,
    max_cases: int = 30,
) -> dict:
    """Offline diagnostics for semantic intent and runtime feedback mechanics.

    These are synthetic, functional diagnostics. They should not be interpreted
    as production uplift or online user-impact metrics.
    """
    catalog = catalog_df.set_index("item_id", drop=False)
    semantic_precisions = []
    for _, item in catalog_df.head(max_cases).iterrows():
        query = _query_from_item(item)
        if not query.strip():
            continue
        results = service.search(query=query, limit=k, availability_only=True)
        if results.empty:
            semantic_precisions.append(0.0)
            continue
        labels = _semantic_relevance(results, item)
        semantic_precisions.append(_precision_at_k_from_labels(labels, k))

    relevance = (make_ranking_label(test_df) >= 1.5).astype(int)
    relevant_df = test_df.loc[relevance == 1].copy()
    hybrid_deltas = []
    for user_id, user_relevance in relevant_df.groupby("user_id"):
        if user_id not in service.users.index:
            continue
        relevant_items = set(user_relevance["item_id"].astype(str).tolist())
        source_item_id = next(iter(relevant_items))
        if source_item_id not in catalog.index:
            continue
        query = _query_from_item(catalog.loc[source_item_id])
        personalized = service.recommend(user_id=user_id, limit=k)
        hybrid = service.recommend(user_id=user_id, query=query, limit=k)
        personalized_ndcg = _ndcg_at_k_for_items(
            personalized["item_id"].astype(str).tolist(), relevant_items, k
        )
        hybrid_ndcg = _ndcg_at_k_for_items(
            hybrid["item_id"].astype(str).tolist(), relevant_items, k
        )
        hybrid_deltas.append(hybrid_ndcg - personalized_ndcg)
        if len(hybrid_deltas) >= max_cases:
            break

    suppression_hits = []
    rank_deltas = []
    candidate_users = service.users["user_id"].head(max_cases).tolist()
    for user_id in candidate_users:
        before = service.recommend(user_id=user_id, limit=k)
        if before.empty:
            continue
        item_id = str(before.iloc[0]["item_id"])
        before_rank = int(before.iloc[0]["final_rank"])
        service.record_not_interested(
            user_id=user_id, item_id=item_id, reason="metric_probe"
        )
        after = service.recommend(user_id=user_id, limit=k)
        after_match = after[after["item_id"].astype(str) == item_id]
        after_rank = (
            int(after_match.iloc[0]["final_rank"]) if not after_match.empty else k + 1
        )
        delta = after_rank - before_rank
        rank_deltas.append(delta)
        suppression_hits.append(1.0 if delta > 0 else 0.0)

    return {
        "semantic_search_precision_at_10": (
            float(np.mean(semantic_precisions)) if semantic_precisions else 0.0
        ),
        "hybrid_vs_personalized_ndcg_delta_at_10": (
            float(np.mean(hybrid_deltas)) if hybrid_deltas else 0.0
        ),
        "not_interested_suppression_rate_at_10": (
            float(np.mean(suppression_hits)) if suppression_hits else 0.0
        ),
        "post_feedback_rank_delta_at_10": (
            float(np.mean(rank_deltas)) if rank_deltas else 0.0
        ),
        "semantic_search_cases": int(len(semantic_precisions)),
        "hybrid_eval_cases": int(len(hybrid_deltas)),
        "feedback_eval_cases": int(len(rank_deltas)),
        "notes": (
            "Synthetic offline diagnostics only; not online uplift, not an A/B test."
        ),
    }


def save_metrics(metrics: dict, output_path: Path) -> Path:
    dump_json(output_path, metrics)
    return output_path
