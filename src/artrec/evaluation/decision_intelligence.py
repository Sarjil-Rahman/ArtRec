from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from artrec.data.io import read_csv_with_types, write_csv
from artrec.models.ranker import ArtRanker
from artrec.rerank.business import apply_business_rerank


EVENT_COLUMNS = ["clicked", "saved", "add_to_cart", "purchased_immediate"]
DEFAULT_TOP_K = 10


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows generated._"
    shown = df.copy()
    for col in shown.select_dtypes(include=["float"]).columns:
        shown[col] = shown[col].map(lambda value: f"{value:.4f}")
    headers = [str(col) for col in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in shown.columns) + " |")
    return "\n".join(lines)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _safe_lift(treatment: float, control: float) -> float:
    return float((treatment - control) / control * 100.0) if control else 0.0


def _gini_simpson(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    shares = values.astype(str).value_counts(normalize=True)
    return float(1.0 - np.square(shares).sum())


def add_model_scores(frame: pd.DataFrame, ranker_path: Path | None = None) -> pd.DataFrame:
    work = frame.copy()
    if "model_score" in work.columns:
        return work
    if ranker_path and ranker_path.exists():
        ranker = ArtRanker.load(ranker_path)
        work["model_score"] = ranker.score(work)
    else:
        work["model_score"] = work.get(
            "retrieval_similarity_pre", work.get("logging_policy_score", 0.0)
        )
    return work


def _rank_session(session: pd.DataFrame, strategy: str, top_k: int) -> pd.DataFrame:
    if strategy == "popularity":
        ranked = session.sort_values(
            ["popularity_score_pre", "logging_policy_score"], ascending=False
        )
    elif strategy == "model":
        ranked = session.sort_values("model_score", ascending=False)
    elif strategy == "business":
        prepared = session.copy()
        prepared["price_above_budget"] = prepared.get("price_above_budget", 0)
        prepared["margin_to_price"] = prepared.get(
            "margin_to_price", prepared["margin"] / prepared["price"].clip(lower=1.0)
        )
        ranked = apply_business_rerank(prepared, score_col="model_score", top_k=len(prepared))
    elif strategy == "diversity":
        ranked = _diversity_rank(session)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    out = ranked.head(top_k).copy()
    out["strategy"] = strategy
    out["sim_rank"] = range(1, len(out) + 1)
    return out


def _diversity_rank(session: pd.DataFrame) -> pd.DataFrame:
    remaining = session.sort_values("model_score", ascending=False).copy()
    selected = []
    seen_artists: set[str] = set()
    seen_styles: set[str] = set()
    while not remaining.empty:
        scored = remaining.copy()
        scored["_diversity_score"] = scored["model_score"]
        scored["_diversity_score"] -= scored["artist_id"].astype(str).isin(seen_artists) * 0.10
        scored["_diversity_score"] -= scored["style"].astype(str).isin(seen_styles) * 0.06
        best_idx = scored["_diversity_score"].idxmax()
        row = remaining.loc[[best_idx]]
        selected.append(row)
        seen_artists.add(str(row.iloc[0]["artist_id"]))
        seen_styles.add(str(row.iloc[0]["style"]))
        remaining = remaining.drop(index=best_idx)
    return pd.concat(selected, ignore_index=True)


def build_strategy_recommendations(
    frame: pd.DataFrame,
    strategies: tuple[str, ...] = ("popularity", "model", "business", "diversity"),
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    rows = []
    for _, session in frame.groupby("session_id", sort=False):
        for strategy in strategies:
            rows.append(_rank_session(session, strategy=strategy, top_k=top_k))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_strategy_metrics(recs: pd.DataFrame, catalog_size: int | None = None) -> pd.DataFrame:
    rows = []
    total_catalog = catalog_size or int(recs["item_id"].nunique())
    for strategy, group in recs.groupby("strategy"):
        impressions = int(len(group))
        purchases = float(group["purchased_immediate"].sum())
        revenue = float((group["price"] * group["purchased_immediate"]).sum())
        profit = float((group["margin"] * group["purchased_immediate"]).sum())
        exposure = group["item_id"].astype(str).value_counts(normalize=True)
        rows.append(
            {
                "strategy": strategy,
                "impressions": impressions,
                "ctr_proxy": _safe_rate(float(group["clicked"].sum()), impressions),
                "save_rate_proxy": _safe_rate(float(group["saved"].sum()), impressions),
                "cart_rate_proxy": _safe_rate(float(group["add_to_cart"].sum()), impressions),
                "purchase_rate_proxy": _safe_rate(purchases, impressions),
                "revenue_proxy": revenue,
                "profit_proxy": profit,
                "catalog_coverage": _safe_rate(float(group["item_id"].nunique()), total_catalog),
                "artist_diversity": _gini_simpson(group["artist_id"]),
                "style_diversity": _gini_simpson(group["style"]),
                "average_recommended_price": float(group["price"].mean()) if impressions else 0.0,
                "exposure_concentration": float(np.square(exposure).sum()) if len(exposure) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def write_business_tradeoff_report(metrics: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_profit = metrics.sort_values("profit_proxy", ascending=False).iloc[0]
    best_diversity = metrics.sort_values("artist_diversity", ascending=False).iloc[0]
    text = [
        "# Business Trade-off Report",
        "",
        "This is a production-style demo report built from synthetic offline data. "
        "The revenue and profit values are proxy metrics, not real commercial impact.",
        "",
        f"- Highest profit proxy strategy: `{best_profit['strategy']}`.",
        f"- Highest artist diversity strategy: `{best_diversity['strategy']}`.",
        "- Higher profit proxy can coincide with lower catalogue coverage or diversity because "
        "reranking gives more exposure to high-margin items.",
        "- Popularity-oriented rankings are easy to explain, but they can concentrate exposure.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def assign_experiment_group(user_id: str, experiment: str = "offline_ab_v1") -> str:
    digest = sha256(f"{experiment}:{user_id}".encode("utf-8")).hexdigest()
    return "treatment" if int(digest[:8], 16) % 2 else "control"


def summarize_group_metrics(recs: pd.DataFrame, catalog_size: int | None = None) -> pd.DataFrame:
    rows = []
    total_catalog = catalog_size or int(recs["item_id"].nunique())
    for group_name, group in recs.groupby("ab_group"):
        impressions = int(len(group))
        purchases = float(group["purchased_immediate"].sum())
        rows.append(
            {
                "group": group_name,
                "users": int(group["user_id"].nunique()),
                "impressions": impressions,
                "clicks": int(group["clicked"].sum()),
                "saves": int(group["saved"].sum()),
                "carts": int(group["add_to_cart"].sum()),
                "purchases": int(group["purchased_immediate"].sum()),
                "ctr": _safe_rate(float(group["clicked"].sum()), impressions),
                "save_rate": _safe_rate(float(group["saved"].sum()), impressions),
                "cart_rate": _safe_rate(float(group["add_to_cart"].sum()), impressions),
                "purchase_rate": _safe_rate(purchases, impressions),
                "revenue_proxy": float((group["price"] * group["purchased_immediate"]).sum()),
                "profit_proxy": float((group["margin"] * group["purchased_immediate"]).sum()),
                "diversity": _gini_simpson(group["artist_id"]),
                "catalog_coverage": _safe_rate(float(group["item_id"].nunique()), total_catalog),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def run_ab_simulation(
    frame: pd.DataFrame,
    control_strategy: str = "popularity",
    treatment_strategy: str = "business",
    top_k: int = DEFAULT_TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    control = build_strategy_recommendations(frame, (control_strategy,), top_k)
    treatment = build_strategy_recommendations(frame, (treatment_strategy,), top_k)
    control["ab_group"] = "control"
    treatment["ab_group"] = "treatment"
    assigned = pd.concat([control, treatment], ignore_index=True)
    assigned["assigned_group"] = assigned["user_id"].astype(str).map(assign_experiment_group)
    assigned = assigned[assigned["ab_group"] == assigned["assigned_group"]].copy()
    metrics = summarize_group_metrics(assigned, catalog_size=int(frame["item_id"].nunique()))
    metrics = append_lift_columns(metrics)
    return assigned, metrics


LIFT_METRICS = [
    "ctr",
    "save_rate",
    "cart_rate",
    "purchase_rate",
    "revenue_proxy",
    "profit_proxy",
    "diversity",
    "catalog_coverage",
]


def append_lift_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    work = metrics.copy()
    if set(work["group"]) >= {"control", "treatment"}:
        control = work.set_index("group").loc["control"]
        treatment = work.set_index("group").loc["treatment"]
        for metric in LIFT_METRICS:
            work[f"{metric}_lift_pct"] = _safe_lift(float(treatment[metric]), float(control[metric]))
    return work


def summarize_lift_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if not set(metrics["group"]) >= {"control", "treatment"}:
        return pd.DataFrame()
    by_group = metrics.set_index("group")
    rows = []
    for metric in LIFT_METRICS:
        control = float(by_group.loc["control", metric])
        treatment = float(by_group.loc["treatment", metric])
        rows.append(
            {
                "metric": metric,
                "control": control,
                "treatment": treatment,
                "absolute_delta": treatment - control,
                "lift_pct": _safe_lift(treatment, control),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_lift_ci(
    assigned: pd.DataFrame,
    metric: str = "profit_proxy",
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    user_metrics = _user_metric_table(assigned, metric)
    if user_metrics.empty:
        return (0.0, 0.0)
    users = user_metrics.index.to_numpy()
    lifts = []
    for _ in range(n_boot):
        sample = rng.choice(users, size=len(users), replace=True)
        sampled = user_metrics.loc[sample]
        control = float(sampled["control"].mean())
        treatment = float(sampled["treatment"].mean())
        lifts.append(_safe_lift(treatment, control))
    return (float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5)))


def _user_metric_table(assigned: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = assigned.copy()
    if metric == "profit_proxy":
        work["_metric"] = work["margin"] * work["purchased_immediate"]
    elif metric == "revenue_proxy":
        work["_metric"] = work["price"] * work["purchased_immediate"]
    else:
        work["_metric"] = work[metric]
    return work.pivot_table(index="user_id", columns="ab_group", values="_metric", aggfunc="mean").fillna(0.0)


def write_ab_report(metrics: pd.DataFrame, ci: tuple[float, float], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_columns = [
        "group",
        "users",
        "impressions",
        "clicks",
        "saves",
        "carts",
        "purchases",
        "ctr",
        "save_rate",
        "cart_rate",
        "purchase_rate",
        "revenue_proxy",
        "profit_proxy",
        "diversity",
        "catalog_coverage",
    ]
    group_metrics = metrics[[col for col in group_columns if col in metrics.columns]]
    lift_metrics = summarize_lift_metrics(metrics)
    text = [
        "# Offline Logged-Policy Replay",
        "",
        "This is an offline synthetic replay diagnostic using deterministic user assignment. "
        "It reuses logged outcomes from already displayed slates, so it is not a live A/B test, "
        "not a counterfactual estimator, and not proof of causal or commercial uplift.",
        "",
        f"- Bootstrap 95% CI for profit proxy lift: {ci[0]:.2f}% to {ci[1]:.2f}%.",
        "",
        "## Group Metrics",
        "",
        _markdown_table(group_metrics),
        "",
        "## Lift Summary",
        "",
        _markdown_table(lift_metrics),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def build_user_segments(
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    users: pd.DataFrame,
    cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    merged = interactions.copy()
    if cutoff is not None and "timestamp" in merged.columns:
        merged = merged[pd.to_datetime(merged["timestamp"]) <= pd.to_datetime(cutoff)].copy()
    if "medium" not in merged.columns:
        merged = merged.merge(
            catalog[["item_id", "medium"]],
            on="item_id",
            how="left",
        )
    merged["medium"] = merged.get("medium", "unknown").fillna("unknown")
    grouped = merged.groupby("user_id")
    features = grouped.agg(
        impressions=("item_id", "size"),
        clicks=("clicked", "sum"),
        saves=("saved", "sum"),
        carts=("add_to_cart", "sum"),
        purchases=("purchased_immediate", "sum"),
        average_viewed_price=("price", "mean"),
        style_diversity=("style", pd.Series.nunique),
        artist_diversity=("artist_id", pd.Series.nunique),
        last_seen=("timestamp", "max"),
    ).reset_index()
    for col, event in {
        "click_rate": "clicks",
        "save_rate": "saves",
        "cart_rate": "carts",
        "purchase_rate": "purchases",
    }.items():
        features[col] = features[event] / features["impressions"].clip(lower=1)
    purchases = merged[merged["purchased_immediate"] == 1]
    avg_purchase = purchases.groupby("user_id")["price"].mean().rename("average_purchased_price_proxy")
    features = features.merge(avg_purchase, on="user_id", how="left")
    features["average_purchased_price_proxy"] = features["average_purchased_price_proxy"].fillna(0.0)
    features["preferred_style"] = grouped["style"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown").values
    features["preferred_medium"] = grouped["medium"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown").values
    features = features.merge(users[["user_id", "budget_mean"]], on="user_id", how="left")
    features["monetary_proxy"] = features["average_purchased_price_proxy"] * features["purchases"]
    features["frequency_proxy"] = features["impressions"]
    features["segment_label"] = features.apply(_assign_segment, axis=1)
    return features


def _assign_segment(row: pd.Series) -> str:
    if row["purchase_rate"] >= 0.04 or row["cart_rate"] >= 0.08:
        return "high-intent collectors"
    if row["average_viewed_price"] >= max(500.0, row.get("budget_mean", 0.0) * 1.1):
        return "premium buyers"
    if row["average_viewed_price"] <= row.get("budget_mean", 0.0) * 0.75:
        return "budget-conscious buyers"
    if row["style_diversity"] >= 5:
        return "exploratory users"
    if row["style_diversity"] <= 2 and row["impressions"] >= 5:
        return "style-loyal users"
    return "casual browsers"


def write_user_segments_summary(segments: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = segments.groupby("segment_label").agg(users=("user_id", "nunique"), click_rate=("click_rate", "mean"), purchase_rate=("purchase_rate", "mean")).reset_index()
    text = [
        "# User Segments Summary",
        "",
        "Segments are explainable rules over synthetic user interaction features. "
        "When used for uplift cuts, they are built from the pre-period only to avoid future-label leakage. "
        "They are descriptive labels for this demo, not production customer personas.",
        "",
        _markdown_table(summary),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def build_uplift_by_segment(assigned: pd.DataFrame, user_segments: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    artist_counts = catalog["artist_id"].value_counts()
    artist_tier = artist_counts.rank(method="first", ascending=False, pct=True)
    catalog_tiers = catalog[["item_id", "artist_id", "style"]].copy()
    catalog_tiers["artist_popularity_tier"] = catalog_tiers["artist_id"].map(
        lambda artist: "head" if artist_tier.get(artist, 1.0) <= 0.25 else "mid_tail"
    )
    work = assigned.merge(user_segments[["user_id", "segment_label"]], on="user_id", how="left")
    work = work.merge(catalog_tiers[["item_id", "artist_popularity_tier"]], on="item_id", how="left")
    work["price_band"] = pd.cut(work["price"], bins=[0, 150, 400, np.inf], labels=["budget", "mid", "premium"], include_lowest=True)
    pieces = []
    for dimension in ["segment_label", "price_band", "style", "artist_popularity_tier"]:
        pieces.append(_uplift_for_dimension(work, dimension))
    return pd.concat(pieces, ignore_index=True)


def _uplift_for_dimension(work: pd.DataFrame, dimension: str) -> pd.DataFrame:
    rows = []
    for value, group in work.groupby(dimension, dropna=False, observed=False):
        metric = summarize_group_metrics(group, catalog_size=int(work["item_id"].nunique()))
        if set(metric["group"]) >= {"control", "treatment"}:
            by_group = metric.set_index("group")
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "control_purchase_rate": by_group.loc["control", "purchase_rate"],
                    "treatment_purchase_rate": by_group.loc["treatment", "purchase_rate"],
                    "purchase_rate_lift_pct": _safe_lift(by_group.loc["treatment", "purchase_rate"], by_group.loc["control", "purchase_rate"]),
                    "control_profit_proxy": by_group.loc["control", "profit_proxy"],
                    "treatment_profit_proxy": by_group.loc["treatment", "profit_proxy"],
                    "profit_lift_pct": _safe_lift(by_group.loc["treatment", "profit_proxy"], by_group.loc["control", "profit_proxy"]),
                }
            )
    return pd.DataFrame(rows)


def write_uplift_summary(uplift: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = uplift.sort_values("profit_lift_pct", ascending=False).head(8)
    text = [
        "# Simulated Uplift Summary",
        "",
        "This report shows offline treatment-control replay diagnostics from the synthetic simulator. "
        "Segments are pre-period labels, and the result is not causal evidence or real commercial uplift.",
        "",
        _markdown_table(top),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def build_cohort_retention(interactions: pd.DataFrame) -> pd.DataFrame:
    work = interactions.copy()
    work["week"] = work["timestamp"].dt.to_period("W").apply(lambda p: p.start_time)
    first = work.groupby("user_id")["week"].min().rename("first_seen_week")
    work = work.merge(first, on="user_id", how="left")
    work["week_number"] = ((work["week"] - work["first_seen_week"]).dt.days // 7).astype(int)
    active = work.groupby(["first_seen_week", "week_number"])["user_id"].nunique().reset_index(name="active_users")
    cohort_size = active[active["week_number"] == 0][["first_seen_week", "active_users"]].rename(columns={"active_users": "cohort_users"})
    out = active.merge(cohort_size, on="first_seen_week", how="left")
    out["weekly_retention"] = out["active_users"] / out["cohort_users"].clip(lower=1)
    repeat = work.groupby(["first_seen_week", "user_id"])["week"].nunique().reset_index(name="active_weeks")
    repeat_rate = repeat.groupby("first_seen_week")["active_weeks"].apply(lambda s: float((s > 1).mean())).rename("repeat_engagement_rate")
    out = out.merge(repeat_rate, on="first_seen_week", how="left")
    purchases = work[work["purchased_immediate"] == 1].groupby(["first_seen_week", "user_id"]).size().reset_index(name="purchases")
    repeat_purchase = purchases.groupby("first_seen_week")["purchases"].apply(lambda s: float((s > 1).mean())).rename("repeat_purchase_proxy")
    return out.merge(repeat_purchase, on="first_seen_week", how="left").fillna({"repeat_purchase_proxy": 0.0})


def write_cohort_summary(retention: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Cohort Engagement Retention",
        "",
        "Weekly retention is based on synthetic interaction timestamps and first seen week. This is engagement retention, not subscription churn.",
        "",
        _markdown_table(retention.head(20)),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


@dataclass
class SequentialResult:
    recommendations: pd.DataFrame
    metrics: pd.DataFrame


def evaluate_sequential_baseline(interactions: pd.DataFrame, top_k: int = 10) -> SequentialResult:
    work = interactions.sort_values("timestamp").copy()
    cutoff = work["timestamp"].quantile(0.8)
    train = work[work["timestamp"] <= cutoff]
    test = work[work["timestamp"] > cutoff]
    popular_items = train.groupby("item_id").size().sort_values(ascending=False)
    catalog_attrs = work.drop_duplicates("item_id").set_index("item_id")
    rec_rows = []
    metric_rows = []
    for session_id, session in test.groupby("session_id", sort=False):
        user_id = str(session["user_id"].iloc[0])
        ts = session["timestamp"].min()
        history = train[(train["user_id"] == user_id) & (train["timestamp"] < ts)]
        recent = history.sort_values("timestamp").tail(5)
        relevant = set(session.loc[(session["clicked"] == 1) | (session["saved"] == 1) | (session["add_to_cart"] == 1) | (session["purchased_immediate"] == 1), "item_id"].astype(str))
        for baseline in ["recent_style_popularity", "global_popularity"]:
            scores = popular_items.astype(float).rename("score").reset_index()
            scores = scores.merge(catalog_attrs[["style", "artist_id"]], left_on="item_id", right_index=True, how="left")
            if baseline == "recent_style_popularity" and not recent.empty:
                recent = recent.copy()
                recent["_recency_weight"] = np.linspace(0.5, 1.0, len(recent))
                style_weights = recent.groupby("style")["_recency_weight"].sum()
                artist_weights = recent.groupby("artist_id")["_recency_weight"].sum()
                scores["score"] += scores["style"].map(style_weights).fillna(0.0) * 2.0
                scores["score"] += scores["artist_id"].map(artist_weights).fillna(0.0) * 1.2
            recs = scores.sort_values("score", ascending=False).head(top_k)
            rec_items = recs["item_id"].astype(str).tolist()
            hits = [1 if item in relevant else 0 for item in rec_items]
            for rank, item_id in enumerate(rec_items, start=1):
                rec_rows.append(
                    {
                        "baseline": baseline,
                        "session_id": session_id,
                        "user_id": user_id,
                        "timestamp": ts,
                        "item_id": item_id,
                        "rank": rank,
                    }
                )
            metric_rows.append(
                {
                    "baseline": baseline,
                    "session_id": session_id,
                    "recall_at_k": _safe_rate(sum(hits), len(relevant)),
                    "hit_rate_at_k": 1.0 if any(hits) else 0.0,
                    "ndcg_at_k": _ndcg(hits, min(len(relevant), top_k)),
                    "coverage": len(set(rec_items)) / max(1, work["item_id"].nunique()),
                    "diversity": _gini_simpson(recs["artist_id"]),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    summary_rows = []
    for baseline, group in metrics.groupby("baseline"):
        summary_rows.append(
            {
                "baseline": baseline,
                "sessions": int(len(group)),
                "recall_at_k": float(group["recall_at_k"].mean()) if len(group) else 0.0,
                "ndcg_at_k": float(group["ndcg_at_k"].mean()) if len(group) else 0.0,
                "hit_rate_at_k": float(group["hit_rate_at_k"].mean()) if len(group) else 0.0,
                "coverage": float(group["coverage"].mean()) if len(group) else 0.0,
                "diversity": float(group["diversity"].mean()) if len(group) else 0.0,
                "train_cutoff": str(cutoff),
            }
        )
    order = {"recent_style_popularity": 0, "global_popularity": 1}
    summary = pd.DataFrame(summary_rows)
    summary["_order"] = summary["baseline"].map(order).fillna(99)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return SequentialResult(pd.DataFrame(rec_rows), summary)


def _ndcg(hits: list[int], ideal_relevant: int) -> float:
    if not hits or ideal_relevant <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float(np.sum(np.array(hits) * discounts))
    ideal = np.ones(min(ideal_relevant, len(hits)))
    ideal_dcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def write_sequential_summary(metrics: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Sequential Recommendation Baseline",
        "",
        "The baselines use only interactions at or before the time-based training cutoff. They compare global popularity with a recency-aware style and artist preference variant.",
        "",
        _markdown_table(metrics),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def load_demo_frames(data_dir: Path, artifact_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = read_csv_with_types(data_dir / "raw" / "catalog.csv")
    users = read_csv_with_types(data_dir / "raw" / "users.csv")
    interactions = read_csv_with_types(data_dir / "raw" / "interactions.csv")
    frame_path = data_dir / "processed" / "test_features.csv"
    frame = read_csv_with_types(frame_path if frame_path.exists() else data_dir / "raw" / "impressions.csv")
    ranker_path = artifact_dir / "ranker.joblib" if artifact_dir else None
    frame = add_model_scores(frame, ranker_path)
    return catalog, users, interactions, frame


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    return write_csv(df, path)
