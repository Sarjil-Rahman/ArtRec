from __future__ import annotations
import pandas as pd


def apply_business_rerank(
    df: pd.DataFrame, score_col: str = "model_score", top_k: int = 12
) -> pd.DataFrame:
    work = df.copy()
    work["diversity_penalty"] = work.groupby("artist_id").cumcount() * 0.08
    work["business_score"] = (
        work[score_col]
        + 0.08 * work["freshness"]
        + 0.05 * (work["margin"] / work["margin"].clip(lower=1.0).max())
        - 0.06 * work["price_above_budget"]
        - work["diversity_penalty"]
    )
    if "semantic_similarity" in work.columns:
        work["business_score"] += 0.06 * work["semantic_similarity"].fillna(0).clip(
            0, 1
        )
    if "user_item_not_interested_before" in work.columns:
        work["business_score"] -= 0.20 * work["user_item_not_interested_before"].fillna(
            0
        ).clip(0, 1)
    if "user_artist_not_interested_before" in work.columns:
        work["business_score"] -= 0.08 * work[
            "user_artist_not_interested_before"
        ].fillna(0).clip(0, 3)
    if "user_style_not_interested_before" in work.columns:
        work["business_score"] -= 0.06 * work[
            "user_style_not_interested_before"
        ].fillna(0).clip(0, 3)
    reason_codes = []
    for _, row in work.iterrows():
        reasons = []
        if row.get("retrieval_similarity", 0) >= 0.75:
            reasons.append("high_similarity")
        if row.get("semantic_similarity", 0) >= 0.20:
            reasons.append("semantic_match")
        if row["freshness"] >= 0.70:
            reasons.append("fresh_item")
        if row["margin_to_price"] >= work["margin_to_price"].median():
            reasons.append("strong_margin")
        if row["price_above_budget"] == 0:
            reasons.append("price_fit")
        if not reasons:
            reasons.append("general_match")
        reason_codes.append("|".join(reasons))
    work["reason_codes"] = reason_codes
    work = (
        work.sort_values("business_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    work["final_rank"] = range(1, len(work) + 1)
    return work
