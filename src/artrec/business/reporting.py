from __future__ import annotations
from pathlib import Path
import pandas as pd
from artrec.utils.common import ensure_dir


def _format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def summarise_business_metrics(
    impressions_df: pd.DataFrame, conversions_df: pd.DataFrame
) -> dict:
    impressions = float(len(impressions_df))
    clicks = float(impressions_df["clicked"].sum())
    saves = float(impressions_df["saved"].sum())
    carts = float(impressions_df["add_to_cart"].sum())
    immediate_purchases = float(impressions_df["purchased_immediate"].sum())
    total_purchases = float(len(conversions_df))
    delayed_purchases = (
        float(conversions_df["delayed"].sum()) if not conversions_df.empty else 0.0
    )
    total_revenue = (
        float(conversions_df["revenue"].sum()) if not conversions_df.empty else 0.0
    )
    total_margin = (
        float(conversions_df["margin"].sum()) if not conversions_df.empty else 0.0
    )
    return {
        "rows": int(impressions),
        "sessions": int(impressions_df["session_id"].nunique()),
        "unique_users": int(impressions_df["user_id"].nunique()),
        "unique_items_served": int(impressions_df["item_id"].nunique()),
        "ctr": clicks / impressions if impressions else 0.0,
        "save_rate": saves / impressions if impressions else 0.0,
        "cart_rate": carts / impressions if impressions else 0.0,
        "immediate_purchase_rate": (
            immediate_purchases / impressions if impressions else 0.0
        ),
        "total_purchases": total_purchases,
        "delayed_purchase_share": (
            delayed_purchases / total_purchases if total_purchases else 0.0
        ),
        "revenue": total_revenue,
        "margin": total_margin,
        "revenue_per_session": total_revenue
        / max(1, impressions_df["session_id"].nunique()),
        "margin_per_session": total_margin
        / max(1, impressions_df["session_id"].nunique()),
    }


def write_business_report(
    business_metrics: dict, model_metrics: dict, output_path: Path
) -> Path:
    ensure_dir(output_path.parent)
    uplift = model_metrics.get("proxy_uplift", {})
    lines = [
        "# Business report",
        "",
        "## Executive summary",
        f"- Sessions simulated: **{business_metrics['sessions']:,}**",
        f"- Unique users: **{business_metrics['unique_users']:,}**",
        f"- Unique artworks served: **{business_metrics['unique_items_served']:,}**",
        f"- CTR: **{business_metrics['ctr']:.2%}**",
        f"- Cart rate: **{business_metrics['cart_rate']:.2%}**",
        f"- Total purchases: **{business_metrics['total_purchases']:.0f}**",
        f"- Revenue: **GBP {business_metrics['revenue']:,.2f}**",
        f"- Margin: **GBP {business_metrics['margin']:,.2f}**",
        "",
        "## Model quality",
        f"- ROC AUC: **{_format_metric(model_metrics.get('roc_auc'))}**",
        f"- Average precision: **{_format_metric(model_metrics.get('average_precision'))}**",
        "",
        "## Recommender ranking metrics",
    ]
    ranking = model_metrics.get("ranking", {})
    for key in [
        "precision_at_5",
        "recall_at_5",
        "hit_rate_at_5",
        "ndcg_at_5",
        "map_at_5",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10",
        "catalog_coverage",
        "artist_diversity_gini_simpson",
        "novelty_inverse_popularity",
    ]:
        if key in ranking:
            lines.append(f"- {key.replace('_', ' ').title()}: **{ranking[key]:.4f}**")
    lines += [
        "",
        "## Offline logged-slate replay diagnostics",
    ]
    for key, payload in uplift.items():
        pct = payload.get("uplift_pct")
        pct_str = "n/a" if pct is None else f"{pct:.2f}%"
        lines.append(
            f"- {key.replace('_',' ').title()}: baseline={payload['baseline']:.4f}, reranked={payload['reranked']:.4f}, uplift={pct_str}"
        )
    lines += [
        "",
        "## Evidence status",
        "- Offline proxy only: metrics come from synthetic held-out data.",
        "- Not counterfactual: logged slate reranking can still be affected by exposure and selection bias.",
        "- Not A/B tested: no live users, production traffic, or commercial uplift proof is included.",
        f"- Replay note: {model_metrics.get('proxy_uplift_notes', 'not provided')}",
        "",
        "## Notes on interpretation",
        "- All metrics in this report come from synthetic data.",
        "- Use the figures to judge the pipeline, evaluation workflow, and trade-off analysis, not real commercial impact.",
        "- The recommendation layer is personalised rather than popularity-only.",
        "- The re-ranking layer adds business logic: freshness, diversity, and margin-awareness.",
        "- The replay figures above are offline diagnostics from held-out simulated sessions, not live experimental proof.",
        "- The report is kept even when the reranker underperforms the logged baseline. In that case, the result points to weight tuning, ablation testing, or replacing the heuristic layer with a learned ranking policy.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
