# KPI dictionary

These are the main metrics used in the project.

| Metric | Meaning | Notes |
|---|---|---|
| CTR | Clicks divided by impressions | Useful for interest, but easy to over-optimise. |
| Save rate | Saves divided by impressions | Proxy for consideration. |
| Cart rate | Carts divided by impressions | Stronger intent than clicks. |
| Purchase rate | Purchases divided by impressions or sessions | Main commercial proxy in the demo. |
| Revenue proxy | Sum of synthetic item prices for purchases | Not real revenue. |
| Profit proxy | Revenue minus synthetic cost/margin assumptions | Not real profit. |
| Precision@K | Relevant items in the top K recommendations | Offline metric only. |
| Recall@K | Share of relevant items recovered in top K | Depends on the logged labels. |
| NDCG@K | Ranking quality with higher weight near the top | Useful for slate quality. |
| MAP@K | Average precision across ranked recommendations | Useful for comparing rankers. |
| Catalogue coverage | Share of catalogue receiving exposure | Helps detect over-concentration. |
| Artist diversity | Spread of exposure across artists | Important for marketplace fairness and discovery. |
| Novelty | Preference for less commonly exposed items | Can improve discovery but may reduce short-term clicks. |

All business values in this repo are synthetic proxies. They should be used for discussion and regression checks, not commercial claims.
