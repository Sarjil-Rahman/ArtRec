# Business report

## Executive summary
- Sessions simulated: **620**
- Unique users: **80**
- Unique artworks served: **151**
- CTR: **8.29%**
- Cart rate: **0.98%**
- Total purchases: **62**
- Revenue: **GBP 12,523.84**
- Margin: **GBP 4,775.34**

## Model quality
- ROC AUC: **0.5680**
- Average precision: **0.0407**

## Recommender ranking metrics
- Precision At 5: **0.0419**
- Recall At 5: **0.1519**
- Hit Rate At 5: **0.1855**
- Ndcg At 5: **0.0861**
- Map At 5: **0.0736**
- Precision At 10: **0.0379**
- Recall At 10: **0.2890**
- Hit Rate At 10: **0.3387**
- Ndcg At 10: **0.1317**
- Map At 10: **0.0901**
- Catalog Coverage: **0.9172**
- Artist Diversity Gini Simpson: **0.8517**
- Novelty Inverse Popularity: **0.8625**

## Offline logged-slate replay diagnostics
- Weighted Clicks: baseline=0.4889, reranked=0.3831, uplift=-21.63%
- Weighted Carts: baseline=0.0674, reranked=0.0637, uplift=-5.50%
- Weighted Purchases: baseline=0.0387, reranked=0.0123, uplift=-68.13%
- Weighted Margin: baseline=2.4798, reranked=0.4572, uplift=-81.56%

## Evidence status
- Offline proxy only: metrics come from synthetic held-out data.
- Not counterfactual: logged slate reranking can still be affected by exposure and selection bias.
- Not A/B tested: no live users, production traffic, or commercial uplift proof is included.
- Replay note: Logged-slate replay diagnostic only. These values reuse observed labels from already displayed items and are not causal uplift estimates.

## Notes on interpretation
- All metrics in this report come from synthetic data.
- Use the figures to judge the pipeline, evaluation workflow, and trade-off analysis, not real commercial impact.
- The recommendation layer is personalised rather than popularity-only.
- The re-ranking layer adds business logic: freshness, diversity, and margin-awareness.
- The replay figures above are offline diagnostics from held-out simulated sessions, not live experimental proof.
- The report is kept even when the reranker underperforms the logged baseline. In that case, the result points to weight tuning, ablation testing, or replacing the heuristic layer with a learned ranking policy.
