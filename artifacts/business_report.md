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
- ROC AUC: **0.6299**
- Average precision: **0.0615**

## Recommender ranking metrics
- Precision At 5: **0.0581**
- Recall At 5: **0.2406**
- Hit Rate At 5: **0.2742**
- Ndcg At 5: **0.1551**
- Map At 5: **0.1380**
- Precision At 10: **0.0403**
- Recall At 10: **0.3266**
- Hit Rate At 10: **0.3710**
- Ndcg At 10: **0.1853**
- Map At 10: **0.1506**
- Catalog Coverage: **0.7724**
- Artist Diversity Gini Simpson: **0.8560**
- Novelty Inverse Popularity: **0.8658**

## Offline proxy uplift (honest simulated estimate)
- Weighted Clicks: baseline=0.4889, reranked=0.4666, uplift=-4.55%
- Weighted Carts: baseline=0.0674, reranked=0.0753, uplift=11.72%
- Weighted Purchases: baseline=0.0387, reranked=0.0382, uplift=-1.25%
- Weighted Margin: baseline=2.4798, reranked=2.4538, uplift=-1.05%

## Evidence status
- Offline proxy only: metrics come from synthetic held-out data.
- Not counterfactual: logged slate reranking can still be affected by exposure and selection bias.
- Not A/B tested: no live users, production traffic, or commercial uplift proof is included.

## Notes on interpretation
- All metrics in this report come from synthetic data.
- Use the figures to judge the pipeline, evaluation workflow, and trade-off analysis, not real commercial impact.
- The recommendation layer is personalised rather than popularity-only.
- The re-ranking layer adds business logic: freshness, diversity, and margin-awareness.
- The uplift figures above are offline proxy estimates from held-out simulated sessions, not live experimental proof.
- The report is kept even when the reranker underperforms the logged baseline. In that case, the result points to weight tuning, ablation testing, or replacing the heuristic layer with a learned ranking policy.
