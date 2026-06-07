# Model Comparison

Offline comparison on synthetic time-based splits. These metrics are diagnostics for model selection, not production uplift or A/B-test proof.

## Train Split

| model | rows | positive_rate | roc_auc | average_precision | ndcg_at_5 | map_at_5 | recall_at_5 | catalog_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_ranker | 7440 | 0.0383 | 0.6950 | 0.0761 | 0.1669 | 0.1521 | 0.2419 | 0.8621 |
| lightgbm_lambdarank | 7440 | 0.0383 | 0.9495 | 0.5821 | 0.5209 | 0.5351 | 0.5206 | 0.9103 |

## Validation Split

| model | rows | positive_rate | roc_auc | average_precision | ndcg_at_5 | map_at_5 | recall_at_5 | catalog_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_ranker | 2480 | 0.0298 | 0.6484 | 0.0586 | 0.1041 | 0.0996 | 0.1458 | 0.8182 |
| lightgbm_lambdarank | 2480 | 0.0298 | 0.6233 | 0.0470 | 0.1119 | 0.0942 | 0.1882 | 0.9231 |

## Test Split

| model | rows | positive_rate | roc_auc | average_precision | ndcg_at_5 | map_at_5 | recall_at_5 | catalog_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_ranker | 2480 | 0.0319 | 0.5932 | 0.0458 | 0.1189 | 0.1120 | 0.1761 | 0.8069 |
| lightgbm_lambdarank | 2480 | 0.0319 | 0.5889 | 0.0411 | 0.0995 | 0.0810 | 0.1815 | 0.8828 |

## Interpretation

- Prefer the validation and test splits over the training split when discussing generalisation.
- Treat wins as offline evidence only; real impact still requires production logging and an online experiment.
- The LightGBM baseline is a stronger learning-to-rank comparison, but it is not promoted to the serving artifact by default.
