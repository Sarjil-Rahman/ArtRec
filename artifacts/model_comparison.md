# Model Comparison

Offline comparison on synthetic time-based splits. These metrics are diagnostics for model selection, not production uplift or A/B-test proof.

## Train Split

| model | rows | positive_rate | roc_auc | average_precision | ndcg_at_5 | map_at_5 | recall_at_5 | catalog_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_ranker | 7440 | 0.0383 | 0.6652 | 0.0648 | 0.1380 | 0.1220 | 0.2151 | 0.9517 |
| lightgbm_lambdarank | 7440 | 0.0383 | 0.9353 | 0.4905 | 0.5124 | 0.5245 | 0.5181 | 0.9517 |

## Validation Split

| model | rows | positive_rate | roc_auc | average_precision | ndcg_at_5 | map_at_5 | recall_at_5 | catalog_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_ranker | 2480 | 0.0298 | 0.5866 | 0.0405 | 0.0803 | 0.0794 | 0.1163 | 0.9021 |
| lightgbm_lambdarank | 2480 | 0.0298 | 0.5619 | 0.0391 | 0.0958 | 0.0899 | 0.1358 | 0.8881 |

## Test Split

| model | rows | positive_rate | roc_auc | average_precision | ndcg_at_5 | map_at_5 | recall_at_5 | catalog_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_ranker | 2480 | 0.0319 | 0.5680 | 0.0407 | 0.0861 | 0.0736 | 0.1519 | 0.9172 |
| lightgbm_lambdarank | 2480 | 0.0319 | 0.5500 | 0.0387 | 0.0804 | 0.0801 | 0.1277 | 0.9034 |

## Interpretation

- Prefer the validation and test splits over the training split when discussing generalisation.
- Treat wins as offline evidence only; real impact still requires production logging and an online experiment.
- The LightGBM baseline is a stronger learning-to-rank comparison, but it is not promoted to the serving artifact by default.
