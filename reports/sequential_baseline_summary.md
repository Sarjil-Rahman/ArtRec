# Sequential Recommendation Baseline

The baselines use only interactions at or before the time-based training cutoff. They compare global popularity with a recency-aware style and artist preference variant.

| baseline | sessions | recall_at_k | ndcg_at_k | hit_rate_at_k | coverage | diversity | train_cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| recent_style_popularity | 106 | 0.2741 | 0.1566 | 0.4434 | 0.0806 | 0.8781 | 2026-01-09 08:27:00 |
| global_popularity | 106 | 0.2505 | 0.1543 | 0.4151 | 0.0806 | 0.8800 | 2026-01-09 08:27:00 |
