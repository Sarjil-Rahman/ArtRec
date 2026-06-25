# Offline Logged-Policy Replay

This is an offline synthetic replay diagnostic using deterministic user assignment. It reuses logged outcomes from already displayed slates, so it is not a live A/B test, not a counterfactual estimator, and not proof of causal or commercial uplift.

- Bootstrap 95% CI for profit proxy lift: -100.00% to 231.46%.

## Group Metrics

| group | users | impressions | clicks | saves | carts | purchases | ctr | save_rate | cart_rate | purchase_rate | revenue_proxy | profit_proxy | diversity | catalog_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 36 | 700 | 54 | 19 | 4 | 3 | 0.0771 | 0.0271 | 0.0057 | 0.0043 | 384.3600 | 141.3200 | 0.9497 | 0.5379 |
| treatment | 24 | 540 | 48 | 15 | 10 | 3 | 0.0889 | 0.0278 | 0.0185 | 0.0056 | 446.6100 | 163.8900 | 0.9648 | 0.7034 |

## Lift Summary

| metric | control | treatment | absolute_delta | lift_pct |
| --- | --- | --- | --- | --- |
| ctr | 0.0771 | 0.0889 | 0.0117 | 15.2263 |
| save_rate | 0.0271 | 0.0278 | 0.0006 | 2.3392 |
| cart_rate | 0.0057 | 0.0185 | 0.0128 | 224.0741 |
| purchase_rate | 0.0043 | 0.0056 | 0.0013 | 29.6296 |
| revenue_proxy | 384.3600 | 446.6100 | 62.2500 | 16.1958 |
| profit_proxy | 141.3200 | 163.8900 | 22.5700 | 15.9708 |
| diversity | 0.9497 | 0.9648 | 0.0151 | 1.5929 |
| catalog_coverage | 0.5379 | 0.7034 | 0.1655 | 30.7692 |
