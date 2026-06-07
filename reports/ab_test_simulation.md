# Offline A/B Test Simulation

This is an offline synthetic experiment simulator using deterministic user assignment. It is not a live A/B test and does not prove causal or commercial uplift.

- Bootstrap 95% CI for profit proxy lift: -100.00% to 231.46%.

## Group Metrics

| group | users | impressions | clicks | saves | carts | purchases | ctr | save_rate | cart_rate | purchase_rate | revenue_proxy | profit_proxy | diversity | catalog_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 36 | 700 | 54 | 19 | 4 | 3 | 0.0771 | 0.0271 | 0.0057 | 0.0043 | 384.3600 | 141.3200 | 0.9497 | 0.5379 |
| treatment | 24 | 540 | 52 | 15 | 11 | 3 | 0.0963 | 0.0278 | 0.0204 | 0.0056 | 446.6100 | 163.8900 | 0.9643 | 0.6345 |

## Lift Summary

| metric | control | treatment | absolute_delta | lift_pct |
| --- | --- | --- | --- | --- |
| ctr | 0.0771 | 0.0963 | 0.0192 | 24.8285 |
| save_rate | 0.0271 | 0.0278 | 0.0006 | 2.3392 |
| cart_rate | 0.0057 | 0.0204 | 0.0147 | 256.4815 |
| purchase_rate | 0.0043 | 0.0056 | 0.0013 | 29.6296 |
| revenue_proxy | 384.3600 | 446.6100 | 62.2500 | 16.1958 |
| profit_proxy | 141.3200 | 163.8900 | 22.5700 | 15.9708 |
| diversity | 0.9497 | 0.9643 | 0.0145 | 1.5309 |
| catalog_coverage | 0.5379 | 0.6345 | 0.0966 | 17.9487 |
