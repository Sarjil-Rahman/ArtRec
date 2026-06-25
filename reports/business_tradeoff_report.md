# Business Trade-off Report

This is a production-style demo report built from synthetic offline data. The revenue and profit values are proxy metrics, not real commercial impact.

- Highest profit proxy strategy: `business`.
- Highest artist diversity strategy: `business`.
- Higher profit proxy can coincide with lower catalogue coverage or diversity because reranking gives more exposure to high-margin items.
- Popularity-oriented rankings are easy to explain, but they can concentrate exposure.

## Metrics

| strategy | impressions | ctr_proxy | save_rate_proxy | cart_rate_proxy | purchase_rate_proxy | revenue_proxy | profit_proxy | catalog_coverage | artist_diversity | style_diversity | average_recommended_price | exposure_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| business | 1240 | 0.0879 | 0.0290 | 0.0137 | 0.0065 | 1322.7300 | 460.7700 | 0.7875 | 0.9662 | 0.8488 | 187.5479 | 0.0151 |
| diversity | 1240 | 0.0968 | 0.0290 | 0.0145 | 0.0065 | 1192.8700 | 436.8100 | 0.7750 | 0.9643 | 0.8460 | 177.2996 | 0.0163 |
| model | 1240 | 0.0944 | 0.0290 | 0.0137 | 0.0065 | 1071.3500 | 377.9500 | 0.7812 | 0.9626 | 0.8397 | 178.7482 | 0.0164 |
| popularity | 1240 | 0.0863 | 0.0282 | 0.0081 | 0.0056 | 1001.7400 | 375.4000 | 0.5000 | 0.9500 | 0.8312 | 194.0892 | 0.0240 |
