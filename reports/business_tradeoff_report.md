# Business Trade-off Report

This is a production-style demo report built from synthetic offline data. The revenue and profit values are proxy metrics, not real commercial impact.

- Highest profit proxy strategy: `business`.
- Highest artist diversity strategy: `business`.
- Higher profit proxy can coincide with lower catalogue coverage or diversity because reranking gives more exposure to high-margin items.
- Popularity-oriented rankings are easy to explain, but they can concentrate exposure.

## Metrics

| strategy | impressions | ctr_proxy | save_rate_proxy | cart_rate_proxy | purchase_rate_proxy | revenue_proxy | profit_proxy | catalog_coverage | artist_diversity | style_diversity | average_recommended_price | exposure_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| business | 1240 | 0.0903 | 0.0290 | 0.0137 | 0.0065 | 1322.7300 | 460.7700 | 0.6813 | 0.9639 | 0.8536 | 195.8237 | 0.0176 |
| diversity | 1240 | 0.0968 | 0.0315 | 0.0145 | 0.0056 | 1117.5000 | 410.9100 | 0.6500 | 0.9622 | 0.8518 | 181.4040 | 0.0199 |
| model | 1240 | 0.0976 | 0.0323 | 0.0129 | 0.0065 | 1230.2300 | 440.9500 | 0.6625 | 0.9591 | 0.8428 | 182.7495 | 0.0204 |
| popularity | 1240 | 0.0863 | 0.0282 | 0.0081 | 0.0056 | 1001.7400 | 375.4000 | 0.5000 | 0.9500 | 0.8312 | 193.8873 | 0.0240 |
