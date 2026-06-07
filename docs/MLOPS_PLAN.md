# MLOps plan

This is how I would turn the current demo into a more realistic ML service.

## Current state

The repo already has a few practical pieces:

- repeatable build scripts;
- saved artifacts with a manifest;
- tests and smoke tests;
- GitHub Actions CI;
- Docker and docker-compose;
- Prometheus-style metrics;
- an offline drift report.

That is enough for a portfolio demo, but not enough for a real production recommender.

## What I would add next

### Data and feature control

- Scheduled data validation.
- Checks for missing columns, type changes, duplicates, and unexpected category drift.
- Clear separation between training-time and serving-time features.
- A feature store or at least versioned feature snapshots.

### Model lifecycle

- Model registry with version, metrics, training data window, and owner.
- Promotion rules based on validation, offline checks, and experiment results.
- Rollback process if a model hurts guardrail metrics.
- Periodic retraining only after checking data quality and business need.

### Monitoring

- API latency and error rate.
- Recommendation coverage and diversity.
- Drift in user, item, and score distributions.
- Click, cart, purchase, and revenue proxy trends.
- Low-confidence or fallback recommendation rates.

### Deployment

- Secrets managed outside the repo.
- Private networking for metrics.
- TLS, rate limiting, and request logging.
- Container image scanning.
- Staging environment before production.

## Practical first milestone

The first real milestone would be a staging deployment with production-like logging, not a stronger model. Without reliable logging, it is hard to know whether any model change is genuinely helping.
