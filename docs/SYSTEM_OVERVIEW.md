# System overview

This project is a local demo of an art recommendation service. I designed it as if it were the first internal version of a marketplace recommender: small enough to run on a laptop, but structured enough to discuss in a technical interview.

## Flow

```text
catalogue + users + events
-> feature engineering
-> retrieval
-> ranking model
-> business reranking
-> API serving
-> reporting and monitoring checks
```

## Main components

- `src/artrec/simulation`: creates synthetic users, artwork, impressions, and interaction events.
- `src/artrec/features`: builds train, validation, and test feature tables.
- `src/artrec/retrieval`: retrieves candidate artworks using nearest-neighbour and TF-IDF approaches.
- `src/artrec/models`: trains the serving ranker and a stronger LightGBM comparison model.
- `src/artrec/rerank`: applies business rules after model scoring.
- `src/artrec/serve`: keeps the recommendation service logic separate from the API layer.
- `src/artrec/api`: exposes the service through FastAPI.
- `src/artrec/monitoring`: adds metrics and drift checks.
- `scripts/`: rebuilds data, artifacts, evaluations, and reports.

## Serving behaviour

The API loads the committed demo artifacts from `artifacts/`. A recommendation request retrieves candidates, scores them, applies reranking, and returns a ranked list with enough fields to inspect why items were shown.

Search requests use artwork metadata. Similar-item requests use the nearest-neighbour retriever. Feedback requests update an in-memory store for the local demo.

## What is deliberately simple

The project does not include a real feature store, model registry, online experiment service, identity system, secret manager, or cloud deployment. Those would be the next layers after proving the first version with real data and traffic.
