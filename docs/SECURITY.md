# Security notes

This repo is a local demo, but I still treated a few security habits as non-negotiable.

## Included safeguards

- Real `.env` files are ignored.
- `.env.example` contains placeholders only.
- The API can run with an optional API key for protected routes.
- `/health` is left open for the local demo.
- `/metrics` is open only when no `ARTREC_API_KEY` is configured; with an API key set, metrics require `X-API-Key`.
- Docker ignores local virtual environments, cache files, private notes, and generated evaluation outputs.

## Important production gaps

A real deployment would need more:

- managed secrets instead of local environment variables;
- TLS termination;
- authentication and authorisation;
- rate limiting;
- request size limits;
- private access for metrics;
- audit logging;
- dependency and container scanning;
- privacy review for any user-level data.

## Document-intelligence safety

The RAG-style demo uses synthetic notes only. It should not be used for real legal, medical, insurance, or compliance decisions. A real system would need access control, data handling rules, human review, audit trails, and domain expert validation.
