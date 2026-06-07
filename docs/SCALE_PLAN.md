# Scale plan

The current project is small by design. This is how I would scale it if it moved towards real usage.

## First step: better logs

Before changing the model, I would make sure the product logs the right events: impressions, positions, clicks, saves, carts, purchases, returns, timestamps, user/session IDs, item IDs, and experiment assignment.

## Model and retrieval scale

- Move candidate retrieval to a vector index or search service.
- Refresh item and user features on a schedule.
- Add fallback recommendations for cold-start users and new items.
- Version all artifacts and keep rollback copies.

## Serving scale

- Package the API in a container.
- Run behind a gateway with authentication, TLS, and rate limiting.
- Cache common responses where possible.
- Monitor p50/p95/p99 latency and error rate.

## Business scale

- Track coverage and diversity so the marketplace does not over-expose only the most popular artists.
- Test ranking changes with controlled experiments.
- Review guardrails before increasing traffic.

The main idea: scale the measurement system before scaling model complexity.
