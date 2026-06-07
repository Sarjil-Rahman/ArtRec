# Engineering notes

## Initial goal

ArtRec started as a recommendation system project. I expanded it into a
production-style ML system so the work could be reviewed as more than a
notebook model. The current project connects synthetic data generation,
retrieval, ranking, business reranking, API serving, tests, monitoring outputs,
business reports, and document-grounded explanation.

## Main technical choices

### Package structure

The project uses a `src/` layout so application code is imported as an installed
package rather than from whichever directory the command happens to run in. This
makes tests, scripts, and the API use the same import paths.

### Reproducible scripts

The main workflows are script-based instead of notebook-only. This makes the
pipeline easier to rerun from a clean checkout, easier to test in CI, and easier
to explain in a portfolio review.

### Testing strategy

The tests cover synthetic data generation, training smoke checks, bundled
artifacts, API behaviour, semantic search, drift monitoring, decision reports,
and the document-intelligence workflow. The aim is not exhaustive production
coverage, but enough checks to catch broken artifacts, invalid API responses,
and accidental regressions in the main demo paths.

### API design

The FastAPI layer is intentionally small. It loads the service artifacts,
validates request payloads, records basic metrics, and delegates recommendation
logic to the service layer. Keeping the API thin makes it easier to inspect what
is serving logic and what is modelling or retrieval logic.

### Monitoring

The project includes Prometheus-style request metrics and an offline drift
report. These are portfolio monitoring examples: they show where latency,
request counts, and feature-distribution checks would fit before a real
deployment. The bundled drift alerts are expected because the synthetic
reference and current windows differ.

### Document intelligence

The document-intelligence component uses a deterministic local provider by
default. This is intentional because it gives:

- reproducible CI;
- no external API key requirement;
- easier tests for citations, refusals, and schema extraction;
- a clear replacement point for a live LLM provider in a production version.

The current implementation should be read as a document-grounded retrieval and
evaluation demo, not as a deployed LLM application.

## Things I would improve next

- Real user feedback loop.
- MLflow or another model registry.
- Proper vector database for retrieval.
- Scheduled retraining.
- Cloud deployment.
- Authentication and rate limiting.
- Live LLM provider with prompt evaluation and cost monitoring.

## Development timeline

1. Built first synthetic recommendation pipeline.
2. Added offline evaluation and ranking metrics.
3. Added semantic/content search.
4. Added business reranking.
5. Wrapped the model in FastAPI.
6. Added tests.
7. Added monitoring, A/B simulation, uplift, cohorts, and reports.
8. Cleaned the repo for GitHub portfolio review.
