# ArtRec: Production-Style Art Recommendation and Document Intelligence System

ArtRec is an end-to-end portfolio project built around a synthetic art
marketplace. It combines recommendation modelling, retrieval/search,
business-aware reranking, FastAPI serving, monitoring outputs, evaluation
reports, and deterministic document-grounded RAG-style explanation.

The project uses synthetic data. I do not claim real commercial uplift,
production traffic, or real customer behaviour. The value of the project is in
the system design, engineering structure, testing, and evaluation workflow.

```text
synthetic catalogue and users
-> candidate retrieval
-> ranking model
-> business-aware reranking
-> FastAPI service
-> metrics, drift checks, reports, tests, Docker, CI
-> deterministic document-grounded explanation
```

## Why I Built This

I built this project to practise a production-style recommendation system
rather than just a notebook model. My goal was to connect the full workflow:
synthetic user-item interaction data, retrieval, ranking, business reranking,
API serving, monitoring, evaluation, and portfolio-level documentation.

I chose an art recommendation use case because it is easy to explain visually,
but still has realistic recommendation problems such as sparse interactions,
cold-start users, popularity bias, ranking trade-offs, and business constraints.

The project uses synthetic data, so I do not claim real commercial uplift. The
value of the project is in the system design, engineering structure, testing,
and evaluation workflow.

## What This Project Demonstrates

For data science roles:

- feature engineering;
- ranking evaluation;
- cohort analysis;
- uplift-style thinking;
- business metric trade-offs.

For ML engineering roles:

- package structure;
- model artifacts;
- API serving;
- tests;
- Docker;
- CI;
- monitoring endpoints.

For AI engineering roles:

- retrieval;
- document-grounded answers;
- citation-style evidence;
- deterministic evaluation;
- safe fallback behaviour.

## What Is Included

- Synthetic art catalogue, user profiles, impressions, interactions, and
  purchases.
- Candidate retrieval using nearest neighbours over item and user vectors.
- A lightweight scikit-learn ranking model trained on pre-impression features.
- Business-aware reranking for price fit, margin, freshness, diversity, semantic
  intent, and negative feedback.
- TF-IDF search over artwork metadata.
- FastAPI endpoints for recommendations, search, similar items, feedback, health
  checks, metrics, and a small document-intelligence demo.
- Prometheus-compatible metrics and an offline drift check.
- Docker, docker-compose, GitHub Actions CI, tests, model/data docs, and
  generated reports.

## Design Decisions

Synthetic data was used because the project is for portfolio review and should
be runnable without private customer data. It lets the full pipeline be tested,
but it cannot prove real-world business impact.

Time-based validation is used because recommender systems should avoid training
on future user behaviour. Random splits can make a ranking model look better
than it would be when serving new sessions.

The serving model is intentionally simple and inspectable. A smaller model with
clear artifacts, tests, and API behaviour is easier to review than a complex
model that hides data leakage or serving assumptions. The model feature allowlist
excludes post-event labels, simulator-only utility scores, and logged position
features.

Business reranking is applied after relevance scoring so the relevance versus
exposure trade-off can be inspected. This makes it possible to discuss price
fit, diversity, margin, novelty, and popularity bias without claiming that every
setting improves all metrics.

Drift and monitoring are included to show where production monitoring would
fit. The bundled drift alerts are expected in the synthetic example because the
reference and current windows differ; they demonstrate monitoring behaviour and
are not a real production incident.

The document-intelligence component is deterministic by default. This keeps CI
and local testing reproducible, avoids external API keys, and makes citation and
refusal behaviour easier to test. A production version could replace this with
a live LLM provider and separate prompt, safety, and cost evaluation.

## Build Notes From The Project

The hardest part was not getting a model to train. The harder part was making
sure the numbers were not accidentally too good.

Early on, it was very easy to create leakage without meaning to. For example,
features such as logged position, simulator affinity, dwell time, or later user
feedback can make offline metrics look better, but they are not fair serving
features. I ended up keeping the supervised labels in the processed tables for
training and evaluation, while using an explicit model feature allowlist for
serving-safe inputs only.

The synthetic behaviour simulator also took more tuning than I expected. If the
click and purchase rates are too clean, the recommender looks unrealistically
strong. If they are too noisy, the rest of the pipeline becomes hard to inspect.
I spent time balancing the funnel so the generated data is useful for testing
without pretending to be real marketplace behaviour.

Another awkward part was the reporting layer. Reranking an already logged slate
can show useful diagnostics, but it is not the same thing as a real A/B test. I
kept the reports because they are useful for discussing trade-offs, but I changed
the wording to logged-policy replay diagnostics rather than claiming causal
uplift.

The model artifacts were also a practical reminder that saved `joblib` files are
environment-sensitive. That is why the project pins the Python/dependency
versions and includes rebuild scripts instead of treating the committed artifacts
as magic files.

## What Did Not Work Perfectly

- Some business reranking settings trade off relevance for diversity, freshness,
  margin, or wider artist exposure.
- Synthetic data is useful for testing the workflow, but it cannot prove real
  customer behaviour or commercial impact.
- The deterministic local RAG provider is used for reproducible tests and does
  not pretend to be a deployed LLM app.
- Drift alerts in the bundled report are expected because the synthetic windows
  differ. I kept the alerts visible rather than hiding them.

## How I Would Productionise This Further

- Replace synthetic events with real user interaction, impression, feedback, and
  conversion data.
- Add feedback logging for recommendations, searches, skips, saves, carts, and
  purchases.
- Track models and datasets with MLflow or another model registry.
- Add scheduled retraining and validation gates.
- Deploy the API to a cloud environment with reproducible infrastructure.
- Add proper authentication, rate limiting, and private metrics access.
- Use a vector database for retrieval if the catalogue and document store grew.
- Add live LLM integration with API-key management, prompt evaluation, safety
  checks, and cost monitoring.

## Repository Structure

```text
src/artrec/              # Python package
  api/                   # FastAPI app and request/response schemas
  agentic/               # Local document-intelligence / RAG-style demo
  business/              # KPI summary and report writer
  data/                  # Synthetic data generation and adapters
  evaluation/            # Offline recommender metrics
  features/              # Feature engineering
  models/                # Ranking models
  monitoring/            # Prometheus metrics and drift checks
  rerank/                # Business reranking rules
  retrieval/             # Nearest-neighbour and TF-IDF retrieval
  serve/                 # Recommendation service layer
  simulation/            # User behaviour simulator
scripts/                 # Build, evaluation, and report scripts
data/                    # Small bundled synthetic demo data
artifacts/               # Small bundled demo artifacts
reports/                 # Generated portfolio reports
docs/                    # Design notes, cards, limitations, and deployment notes
tests/                   # Unit and smoke tests
ops/                     # Prometheus config and example alerts
.github/workflows/       # CI workflow
Dockerfile
docker-compose.yml
```

`data/` and `artifacts/` are intentionally committed because they are small and
let the API run straight after installation. They can be regenerated with:

```bash
python scripts/build_all.py
```

## Prerequisites

Use Python 3.10 for the local environment. The saved `joblib` model artifacts
are tied to the dependency versions in `requirements.txt`; newer Python or
scikit-learn versions may fail to load them.

Recommended tools:

- Python 3.10.x
- Git
- Docker Desktop, optional, for the container demo

On Windows CMD, check your installed Python versions with:

```cmd
py -0
py -3.10 --version
```

If `py -3.10 --version` fails, install Python 3.10, then reopen CMD.

## Run Locally With Windows CMD

```cmd
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts\build_all.py
python scripts\evaluate_system.py
python scripts\evaluate_agentic_rag.py
python scripts\check_drift.py
python -m pytest -q

python -m uvicorn --app-dir src artrec.api.main:app --host 127.0.0.1 --port 8000
```

## Run Locally With PowerShell

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts\build_all.py
python scripts\evaluate_system.py
python scripts\evaluate_agentic_rag.py
python scripts\check_drift.py
python -m pytest -q

python -m uvicorn --app-dir src artrec.api.main:app --host 127.0.0.1 --port 8000
```

## Run Locally On macOS Or Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/build_all.py
python scripts/evaluate_system.py
python scripts/evaluate_agentic_rag.py
python scripts/check_drift.py
python -m pytest -q

python -m uvicorn --app-dir src artrec.api.main:app --host 127.0.0.1 --port 8000
```

## Quick API Checks

With the API running on `127.0.0.1:8000`:

```cmd
curl http://127.0.0.1:8000/health
```

```cmd
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\"user_id\":\"user_0000\",\"limit\":5}"
```

```cmd
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d "{\"query\":\"blue abstract landscape painting\",\"limit\":5}"
```

```cmd
curl http://127.0.0.1:8000/metrics
```

Other useful routes:

- `GET /similar/{item_id}`
- `POST /feedback/not-interested`
- `POST /agentic/ask`
- `POST /agentic/extract`
- `POST /agentic/evaluate`

## Example API Output

Example `POST /recommend` request:

```json
{
  "user_id": "user_0000",
  "limit": 1,
  "query": "blue abstract landscape painting",
  "max_price": 500
}
```

Example response shape:

```json
{
  "user_id": "user_0000",
  "count": 1,
  "recommendations": [
    {
      "item_id": "item_0049",
      "title": "Silence in Mirror Lake",
      "artist_display_name": "Ari Valette",
      "style": "abstract",
      "medium": "Gouache on board",
      "price": 132.12,
      "retrieval_similarity": 0.6562,
      "semantic_similarity": 0.1259,
      "model_score": 0.5034,
      "business_score": 0.5417,
      "reason_codes": "price_fit",
      "final_rank": 1
    }
  ]
}
```

The exact artwork IDs and scores can change after regenerating synthetic data or
model artifacts.

## Run With Docker

Docker is optional. It is useful if I want to show the API without asking
someone to create a Python environment manually.

Build and run the API container from the project root:

```cmd
docker build -t artrec-platform .
docker run --rm -p 8000:8000 artrec-platform
```

Then test it from another CMD window:

```cmd
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

To run the API with the monitoring demo:

```cmd
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:9090
```

Stop with `Ctrl + C`, then clean up with:

```cmd
docker compose down
```

If CMD does not recognise `docker`, install Docker Desktop and reopen CMD.

## Optional API Key Mode

By default, the API runs in local demo mode without authentication. To require
an API key on protected endpoints:

```cmd
set ARTREC_API_KEY=local-secret
python -m uvicorn --app-dir src artrec.api.main:app --host 127.0.0.1 --port 8000
```

Send `X-API-Key: local-secret` with protected requests. `/health` remains public
for the local demo. `/metrics` is public only when no `ARTREC_API_KEY` is set;
when an API key is configured, metrics require the same header. In a real
deployment, `/metrics` should be private or behind internal networking.

## Main Outputs

- `data/raw/catalog.csv`: synthetic art catalogue.
- `data/raw/users.csv`: synthetic users and preferences.
- `data/raw/impressions.csv`: synthetic recommendation impressions.
- `data/raw/interactions.csv`: synthetic clicks, saves, carts, and purchases.
- `data/raw/conversions.csv`: synthetic immediate and delayed purchases.
- `data/processed/train_features.csv`: training feature rows.
- `data/processed/validation_features.csv`: disjoint later validation feature rows.
- `data/processed/test_features.csv`: held-out feature rows.
- `artifacts/retriever.joblib`: nearest-neighbour retriever.
- `artifacts/semantic_retriever.joblib`: TF-IDF search retriever.
- `artifacts/ranker.joblib`: serving ranker.
- `artifacts/model_metrics.json`: classifier and recommender metrics.
- `artifacts/model_comparison.md`: train/validation/test comparison.
- `artifacts/business_report.md`: generated business summary.
- `artifacts/manifest.json`: lightweight artifact manifest.
- `artifacts/drift_report.json`: offline drift report.

## Notes On Interpretation

The project reports ROC AUC, average precision, Precision@K, Recall@K,
HitRate@K, NDCG@K, MAP@K, catalogue coverage, artist diversity, novelty, and
offline logged-slate replay diagnostics.

These are useful engineering metrics, but they are not live business proof.
Metrics are from synthetic data and should be used to judge the pipeline and
evaluation workflow, not claimed real commercial impact. Recommender metrics can
also be biased by what the user was shown, where items appeared on the page, and
how the simulator generates behaviour. A real deployment would need online
testing, careful logging, and guardrail metrics.

The bundled `artifacts/drift_report.json` currently contains alerts. They are
expected in this demo because the synthetic reference and current windows are
different. The alerts demonstrate how monitoring would surface distribution
changes; they are not a real production incident.

## Decision And Reporting Layer

I added this layer to make the recommendation work easier to discuss from a
product and business point of view. It includes:

- business trade-off reports across popularity, model, business-aware, and
  diversity-aware strategies;
- an offline A/B simulation with deterministic assignment and a simple bootstrap
  interval;
- uplift cuts by segment, price band, style, and artist popularity tier;
- explainable user segments from interaction features;
- weekly cohort engagement retention;
- sequence-aware baselines using time-based splits.

Run the reports with:

```cmd
python scripts\generate_business_tradeoff_report.py
python scripts\run_ab_test_simulation.py
python scripts\build_user_segments.py
python scripts\run_uplift_analysis.py
python scripts\generate_cohort_retention.py
python scripts\evaluate_sequential_baselines.py
python -m pytest -q
```

Generated outputs are written to `reports/` and
`data/processed/user_segments.csv`.

## Search, Hybrid Recommendation, And Feedback

The personalised path retrieves candidates from synthetic user and item vectors.
The `/search` endpoint uses TF-IDF over metadata such as title, artist, style,
medium, tags, palette, hue, and description.

When `/recommend` includes a query, the service combines personalised candidates
and search candidates, then reranks them. Explicit `not_interested` feedback is
used in the simulator, training features, and in-memory serving feedback. The
model treats repeated negative feedback as a signal to lower similar items,
artists, and styles.

## Document-Intelligence Demo

The `src/artrec/agentic/` package is a small local RAG-style workflow over
synthetic legal, medical, insurance, compliance, and support notes.

```text
load sample documents
-> chunk with metadata
-> retrieve with TF-IDF
-> rerank
-> check evidence sufficiency
-> answer with citations or refuse
-> optional structured extraction
```

It uses a deterministic local provider by default, so it does not need paid APIs
or secrets. This part is useful for showing retrieval evaluation, citations,
refusal behaviour, schema validation, and traces. It is not suitable for real
legal, medical, insurance, or compliance use.

## GitHub Hygiene Before Committing

Before committing, check that local-only files are not tracked:

```cmd
git status --ignored
git ls-files | findstr /i ".venv __pycache__ .pytest_cache .env .pyc .zip"
```

Do not commit real `.env` files, virtual environments, cache folders, local
archives, prompt files, local assistant settings, instruction files, or private
scratch notes. `.env.example` is fine because it contains placeholders, not
secrets.

## Documentation

- [System overview](docs/SYSTEM_OVERVIEW.md)
- [Engineering notes](docs/ENGINEERING_NOTES.md)
- [Business case](docs/BUSINESS_CASE.md)
- [KPI dictionary](docs/KPI_DICTIONARY.md)
- [Data card](docs/DATA_CARD.md)
- [Model card](docs/MODEL_CARD.md)
- [Limitations](docs/LIMITATIONS.md)
- [Experimentation plan](docs/EXPERIMENTATION_PLAN.md)
- [Evaluation limitations](docs/EVALUATION_LIMITATIONS.md)
- [MLOps plan](docs/MLOPS_PLAN.md)
- [Security notes](docs/SECURITY.md)
- [Scale plan](docs/SCALE_PLAN.md)
- [Deployment notes](docs/DEPLOYMENT_NOTES.md)
- [Product feedback loop](docs/PRODUCT_FEEDBACK_LOOP.md)
- [Agentic RAG system](docs/agentic_rag_system.md)
- [LLM evaluation framework](docs/llm_evaluation_framework.md)
- [Release checklist template](docs/RELEASE_CHECKLIST.md)

## License

This project is released under the [MIT License](LICENSE).
