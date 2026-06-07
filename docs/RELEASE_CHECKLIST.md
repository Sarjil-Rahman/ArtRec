# Release checklist template

This is a reusable checklist for a future release or portfolio demo. It is not a
list of unfinished product work.

## Local run

- [ ] Python 3.10 environment created.
- [ ] `pip install -r requirements.txt` completed.
- [ ] `pip install -e .` completed.
- [ ] `python scripts/build_all.py` completed.
- [ ] `python scripts/evaluate_system.py` completed.
- [ ] `python scripts/evaluate_agentic_rag.py` completed.
- [ ] `python scripts/check_drift.py` completed.
- [ ] `python -m pytest -q` passed.

## API smoke test

- [ ] `GET /health` returns OK.
- [ ] `POST /recommend` returns a ranked list.
- [ ] `POST /search` returns catalogue matches.
- [ ] `GET /similar/{item_id}` returns similar artworks.
- [ ] `GET /metrics` returns Prometheus-style metrics.

## GitHub hygiene

- [ ] `.venv/` is not tracked.
- [ ] `.env` is not tracked.
- [ ] Private scratch notes are not tracked.
- [ ] `__pycache__/` and `.pytest_cache/` are not tracked.
- [ ] Zip files are not tracked.
- [ ] `.env.example` is tracked and contains placeholders only.

## Portfolio review

- [ ] README explains how to run locally.
- [ ] README explains Docker usage.
- [ ] Limitations are honest about synthetic data.
- [ ] Reports are clearly labelled as offline or simulated.
- [ ] CI is present and uses Python 3.10.
- [ ] The project can be discussed as an end-to-end ML workflow, not just a notebook.
