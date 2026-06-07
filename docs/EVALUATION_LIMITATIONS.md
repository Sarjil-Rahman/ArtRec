# Evaluation limitations

The evaluation in this repo is useful, but it has clear boundaries.

## What the current evaluation can show

- Whether the pipeline runs end to end.
- Whether model artifacts can be loaded.
- Whether ranking metrics improve or regress on the synthetic holdout set.
- Whether retrieval and RAG-style answers pass small regression checks.
- Whether business reports are generated consistently.

## What it cannot show

- Real customer satisfaction.
- Real revenue or margin lift.
- True causal effect of the reranker.
- Long-term user retention.
- Safety in regulated domains.
- Production reliability under real traffic.

## Why offline metrics can mislead

Users only interact with the items they are shown. That means the logged data already contains bias. A model can also look better offline because it learns the simulator's rules rather than real human behaviour.

For this reason I treat the numbers as diagnostics, not proof.

## How I would evaluate it properly

For a real marketplace I would start with better logging: impression IDs, user/session IDs, item IDs, rank positions, timestamps, experiment group, conversion events, returns/refunds, and inventory state. After that I would run a controlled experiment with guardrails for latency, conversion, revenue, diversity, coverage, and customer experience.
