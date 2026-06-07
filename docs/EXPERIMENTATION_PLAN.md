# Experimentation plan

This is the experiment plan I would use if the recommender moved from synthetic data to a real marketplace.

## Goal

Test whether the new ranking and reranking approach improves discovery and commercial outcomes without damaging user experience or catalogue health.

## Primary metric

Choose one primary metric before the test starts. For an art marketplace, I would likely use purchase rate or gross profit per session, depending on the business goal.

## Guardrail metrics

I would track:

- API latency;
- error rate;
- click-through rate;
- save and cart rate;
- catalogue coverage;
- artist diversity;
- exposure concentration;
- refund or return proxy if available;
- customer complaints or support contacts if available.

## Assignment

Users should be assigned to control or treatment with a stable experiment ID. The assignment must be logged with every impression so the analysis can connect exposure to outcomes.

## Analysis

I would compare treatment and control on the primary metric, check guardrails, and segment the results by new/returning users, price band, style, and traffic source. I would avoid launching if the primary metric improves but guardrails show poor diversity, slower responses, or worse user experience.

## Rollout plan

Start small, for example 5-10% of eligible traffic, then increase only if the metrics are stable. Keep a rollback path ready.

## Why the current repo only simulates this

The repo does not have real traffic, so it cannot run a true A/B test. The offline A/B report is included to show how I think about experiment structure, not to claim causal uplift.
