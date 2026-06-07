# Model card

## Model purpose

The serving model ranks candidate artworks for a synthetic art marketplace. It is used after candidate retrieval and before business reranking.

The model is not intended for high-stakes decisions. It is a portfolio model for recommendation, evaluation, and deployment practice.

## Serving artifact

- Artifact: `artifacts/ranker.joblib`
- Main class: scikit-learn pipeline
- Runtime: Python 3.10 with pinned dependencies in `requirements.txt`
- Input: pre-impression features for a user-item pair
- Output: probability-like ranking score

## Training setup

The training pipeline uses synthetic logged impressions and interaction labels. Feature tables are split by time rather than randomly, so validation and test rows happen after the training rows.

I used pre-impression features only for the serving model. That matters because a serving system cannot use future clicks, carts, or purchases to rank the current slate.

## Models compared

The project includes:

- a lightweight logistic ranker used as the committed serving artifact;
- a LightGBM LambdaRank comparison model used in offline evaluation.

The LightGBM model can fit the synthetic training data strongly, but it is not automatically promoted to serving. I kept the simpler model as the default because it is easier to inspect and more stable for a small portfolio demo.

## Metrics

The project reports both classifier-style and recommender-style metrics:

- ROC AUC
- average precision
- Precision@K
- Recall@K
- HitRate@K
- NDCG@K
- MAP@K
- catalogue coverage
- artist diversity
- novelty
- offline proxy uplift

These are offline metrics on simulated data. They are useful for checking regressions and comparing strategies, but they do not prove real user impact.

## Important limitations

- The model is trained on synthetic behaviour.
- There is no real production traffic.
- There is no online A/B test.
- The simulator may make some patterns easier than they would be in real life.
- Saved scikit-learn artifacts can break across library versions, which is why the project pins the environment.

## Risks in a real deployment

A recommender can over-concentrate exposure on popular artists, reduce discovery, reinforce narrow preferences, or optimise for clicks at the expense of long-term value. In production I would monitor coverage, diversity, latency, complaints, refunds/returns, and experiment guardrails before promoting changes.

## Next improvements

- Add richer sequence features.
- Compare learned retrieval or embedding-based retrieval.
- Calibrate ranking scores.
- Add stronger ablation tests for each reranking rule.
- Use real impression logs and a controlled online experiment before claiming impact.
