# Product feedback loop

A recommender improves only when the product logs useful feedback and the team knows how to act on it.

## Feedback signals in this demo

The simulator creates:

- impressions;
- clicks;
- saves;
- carts;
- purchases;
- not-interested feedback.

The API also has a local `not_interested` endpoint so the serving layer can lower rejected artists, styles, and items during the demo.

## How I would use feedback in production

- Treat impressions as the denominator for all ranking analysis.
- Track position because top slots naturally receive more attention.
- Separate weak signals such as clicks from stronger signals such as carts and purchases.
- Keep negative feedback explicit where possible.
- Monitor whether recommendations become too narrow over time.

## Retraining loop

A practical loop would be:

```text
collect events -> validate data -> build features -> train candidate model
-> evaluate offline -> run experiment -> monitor guardrails -> promote or rollback
```

## Human review

For a small marketplace, I would still include human review for unusual outcomes: overexposure of one artist, low diversity, inappropriate recommendations, or sharp drops in engagement.
