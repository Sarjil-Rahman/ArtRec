# Limitations

This file is intentionally direct. The project is useful as a portfolio demo, but it should not be oversold.

## Main limitations

- The data is synthetic.
- The model has not served real users.
- There is no live A/B test.
- Revenue, profit, and uplift values are proxy values from the simulator.
- The document-intelligence demo uses synthetic notes, not real legal or medical documents.
- The API is a local demo, not a managed cloud deployment.

## Why this matters

Offline recommender results can be biased by the logging policy, the position of items on the page, and what users were exposed to. With synthetic data, there is one extra issue: the simulator creates the world the model learns from. That makes the project good for engineering practice, but not enough for commercial proof.

## What would be needed next

Before using this with real users, I would add:

- production impression and conversion logging;
- experiment assignment and exposure tracking;
- privacy and access controls;
- model and data versioning;
- live monitoring and rollback;
- human review for any high-risk document workflows;
- a proper online experiment before claiming impact.
