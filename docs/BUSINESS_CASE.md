# Business case

## Target setting

The imagined product is a small online art marketplace or curated print store. Users browse by style, medium, price range, colour, and artist. The business wants better discovery without showing the same popular work to everyone.

## Problems this project tries to model

- Users reach dead ends because browsing is too flat.
- Similar-item recommendations are weak or too repetitive.
- Returning users do not get enough personalisation.
- High-margin or fresh items are not considered during ranking.
- The business has no clear way to compare ranking strategies.

## Value story

A recommender can help only if it improves the user experience and the commercial outcome together. In this demo I track both sides:

- relevance signals such as clicks, saves, carts, and purchases;
- product health signals such as catalogue coverage, artist diversity, and novelty;
- business signals such as revenue proxy, margin proxy, and price fit.

The point is not to optimise one number blindly. A ranking that increases clicks but hides new artists or low-margin work may not be the right product decision.

## ROI framing

The bundled reports use synthetic data, so the results should be described as offline proxy results. They are useful for comparing strategies and finding failure cases. They are not real ROI.

For a real client or employer, the next step would be proper impression logging, treatment/control assignment, enough traffic for statistical power, and guardrail metrics such as latency, refund/return rate, diversity, and customer complaints.

## Interview discussion points

- How ranking choices affect both user experience and margin.
- Why offline uplift is weaker evidence than an online experiment.
- How I would design a first A/B test for a recommender.
- What events I would log before trusting any model result.
