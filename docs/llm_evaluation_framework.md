# RAG evaluation framework

The evaluation code lives in `src/artrec/agentic/evaluation.py` and can be run with:

```bash
python scripts/evaluate_agentic_rag.py
```

The gold file is `data/eval/rag_gold.jsonl`.

## What I wanted to test

The goal is not to make the metrics look impressive. The goal is to catch obvious failures:

- the retriever misses the right document;
- the answer has no citation;
- the system answers when it should refuse;
- structured extraction produces invalid output;
- a code change breaks the expected behaviour.

## Metrics

Retrieval metrics:

- `recall_at_k`
- `precision_at_k`
- `mrr`
- `ndcg_at_k`

Answer and evidence metrics:

- `citation_coverage`
- `groundedness_proxy`
- `unsupported_claim_count`
- `refusal_correct`
- `schema_valid`

These are proxies. They are helpful for regression testing, but they do not replace human review. A correct paraphrase can score lower than expected, and a superficially grounded answer can still miss a subtle issue.

## Outputs

Generated outputs are written to `artifacts/agentic_eval/`:

- `metrics.json`
- `predictions.jsonl`
- `failure_cases.jsonl`

That folder is ignored by Git because it is an execution artifact.

## Real-world upgrade path

For a real document product I would add labelled examples, disagreement review, red-team cases, document parsing tests, low-evidence monitoring, protected-data handling, and a human review workflow for high-risk answers.
