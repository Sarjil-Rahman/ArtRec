# Local document-intelligence demo

This part of the repo is a small RAG-style workflow over synthetic documents. I added it to show a second applied AI pattern alongside the recommender: retrieval, citation grounding, refusal behaviour, and structured extraction.

It runs locally and does not call an external LLM by default.

## Flow

```text
sample documents
-> deterministic chunking
-> TF-IDF retrieval
-> heuristic reranking
-> evidence sufficiency check
-> cited answer or refusal
-> optional structured extraction
```

## Why it is included

Recommendation systems and document systems share some useful engineering habits: retrieval quality, evaluation sets, traceability, and clear failure handling. This small module lets those ideas be tested without needing paid APIs or private documents.

## API routes

- `POST /agentic/ask`
- `POST /agentic/extract`
- `POST /agentic/evaluate`

## Evaluation

The evaluation script uses a small synthetic gold file and checks retrieval, citations, refusal behaviour, groundedness proxy, and schema validity.

```bash
python scripts/evaluate_agentic_rag.py
```

## Limits

This is not a legal, medical, insurance, or compliance system. The documents are synthetic and the evidence score is a simple proxy. A real deployment would need human-labelled examples, domain review, access control, audit logs, privacy handling, and monitoring for low-evidence answers.
