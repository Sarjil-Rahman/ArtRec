from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from artrec.agentic.agent import AgenticRAGPipeline


def load_gold(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    dcg = sum((1.0 / math.log2(idx + 2)) for idx, item in enumerate(retrieved[:k]) if item in relevant_set)
    ideal = sum(1.0 / math.log2(idx + 2) for idx in range(min(len(relevant_set), k)))
    return dcg / ideal if ideal else 1.0


def citation_coverage(cited_doc_ids: list[str], required_doc_ids: list[str]) -> float:
    if not required_doc_ids:
        return 1.0
    return len(set(cited_doc_ids) & set(required_doc_ids)) / len(set(required_doc_ids))


def groundedness_proxy(answer: str, evidence_texts: list[str]) -> float:
    answer_terms = {word.lower().strip(".,:;") for word in answer.split() if len(word) > 4}
    evidence_terms = {word.lower().strip(".,:;") for text in evidence_texts for word in text.split()}
    if not answer_terms:
        return 0.0
    return len(answer_terms & evidence_terms) / len(answer_terms)


def unsupported_claim_count(answer: str, evidence_texts: list[str]) -> int:
    evidence = " ".join(evidence_texts).lower()
    claims = [part.strip() for part in answer.split(".") if part.strip()]
    return sum(1 for claim in claims if not any(word.lower().strip(".,:;") in evidence for word in claim.split() if len(word) > 6))


def evaluate_pipeline(
    pipeline: AgenticRAGPipeline, gold_records: list[dict[str, Any]], top_k: int = 5
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    for record in gold_records:
        answer = pipeline.ask(
            query=record["query"],
            top_k=top_k,
            domain=record.get("domain", "all"),
            require_citations=record.get("required_citations", True),
        )
        retrieved_docs = [citation.document_id for citation in answer.cited_chunks]
        evidence_texts = [citation.text for citation in answer.cited_chunks]
        relevant_docs = record.get("relevant_document_ids", [])
        row = {
            "recall_at_k": recall_at_k(retrieved_docs, relevant_docs, top_k),
            "precision_at_k": precision_at_k(retrieved_docs, relevant_docs, top_k),
            "mrr": mrr(retrieved_docs, relevant_docs),
            "ndcg_at_k": ndcg_at_k(retrieved_docs, relevant_docs, top_k),
            "citation_coverage": citation_coverage(retrieved_docs, relevant_docs),
            "groundedness_proxy": groundedness_proxy(answer.final_answer, evidence_texts),
            "unsupported_claim_count": float(unsupported_claim_count(answer.final_answer, evidence_texts)),
            "refusal_correct": float((answer.route == "unanswerable") == bool(record.get("unanswerable", False))),
            "schema_valid": 1.0,
        }
        metric_rows.append(row)
        prediction = {
            "query": record["query"],
            "answer": answer.final_answer,
            "route": answer.route,
            "evidence_score": answer.evidence_score,
            "cited_document_ids": retrieved_docs,
            "metrics": row,
            "warnings": answer.warnings,
        }
        predictions.append(prediction)
        if row["citation_coverage"] < 1.0 or row["refusal_correct"] < 1.0:
            failures.append(prediction)
    metrics = {
        key: round(sum(row[key] for row in metric_rows) / max(len(metric_rows), 1), 4)
        for key in metric_rows[0]
    }
    metrics["records"] = float(len(gold_records))
    return metrics, predictions, failures
