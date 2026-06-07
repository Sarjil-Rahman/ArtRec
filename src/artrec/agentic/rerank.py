from __future__ import annotations

import re

from artrec.agentic.models import RetrievalResult


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}


class HeuristicReranker:
    def rerank(
        self, query: str, results: list[RetrievalResult], domain: str | None = None
    ) -> list[RetrievalResult]:
        query_tokens = _tokens(query)
        reranked: list[RetrievalResult] = []
        for result in results:
            chunk = result.chunk
            title_tokens = _tokens(chunk.title)
            text_tokens = _tokens(chunk.text)
            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            title_match = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
            domain_bonus = 0.08 if domain and domain != "all" and chunk.domain == domain else 0.0
            rerank_score = result.score + (0.25 * overlap) + (0.15 * title_match) + domain_bonus
            reasons = []
            if overlap:
                reasons.append("query_terms_in_chunk")
            if title_match:
                reasons.append("query_terms_in_title")
            if domain_bonus:
                reasons.append("domain_filter_match")
            reranked.append(
                RetrievalResult(
                    chunk=chunk,
                    score=result.score,
                    rerank_score=round(float(rerank_score), 6),
                    reasons=reasons,
                )
            )
        reranked.sort(key=lambda item: item.rerank_score or item.score, reverse=True)
        return reranked
