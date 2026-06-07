from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from artrec.agentic.models import DocumentChunk, RetrievalResult


class TfidfRetriever:
    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("chunks cannot be empty")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        corpus = [f"{chunk.title} {chunk.domain} {chunk.text}" for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(
        self, query: str, top_k: int = 5, domain: str | None = None
    ) -> list[RetrievalResult]:
        clean_query = query.strip()
        if not clean_query:
            return []
        query_vector = self.vectorizer.transform([clean_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        results: list[RetrievalResult] = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if domain and domain != "all" and chunk.domain != domain:
                continue
            if score > 0:
                results.append(RetrievalResult(chunk=chunk, score=float(score)))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]
