from __future__ import annotations

from abc import ABC, abstractmethod
import re


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
BOILERPLATE_SENTENCES = {
    "synthetic document for portfolio testing only",
}


class BaseLLMProvider(ABC):
    @abstractmethod
    def answer(self, query: str, evidence: list[str]) -> str:
        raise NotImplementedError


class LocalDeterministicProvider(BaseLLMProvider):
    """Deterministic provider for reproducible local document-grounded answers.

    The current portfolio demo keeps generation local so CI, tests, and API
    smoke checks do not depend on an external LLM API key. A production version
    could add a separately tested live provider with prompt evaluation, safety
    checks, API-key management, and cost monitoring.
    """

    def _select_sentence(self, query: str, evidence: list[str]) -> str:
        query_terms = {token.lower() for token in TOKEN_RE.findall(query)}
        candidates: list[str] = []
        for text in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
                sentence = sentence.strip()
                normalised = sentence.strip(".!?").lower()
                if sentence and normalised not in BOILERPLATE_SENTENCES:
                    candidates.append(sentence)
        if not candidates:
            return ""

        def score(sentence: str) -> tuple[int, int]:
            sentence_terms = {token.lower() for token in TOKEN_RE.findall(sentence)}
            return (len(query_terms & sentence_terms), len(sentence_terms))

        return max(candidates, key=score)

    def answer(self, query: str, evidence: list[str]) -> str:
        if not evidence:
            return "I do not have enough evidence in the provided documents to answer."
        selected_sentence = self._select_sentence(query, evidence)
        if selected_sentence:
            return f"Based on the cited synthetic documents: {selected_sentence}"
        return (
            "Based on the cited synthetic documents, the answer is supported by "
            "the retrieved evidence."
        )
