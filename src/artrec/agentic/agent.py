from __future__ import annotations

from artrec.agentic.extraction import StructuredExtractor
from artrec.agentic.ingestion import chunk_documents
from artrec.agentic.llm import BaseLLMProvider, LocalDeterministicProvider
from artrec.agentic.models import AgentAnswer, AgentTraceStep, Citation, Document, TaskType
from artrec.agentic.rerank import HeuristicReranker
from artrec.agentic.retrieval import TfidfRetriever


class AgenticRAGPipeline:
    def __init__(
        self,
        documents: list[Document],
        chunk_size: int = 700,
        overlap: int = 120,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.documents = documents
        self.chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        self.retriever = TfidfRetriever(self.chunks)
        self.reranker = HeuristicReranker()
        self.extractor = StructuredExtractor(documents)
        self.llm_provider = llm_provider or LocalDeterministicProvider()

    def ask(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = "all",
        require_citations: bool = True,
    ) -> AgentAnswer:
        trace: list[AgentTraceStep] = []
        route = self.classify(query)
        trace.append(AgentTraceStep(step="classify", detail={"route": route}))
        retrieved = self.retriever.search(query, top_k=max(top_k * 2, top_k), domain=domain)
        trace.append(AgentTraceStep(step="retrieve", detail={"candidate_count": len(retrieved)}))
        reranked = self.reranker.rerank(query, retrieved, domain=domain)[:top_k]
        trace.append(
            AgentTraceStep(
                step="rerank",
                detail={"top_chunk_ids": [item.chunk.chunk_id for item in reranked]},
            )
        )
        evidence_score = round(sum(item.rerank_score or item.score for item in reranked[:3]) / 3, 3)
        citations = [
            Citation(
                document_id=item.chunk.document_id,
                chunk_id=item.chunk.chunk_id,
                title=item.chunk.title,
                domain=item.chunk.domain,
                score=round(float(item.rerank_score or item.score), 4),
                text=item.chunk.text[:400],
            )
            for item in reranked
        ]
        warnings: list[str] = []
        sufficient = bool(citations) and evidence_score >= 0.12
        trace.append(AgentTraceStep(step="evidence_check", detail={"sufficient": sufficient, "score": evidence_score}))
        if not sufficient:
            warnings.append("low evidence: no sufficiently relevant synthetic document chunk found")
            return AgentAnswer(
                final_answer="I do not have enough evidence in the provided synthetic documents to answer this. Please narrow the question or add a relevant document.",
                cited_chunks=[],
                evidence_score=evidence_score,
                route="unanswerable",
                warnings=warnings,
                trace=trace,
            )
        answer = self.llm_provider.answer(
            query, [item.chunk.text for item in reranked[:2]]
        )
        if require_citations and not citations:
            warnings.append("citations required but none were found")
        trace.append(AgentTraceStep(step="answer", detail={"citation_count": len(citations)}))
        return AgentAnswer(
            final_answer=answer,
            cited_chunks=citations,
            evidence_score=evidence_score,
            route=route,
            warnings=warnings,
            trace=trace,
        )

    def classify(self, query: str) -> TaskType:
        lowered = query.lower()
        if (
            "patient" in lowered
            or "diagnosis" in lowered
            or "discharge" in lowered
            or "follow-up" in lowered
            or "follow up" in lowered
        ):
            return "patient_summary"
        if "policy" in lowered or "coverage" in lowered or "claim" in lowered:
            return "policy_decision"
        if "contract" in lowered or "clause" in lowered or "obligation" in lowered:
            return "contract_clause"
        if "risk" in lowered or "compliance" in lowered:
            return "risk_finding"
        return "question_answering"
