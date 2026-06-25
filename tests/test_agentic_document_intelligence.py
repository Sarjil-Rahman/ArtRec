from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.agentic.agent import AgenticRAGPipeline
from artrec.agentic.evaluation import mrr, ndcg_at_k, precision_at_k, recall_at_k
from artrec.agentic.ingestion import chunk_documents, load_sample_documents
from artrec.agentic.models import Document
from artrec.agentic.rerank import HeuristicReranker
from artrec.agentic.retrieval import TfidfRetriever


def test_chunking_preserves_metadata_and_offsets():
    docs = load_sample_documents()
    chunks = chunk_documents(docs, chunk_size=160, overlap=20)
    first = chunks[0]
    assert first.document_id == docs[0].document_id
    assert first.title == docs[0].title
    assert first.source_path.endswith(".json")
    assert first.start_char == 0
    assert first.end_char <= len(docs[0].text)
    assert first.text


def test_retrieval_returns_relevant_medical_chunk():
    docs = load_sample_documents()
    chunks = chunk_documents(docs)
    retriever = TfidfRetriever(chunks)
    results = retriever.search("oxygen saturation coverage approved", top_k=3, domain="medical")
    assert results
    assert results[0].chunk.document_id == "doc_med_002"


def test_reranker_prefers_title_and_query_overlap():
    docs = [
        Document(
            document_id="a",
            title="Generic Note",
            domain="legal",
            source_path="memory",
            text="support response credit",
        ),
        Document(
            document_id="b",
            title="Priority One Support Clause",
            domain="legal",
            source_path="memory",
            text="vendor acknowledge priority one support tickets within two business hours",
        ),
    ]
    retriever = TfidfRetriever(chunk_documents(docs, chunk_size=200, overlap=0))
    raw = retriever.search("priority one support obligation", top_k=2, domain="legal")
    reranked = HeuristicReranker().rerank("priority one support obligation", raw, domain="legal")
    assert reranked[0].chunk.document_id == "b"
    assert reranked[0].rerank_score is not None


def test_agent_refuses_when_evidence_is_insufficient():
    pipeline = AgenticRAGPipeline(load_sample_documents())
    result = pipeline.ask("What medication allergy does Jordan Ellis have?", domain="medical")
    assert result.route == "unanswerable"
    assert not result.cited_chunks
    assert result.warnings


def test_agent_answers_with_citations_when_evidence_available():
    pipeline = AgenticRAGPipeline(load_sample_documents())
    result = pipeline.ask("What follow-up was recommended for Jordan Ellis?", domain="medical")
    assert result.route == "patient_summary"
    assert result.cited_chunks
    assert result.cited_chunks[0].document_id == "doc_med_001"
    assert result.evidence_score > 0
    assert "Synthetic document for portfolio testing only" not in result.final_answer
    assert "primary care review within seven days" in result.final_answer


def test_structured_extraction_returns_valid_pydantic_object():
    pipeline = AgenticRAGPipeline(load_sample_documents())
    result = pipeline.extractor.extract("doc_med_001", "patient_summary")
    assert result.extracted["patient_name"] == "Jordan Ellis"
    assert result.extracted["age"] == 54
    assert result.citations
    assert result.confidence >= 0.8


def test_evaluation_metric_helpers():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "d"]
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert mrr(retrieved, relevant) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 3) <= 1
