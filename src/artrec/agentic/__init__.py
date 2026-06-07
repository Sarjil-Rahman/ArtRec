"""Local document-intelligence and agentic RAG utilities."""

from artrec.agentic.agent import AgenticRAGPipeline
from artrec.agentic.extraction import StructuredExtractor
from artrec.agentic.ingestion import load_sample_documents
from artrec.agentic.retrieval import TfidfRetriever

__all__ = [
    "AgenticRAGPipeline",
    "StructuredExtractor",
    "TfidfRetriever",
    "load_sample_documents",
]
