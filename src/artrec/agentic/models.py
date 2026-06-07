from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Domain = Literal["medical", "legal", "insurance", "compliance", "support"]
TaskType = Literal[
    "patient_summary",
    "policy_decision",
    "contract_clause",
    "risk_finding",
    "question_answering",
    "unanswerable",
]


class Document(BaseModel):
    document_id: str
    title: str
    domain: Domain
    source_path: str
    text: str


class DocumentChunk(BaseModel):
    document_id: str
    title: str
    domain: Domain
    source_path: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float
    rerank_score: float | None = None
    reasons: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    domain: str
    score: float
    text: str


class AgentTraceStep(BaseModel):
    step: str
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    final_answer: str
    cited_chunks: list[Citation]
    evidence_score: float
    route: TaskType
    warnings: list[str] = Field(default_factory=list)
    trace: list[AgentTraceStep] = Field(default_factory=list)
    extracted: dict[str, Any] | None = None


class FieldCitation(BaseModel):
    field: str
    document_id: str
    chunk_id: str
    text: str


class ExtractionResult(BaseModel):
    schema_type: str
    extracted: dict[str, Any]
    citations: list[FieldCitation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class PatientSummary(BaseModel):
    patient_name: str | None = None
    age: int | None = None
    diagnosis: str | None = None
    discharge_plan: str | None = None
    follow_up: str | None = None


class PolicyDecision(BaseModel):
    policy_name: str | None = None
    decision: str | None = None
    criteria: list[str] = Field(default_factory=list)
    exclusion: str | None = None


class ContractClauseSummary(BaseModel):
    clause_name: str | None = None
    obligation: str | None = None
    deadline: str | None = None
    penalty: str | None = None


class RiskFinding(BaseModel):
    risk_area: str | None = None
    finding: str | None = None
    severity: str | None = None
    mitigation: str | None = None
