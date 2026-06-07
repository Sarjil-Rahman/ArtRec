from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from artrec.agentic.ingestion import chunk_documents
from artrec.agentic.models import (
    ContractClauseSummary,
    Document,
    ExtractionResult,
    FieldCitation,
    PatientSummary,
    PolicyDecision,
    RiskFinding,
)


SCHEMAS: dict[str, type[BaseModel]] = {
    "patient_summary": PatientSummary,
    "policy_decision": PolicyDecision,
    "contract_clause": ContractClauseSummary,
    "risk_finding": RiskFinding,
}


def _find(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip(" .;\n")


class StructuredExtractor:
    def __init__(self, documents: list[Document]):
        self.documents = {doc.document_id: doc for doc in documents}

    def extract(self, document_id: str, schema_type: str) -> ExtractionResult:
        if document_id not in self.documents:
            raise KeyError(f"Unknown document_id: {document_id}")
        if schema_type not in SCHEMAS:
            raise ValueError(f"Unsupported schema_type: {schema_type}")

        doc = self.documents[document_id]
        chunk = chunk_documents([doc], chunk_size=900, overlap=0)[0]
        text = doc.text
        values = self._extract_values(schema_type, text)
        warnings = [f"missing field: {key}" for key, value in values.items() if value in (None, [], "")]
        citations = [
            FieldCitation(
                field=key,
                document_id=doc.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text[:260],
            )
            for key, value in values.items()
            if value not in (None, [], "")
        ]
        try:
            model = SCHEMAS[schema_type](**values)
            extracted: dict[str, Any] = model.model_dump()
        except ValidationError as exc:
            warnings.append(f"schema validation failed: {exc.errors()[0]['msg']}")
            extracted = values
        populated = sum(1 for value in values.values() if value not in (None, [], ""))
        confidence = populated / max(len(values), 1)
        return ExtractionResult(
            schema_type=schema_type,
            extracted=extracted,
            citations=citations,
            warnings=warnings,
            confidence=round(confidence, 3),
        )

    def _extract_values(self, schema_type: str, text: str) -> dict[str, Any]:
        if schema_type == "patient_summary":
            age = _find(r"Age:\s*(\d+)", text)
            return {
                "patient_name": _find(r"Patient:\s*([^.\n]+)", text),
                "age": int(age) if age else None,
                "diagnosis": _find(r"Diagnosis:\s*([^.\n]+)", text),
                "discharge_plan": _find(r"Discharge plan:\s*([^.\n]+)", text),
                "follow_up": _find(r"Follow-up:\s*([^.\n]+)", text),
            }
        if schema_type == "policy_decision":
            criteria = re.findall(r"-\s*([^\n]+)", text)
            return {
                "policy_name": _find(r"Policy:\s*([^.\n]+)", text),
                "decision": _find(r"Decision:\s*([^.\n]+)", text),
                "criteria": [item.strip() for item in criteria],
                "exclusion": _find(r"Exclusion:\s*([^.\n]+)", text),
            }
        if schema_type == "contract_clause":
            return {
                "clause_name": _find(r"Clause:\s*([^.\n]+)", text),
                "obligation": _find(r"Obligation:\s*([^.\n]+)", text),
                "deadline": _find(r"Deadline:\s*([^.\n]+)", text),
                "penalty": _find(r"Remedy:\s*([^.\n]+)", text),
            }
        return {
            "risk_area": _find(r"Risk area:\s*([^.\n]+)", text),
            "finding": _find(r"Finding:\s*([^.\n]+)", text),
            "severity": _find(r"Severity:\s*([^.\n]+)", text),
            "mitigation": _find(r"Mitigation:\s*([^.\n]+)", text),
        }
