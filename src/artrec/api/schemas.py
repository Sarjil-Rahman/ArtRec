from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any


class HealthResponse(BaseModel):
    status: str


class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = Field(default=12, ge=1, le=50)
    query: str | None = None
    max_price: float | None = Field(default=None, gt=0)
    min_price: float | None = Field(default=None, ge=0)
    style: str | None = None
    medium: str | None = None
    dominant_hue: str | None = None
    availability_only: bool = True

    @model_validator(mode="after")
    def validate_price_range(self):
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must be less than or equal to max_price")
        return self


class RecommendationResponse(BaseModel):
    user_id: str
    count: int
    recommendations: List[Dict[str, Any]]


class SimilarItemsResponse(BaseModel):
    item_id: str
    count: int
    similar_items: List[Dict[str, Any]]


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=12, ge=1, le=50)
    max_price: float | None = Field(default=None, gt=0)
    min_price: float | None = Field(default=None, ge=0)
    style: str | None = None
    medium: str | None = None
    dominant_hue: str | None = None
    availability_only: bool = True

    @model_validator(mode="after")
    def validate_price_range(self):
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must be less than or equal to max_price")
        return self


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    reason: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    user_id: str
    item_id: str
    reason: str | None = None
    message: str


class AgenticAskRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    domain: str = Field(default="all", pattern="^(medical|legal|insurance|compliance|support|all)$")
    require_citations: bool = True


class AgenticAskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    evidence_score: float
    route: str
    warnings: List[str]
    trace: List[Dict[str, Any]]


class AgenticExtractRequest(BaseModel):
    document_id: str
    schema_type: str = Field(
        pattern="^(patient_summary|policy_decision|contract_clause|risk_finding)$"
    )


class AgenticExtractResponse(BaseModel):
    schema_type: str
    extracted: Dict[str, Any]
    citations: List[Dict[str, Any]]
    warnings: List[str]


class AgenticEvaluateResponse(BaseModel):
    metrics: Dict[str, float]
    failure_count: int
