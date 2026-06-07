from __future__ import annotations
from fastapi import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

REQUEST_COUNT = Counter(
    "artrec_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "artrec_request_latency_seconds", "API request latency", ["endpoint"]
)
IN_PROGRESS = Gauge(
    "artrec_requests_in_progress", "Requests currently in progress", ["endpoint"]
)
AGENTIC_REQUESTS = Counter(
    "agentic_requests_total", "Total agentic document-intelligence requests", ["endpoint", "status"]
)
AGENTIC_LATENCY = Histogram(
    "agentic_request_latency_seconds", "Agentic request latency", ["endpoint"]
)
AGENTIC_RETRIEVAL_TOP_K = Histogram(
    "agentic_retrieval_top_k", "Requested top-k values for agentic retrieval"
)
AGENTIC_LOW_EVIDENCE = Counter(
    "agentic_low_evidence_total", "Agentic requests with low evidence"
)
AGENTIC_REFUSALS = Counter(
    "agentic_refusals_total", "Agentic requests that refused to answer"
)
AGENTIC_SCHEMA_VALIDATION_FAILURES = Counter(
    "agentic_schema_validation_failures_total", "Structured extraction validation failures"
)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
