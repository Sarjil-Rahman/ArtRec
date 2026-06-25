from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
import os
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from artrec.api.schemas import (
    AgenticAskRequest,
    AgenticAskResponse,
    AgenticEvaluateResponse,
    AgenticExtractRequest,
    AgenticExtractResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    SearchRequest,
    SearchResponse,
    SimilarItemsResponse,
)
from artrec.agentic.agent import AgenticRAGPipeline
from artrec.agentic.evaluation import evaluate_pipeline, load_gold
from artrec.agentic.ingestion import load_sample_documents
from artrec.data.io import read_csv_with_types
from artrec.monitoring.metrics import (
    AGENTIC_LATENCY,
    AGENTIC_LOW_EVIDENCE,
    AGENTIC_REFUSALS,
    AGENTIC_REQUESTS,
    AGENTIC_RETRIEVAL_TOP_K,
    AGENTIC_SCHEMA_VALIDATION_FAILURES,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    IN_PROGRESS,
    metrics_response,
)
from artrec.serve.recommender import RecommendationService


def _infer_history_cutoff(data_dir: Path):
    train_path = data_dir / "processed" / "train_features.csv"
    if not train_path.exists():
        return None
    train_df = read_csv_with_types(train_path)
    if train_df.empty or "timestamp" not in train_df.columns:
        return None
    return train_df["timestamp"].max()


def _load_service() -> RecommendationService:
    data_dir = Path(os.getenv("ARTREC_DATA_DIR", "data"))
    artifact_dir = Path(os.getenv("ARTREC_ARTIFACT_DIR", "artifacts"))
    required = [
        data_dir / "raw" / "catalog.csv",
        data_dir / "raw" / "users.csv",
        artifact_dir / "retriever.joblib",
        artifact_dir / "semantic_retriever.joblib",
        artifact_dir / "ranker.joblib",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing required ArtRec data/artifacts. Run `python scripts/build_all.py` first. "
            f"Missing: {', '.join(missing)}"
        )
    return RecommendationService.load(
        catalog_path=data_dir / "raw" / "catalog.csv",
        users_path=data_dir / "raw" / "users.csv",
        retriever_path=artifact_dir / "retriever.joblib",
        ranker_path=artifact_dir / "ranker.joblib",
        impressions_path=data_dir / "raw" / "impressions.csv",
        semantic_retriever_path=artifact_dir / "semantic_retriever.joblib",
        history_cutoff=_infer_history_cutoff(data_dir),
    )


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("ARTREC_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


service: RecommendationService | None = None
agentic_pipeline: AgenticRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = _load_service()
    yield


app = FastAPI(title="ArtRec Production-Style API", version="0.1.0", lifespan=lifespan)


def _load_agentic_pipeline() -> AgenticRAGPipeline:
    document_dir = Path(os.getenv("ARTREC_DOCUMENT_DIR", "data/documents/sample"))
    return AgenticRAGPipeline(load_sample_documents(document_dir))


def get_agentic_pipeline() -> AgenticRAGPipeline:
    global agentic_pipeline
    if agentic_pipeline is None:
        agentic_pipeline = _load_agentic_pipeline()
    return agentic_pipeline


def get_service() -> RecommendationService:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not loaded yet.",
        )
    return service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics():
    return metrics_response()


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    dependencies=[Depends(require_api_key)],
)
def recommend(req: RecommendationRequest) -> RecommendationResponse:
    endpoint = "/recommend"
    IN_PROGRESS.labels(endpoint=endpoint).inc()
    started = perf_counter()
    try:
        svc = get_service()
        recs = svc.recommend(
            user_id=req.user_id,
            limit=req.limit,
            query=req.query,
            max_price=req.max_price,
            min_price=req.min_price,
            style=req.style,
            medium=req.medium,
            dominant_hue=req.dominant_hue,
            availability_only=req.availability_only,
        )
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="200").inc()
        return RecommendationResponse(
            user_id=req.user_id,
            count=len(recs),
            recommendations=recs.to_dict(orient="records"),
        )
    except KeyError as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="404").inc()
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint, method="POST", status=str(exc.status_code)
        ).inc()
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - started)
        IN_PROGRESS.labels(endpoint=endpoint).dec()


@app.post(
    "/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)]
)
def search(req: SearchRequest) -> SearchResponse:
    endpoint = "/search"
    IN_PROGRESS.labels(endpoint=endpoint).inc()
    started = perf_counter()
    try:
        svc = get_service()
        results = svc.search(
            query=req.query,
            limit=req.limit,
            max_price=req.max_price,
            min_price=req.min_price,
            style=req.style,
            medium=req.medium,
            dominant_hue=req.dominant_hue,
            availability_only=req.availability_only,
        )
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="200").inc()
        return SearchResponse(
            query=req.query,
            count=len(results),
            results=results.to_dict(orient="records"),
        )
    except ValueError as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="400").inc()
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint, method="POST", status=str(exc.status_code)
        ).inc()
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - started)
        IN_PROGRESS.labels(endpoint=endpoint).dec()


@app.post(
    "/feedback/not-interested",
    response_model=FeedbackResponse,
    dependencies=[Depends(require_api_key)],
)
def feedback_not_interested(req: FeedbackRequest) -> FeedbackResponse:
    endpoint = "/feedback/not-interested"
    IN_PROGRESS.labels(endpoint=endpoint).inc()
    started = perf_counter()
    try:
        svc = get_service()
        result = svc.record_not_interested(
            user_id=req.user_id,
            item_id=req.item_id,
            reason=req.reason,
        )
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="200").inc()
        return FeedbackResponse(**result)
    except KeyError as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="404").inc()
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint, method="POST", status=str(exc.status_code)
        ).inc()
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - started)
        IN_PROGRESS.labels(endpoint=endpoint).dec()


@app.get(
    "/similar/{item_id}",
    response_model=SimilarItemsResponse,
    dependencies=[Depends(require_api_key)],
)
def similar(
    item_id: str, limit: int = Query(default=8, ge=1, le=50)
) -> SimilarItemsResponse:
    endpoint = "/similar"
    IN_PROGRESS.labels(endpoint=endpoint).inc()
    started = perf_counter()
    try:
        svc = get_service()
        sims = svc.similar_items(item_id=item_id, k=limit)
        REQUEST_COUNT.labels(endpoint=endpoint, method="GET", status="200").inc()
        return SimilarItemsResponse(
            item_id=item_id,
            count=len(sims),
            similar_items=sims.to_dict(orient="records"),
        )
    except KeyError as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="GET", status="404").inc()
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint, method="GET", status=str(exc.status_code)
        ).inc()
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method="GET", status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - started)
        IN_PROGRESS.labels(endpoint=endpoint).dec()


@app.post(
    "/agentic/ask",
    response_model=AgenticAskResponse,
    dependencies=[Depends(require_api_key)],
)
def agentic_ask(req: AgenticAskRequest) -> AgenticAskResponse:
    endpoint = "/agentic/ask"
    started = perf_counter()
    AGENTIC_RETRIEVAL_TOP_K.observe(req.top_k)
    try:
        pipeline = get_agentic_pipeline()
        result = pipeline.ask(
            query=req.query,
            top_k=req.top_k,
            domain=req.domain,
            require_citations=req.require_citations,
        )
        if result.evidence_score < 0.12:
            AGENTIC_LOW_EVIDENCE.inc()
        if result.route == "unanswerable":
            AGENTIC_REFUSALS.inc()
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="200").inc()
        return AgenticAskResponse(
            answer=result.final_answer,
            citations=[citation.model_dump() for citation in result.cited_chunks],
            evidence_score=result.evidence_score,
            route=result.route,
            warnings=result.warnings,
            trace=[step.model_dump() for step in result.trace],
        )
    except Exception as exc:
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        AGENTIC_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - started)


@app.post(
    "/agentic/extract",
    response_model=AgenticExtractResponse,
    dependencies=[Depends(require_api_key)],
)
def agentic_extract(req: AgenticExtractRequest) -> AgenticExtractResponse:
    endpoint = "/agentic/extract"
    started = perf_counter()
    try:
        result = get_agentic_pipeline().extractor.extract(req.document_id, req.schema_type)
        if any("schema validation failed" in warning for warning in result.warnings):
            AGENTIC_SCHEMA_VALIDATION_FAILURES.inc()
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="200").inc()
        return AgenticExtractResponse(
            schema_type=result.schema_type,
            extracted=result.extracted,
            citations=[citation.model_dump() for citation in result.citations],
            warnings=result.warnings,
        )
    except KeyError as exc:
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="404").inc()
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="400").inc()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        AGENTIC_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - started)


@app.post(
    "/agentic/evaluate",
    response_model=AgenticEvaluateResponse,
    dependencies=[Depends(require_api_key)],
)
def agentic_evaluate() -> AgenticEvaluateResponse:
    endpoint = "/agentic/evaluate"
    started = perf_counter()
    try:
        gold_path = Path(os.getenv("ARTREC_AGENTIC_GOLD", "data/eval/rag_gold.jsonl"))
        metrics, _predictions, failures = evaluate_pipeline(
            get_agentic_pipeline(), load_gold(gold_path)
        )
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="200").inc()
        return AgenticEvaluateResponse(metrics=metrics, failure_count=len(failures))
    except Exception as exc:
        AGENTIC_REQUESTS.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        AGENTIC_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - started)
