from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from artrec.api import main as api_main
from artrec.pipeline import build_end_to_end
from artrec.serve.recommender import RecommendationService


def _build_service(tmp_path):
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    build_end_to_end(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        n_items=80,
        n_artists=20,
        n_users=40,
        n_days=6,
        seed=3,
    )
    return RecommendationService.load(
        catalog_path=data_dir / "raw" / "catalog.csv",
        users_path=data_dir / "raw" / "users.csv",
        retriever_path=artifact_dir / "retriever.joblib",
        ranker_path=artifact_dir / "ranker.joblib",
        impressions_path=data_dir / "raw" / "impressions.csv",
        semantic_retriever_path=artifact_dir / "semantic_retriever.joblib",
    )


def test_service_smoke(tmp_path):
    service = _build_service(tmp_path)
    recs = service.recommend("user_0000", limit=5)
    assert len(recs) == 5


def test_api_endpoints_and_validation(tmp_path):
    api_main.service = _build_service(tmp_path)
    api_main.agentic_pipeline = None
    client = TestClient(api_main.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/metrics").status_code == 200

    rec = client.post("/recommend", json={"user_id": "user_0000", "limit": 5})
    assert rec.status_code == 200
    assert rec.json()["count"] == 5

    search = client.post(
        "/search",
        json={"query": "blue abstract painting", "limit": 4, "max_price": 500},
    )
    assert search.status_code == 200
    assert search.json()["count"] <= 4
    assert "semantic_similarity" in search.json()["results"][0]

    empty_search = client.post("/search", json={"query": "   ", "limit": 4})
    assert empty_search.status_code in {400, 422}

    hybrid = client.post(
        "/recommend",
        json={
            "user_id": "user_0000",
            "query": "minimalist blue print",
            "limit": 5,
            "max_price": 500,
        },
    )
    assert hybrid.status_code == 200
    assert hybrid.json()["count"] <= 5

    feedback = client.post(
        "/feedback/not-interested",
        json={
            "user_id": "user_0000",
            "item_id": rec.json()["recommendations"][0]["item_id"],
            "reason": "too_expensive",
        },
    )
    assert feedback.status_code == 200
    assert len(api_main.service.runtime_not_interested) == 1

    for bad_limit in [-1, 0, 51]:
        response = client.post(
            "/recommend", json={"user_id": "user_0000", "limit": bad_limit}
        )
        assert response.status_code == 422

    unknown = client.post("/recommend", json={"user_id": "missing", "limit": 5})
    assert unknown.status_code == 404

    similar = client.get("/similar/item_0000?limit=5")
    assert similar.status_code == 200
    assert similar.json()["count"] == 5

    for url in ["/similar/item_0000?limit=0", "/similar/item_0000?limit=51"]:
        assert client.get(url).status_code == 422

    assert client.get("/similar/not_real?limit=5").status_code == 404

    ask = client.post(
        "/agentic/ask",
        json={
            "query": "What follow-up was recommended for Jordan Ellis?",
            "top_k": 3,
            "domain": "medical",
            "require_citations": True,
        },
    )
    assert ask.status_code == 200
    assert ask.json()["citations"]
    assert ask.json()["route"] == "patient_summary"

    extract = client.post(
        "/agentic/extract",
        json={"document_id": "doc_med_001", "schema_type": "patient_summary"},
    )
    assert extract.status_code == 200
    assert extract.json()["extracted"]["patient_name"] == "Jordan Ellis"

    evaluation = client.post("/agentic/evaluate")
    assert evaluation.status_code == 200
    assert "recall_at_k" in evaluation.json()["metrics"]

    metrics = client.get("/metrics").text
    assert "agentic_requests_total" in metrics


def test_api_key_auth_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTREC_API_KEY", "secret")
    api_main.service = _build_service(tmp_path)
    client = TestClient(api_main.app)

    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert (
        client.post("/recommend", json={"user_id": "user_0000", "limit": 3}).status_code
        == 401
    )
    assert (
        client.post("/search", json={"query": "blue painting", "limit": 3}).status_code
        == 401
    )
    assert (
        client.post(
            "/feedback/not-interested",
            json={"user_id": "user_0000", "item_id": "item_0000"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/recommend",
            json={"user_id": "user_0000", "limit": 3},
            headers={"X-API-Key": "secret"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/search",
            json={"query": "blue painting", "limit": 3},
            headers={"X-API-Key": "secret"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/feedback/not-interested",
            json={"user_id": "user_0000", "item_id": "item_0000"},
            headers={"X-API-Key": "secret"},
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/similar/item_0000?limit=3", headers={"X-API-Key": "secret"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/agentic/ask",
            json={"query": "What follow-up was recommended for Jordan Ellis?"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/agentic/extract",
            json={"document_id": "doc_med_001", "schema_type": "patient_summary"},
        ).status_code
        == 401
    )
    assert client.post("/agentic/evaluate").status_code == 401
    assert (
        client.post(
            "/agentic/ask",
            json={"query": "What follow-up was recommended for Jordan Ellis?"},
            headers={"X-API-Key": "secret"},
        ).status_code
        == 200
    )
