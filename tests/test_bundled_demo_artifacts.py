from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from artrec.api import main as api_main
from artrec.serve.recommender import RecommendationService


ROOT = Path(__file__).resolve().parents[1]


def load_bundled_service() -> RecommendationService:
    return RecommendationService.load(
        catalog_path=ROOT / "data" / "raw" / "catalog.csv",
        users_path=ROOT / "data" / "raw" / "users.csv",
        impressions_path=ROOT / "data" / "raw" / "impressions.csv",
        retriever_path=ROOT / "artifacts" / "retriever.joblib",
        ranker_path=ROOT / "artifacts" / "ranker.joblib",
        semantic_retriever_path=ROOT / "artifacts" / "semantic_retriever.joblib",
    )


def test_bundled_artifacts_load_and_recommend_for_demo_user():
    service = load_bundled_service()

    recs = service.recommend("user_0000", limit=5)

    assert len(recs) == 5
    expected_fields = {
        "item_id",
        "title",
        "artist_display_name",
        "style",
        "price",
        "model_score",
        "business_score",
        "final_rank",
    }
    assert expected_fields.issubset(recs.columns)
    assert recs["item_id"].notna().all()


def test_api_uses_bundled_artifacts_in_public_demo_mode(monkeypatch):
    monkeypatch.delenv("ARTREC_API_KEY", raising=False)
    api_main.service = load_bundled_service()
    api_main.agentic_pipeline = None
    client = TestClient(api_main.app)

    assert client.get("/health").json() == {"status": "ok"}

    rec = client.post("/recommend", json={"user_id": "user_0000", "limit": 5})
    assert rec.status_code == 200
    body = rec.json()
    assert body["count"] == 5
    assert body["recommendations"][0]["item_id"]

    search = client.post("/search", json={"query": "blue abstract painting", "limit": 3})
    assert search.status_code == 200
    assert search.json()["count"] <= 3

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "artrec_requests_total" in metrics.text
