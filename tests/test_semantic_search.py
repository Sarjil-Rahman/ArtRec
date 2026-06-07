from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from artrec.data.catalog import CatalogConfig, generate_catalog
from artrec.pipeline import build_end_to_end
from artrec.retrieval.semantic import SemanticCatalogRetriever


def test_build_creates_semantic_retriever_artifact(tmp_path):
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    result = build_end_to_end(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        n_items=70,
        n_artists=15,
        n_users=30,
        n_days=5,
        seed=11,
    )
    assert result["semantic_retriever_built"] is True
    assert (artifact_dir / "semantic_retriever.joblib").exists()


def test_semantic_search_query_filters_and_availability():
    catalog_df = generate_catalog(
        CatalogConfig(n_items=80, n_artists=15, random_seed=12)
    )
    retriever = SemanticCatalogRetriever.fit(catalog_df)

    results = retriever.search("blue abstract painting", k=5)
    assert 0 < len(results) <= 5
    assert "semantic_similarity" in results.columns
    assert (results["availability"] == 1).all()

    max_price = float(results["price"].median())
    price_filtered = retriever.search(
        "blue abstract painting", k=20, max_price=max_price
    )
    assert (price_filtered["price"] <= max_price).all()

    style = catalog_df["style"].iloc[0]
    style_filtered = retriever.search("painting", k=20, style=style)
    assert (style_filtered["style"].str.lower() == style.lower()).all()

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search("   ")


def test_semantic_search_can_include_unavailable_only_when_allowed():
    catalog_df = generate_catalog(
        CatalogConfig(n_items=30, n_artists=8, random_seed=13)
    )
    catalog_df.loc[0, "availability"] = 0
    catalog_df.loc[0, "description"] = (
        "unique unavailable blue abstract painting marker"
    )
    retriever = SemanticCatalogRetriever.fit(catalog_df)

    available = retriever.search(
        "unique unavailable marker", k=10, availability_only=True
    )
    assert catalog_df.loc[0, "item_id"] not in set(available["item_id"])

    all_results = retriever.search(
        "unique unavailable marker", k=10, availability_only=False
    )
    assert catalog_df.loc[0, "item_id"] in set(all_results["item_id"])


def test_semantic_search_does_not_label_zero_similarity_as_semantic_match():
    catalog_df = generate_catalog(
        CatalogConfig(n_items=30, n_artists=8, random_seed=14)
    )
    retriever = SemanticCatalogRetriever.fit(catalog_df)

    results = retriever.search("zzzzzz unmatchedtoken", k=5)

    assert len(results) <= 5
    assert (results["semantic_similarity"] == 0).all()
    assert not results["semantic_reason_codes"].str.contains("semantic_match").any()
