from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import importlib.metadata

import numpy as np
import pandas as pd

from artrec.data.io import read_csv_with_types
from artrec.models.ranker import ArtRanker
from artrec.rerank.business import apply_business_rerank
from artrec.retrieval.index import CatalogRetriever
from artrec.retrieval.semantic import SemanticCatalogRetriever


PINNED_ARTIFACT_ENVIRONMENT = (
    "Python >=3.10,<3.11 with numpy==2.2.6, pandas==2.3.3, "
    "scikit-learn==1.7.2, and joblib==1.5.3"
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _load_artifact(path: Path, loader, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} artifact at {path}. Run `python scripts/build_all.py` "
            "or restore the bundled demo artifacts before starting the API."
        )
    try:
        return loader(path)
    except Exception as exc:
        versions = (
            f"numpy={_package_version('numpy')}, "
            f"pandas={_package_version('pandas')}, "
            f"scikit-learn={_package_version('scikit-learn')}, "
            f"joblib={_package_version('joblib')}"
        )
        raise RuntimeError(
            f"Could not load {label} artifact at {path}. The bundled joblib files "
            f"are intended for {PINNED_ARTIFACT_ENVIRONMENT}. Current packages: "
            f"{versions}. Install with `python -m pip install -r requirements.txt` "
            "or rebuild artifacts with the environment you plan to serve."
        ) from exc


@dataclass
class ServiceAssets:
    catalog_df: pd.DataFrame
    users_df: pd.DataFrame
    impressions_df: pd.DataFrame | None
    retriever: CatalogRetriever
    ranker: ArtRanker
    semantic_retriever: SemanticCatalogRetriever | None = None


@dataclass
class RecommendationService:
    assets: ServiceAssets
    runtime_not_interested: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.catalog = self.assets.catalog_df.set_index("item_id", drop=False)
        self.users = self.assets.users_df.set_index("user_id", drop=False)

    @classmethod
    def load(
        cls,
        catalog_path: Path,
        users_path: Path,
        retriever_path: Path,
        ranker_path: Path,
        impressions_path: Path | None = None,
        semantic_retriever_path: Path | None = None,
    ):
        for label, path in {
            "catalog data": catalog_path,
            "users data": users_path,
        }.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {label} at {path}. Run `python scripts/build_all.py` "
                    "or restore the bundled demo data."
                )

        catalog_df = read_csv_with_types(catalog_path)
        users_df = read_csv_with_types(users_path)
        impressions_df = (
            read_csv_with_types(impressions_path)
            if impressions_path and impressions_path.exists()
            else None
        )
        retriever = _load_artifact(
            retriever_path, CatalogRetriever.load, "nearest-neighbour retriever"
        )
        ranker = _load_artifact(ranker_path, ArtRanker.load, "ranking model")
        semantic_retriever = (
            _load_artifact(
                semantic_retriever_path,
                SemanticCatalogRetriever.load,
                "semantic retriever",
            )
            if semantic_retriever_path and semantic_retriever_path.exists()
            else None
        )
        return cls(
            ServiceAssets(
                catalog_df,
                users_df,
                impressions_df,
                retriever,
                ranker,
                semantic_retriever,
            )
        )

    def _get_history_counts(self, user_id: str) -> dict[str, dict]:
        if self.assets.impressions_df is not None:
            hist = self.assets.impressions_df[
                self.assets.impressions_df["user_id"] == user_id
            ].copy()
        else:
            hist = pd.DataFrame(
                columns=["item_id", "artist_id", "style", "not_interested"]
            )

        item_seen = hist.groupby("item_id").size().to_dict()
        artist_seen = (
            hist.groupby("artist_id").size().to_dict() if "artist_id" in hist else {}
        )
        style_seen = hist.groupby("style").size().to_dict() if "style" in hist else {}

        if "not_interested" in hist.columns:
            negative_hist = hist[hist["not_interested"].fillna(0).astype(int) == 1]
        else:
            negative_hist = hist.iloc[0:0]
        runtime = pd.DataFrame(
            [row for row in self.runtime_not_interested if row["user_id"] == user_id]
        )
        if not runtime.empty:
            negative_hist = pd.concat([negative_hist, runtime], ignore_index=True)

        item_not = negative_hist.groupby("item_id").size().to_dict()
        artist_not = (
            negative_hist.groupby("artist_id").size().to_dict()
            if "artist_id" in negative_hist
            else {}
        )
        style_not = (
            negative_hist.groupby("style").size().to_dict()
            if "style" in negative_hist
            else {}
        )
        return {
            "item_seen": item_seen,
            "artist_seen": artist_seen,
            "style_seen": style_seen,
            "item_not": item_not,
            "artist_not": artist_not,
            "style_not": style_not,
        }

    def _apply_catalog_filters(
        self,
        candidates: pd.DataFrame,
        max_price: float | None = None,
        min_price: float | None = None,
        style: str | None = None,
        medium: str | None = None,
        dominant_hue: str | None = None,
        availability_only: bool = True,
    ) -> pd.DataFrame:
        work = candidates.copy()
        mask = pd.Series(True, index=work.index)
        if availability_only and "availability" in work:
            mask &= work["availability"].fillna(0).astype(int) == 1
        if min_price is not None:
            mask &= work["price"].astype(float) >= float(min_price)
        if max_price is not None:
            mask &= work["price"].astype(float) <= float(max_price)
        if style:
            mask &= (
                work["style"].fillna("").astype(str).str.lower()
                == style.strip().lower()
            )
        if medium:
            mask &= (
                work["medium"].fillna("").astype(str).str.lower()
                == medium.strip().lower()
            )
        if dominant_hue:
            mask &= (
                work["dominant_hue"].fillna("").astype(str).str.lower()
                == dominant_hue.strip().lower()
            )
        return work.loc[mask].copy()

    def _build_personalized_candidates(
        self,
        user_id: str,
        limit: int,
        max_price: float | None = None,
        min_price: float | None = None,
        style: str | None = None,
        medium: str | None = None,
        dominant_hue: str | None = None,
        availability_only: bool = True,
    ) -> pd.DataFrame:
        user_row = self.users.loc[user_id]
        user_vector = np.array(user_row["taste_vector"], dtype=float)
        candidates = self.assets.retriever.retrieve_for_user(
            user_vector=user_vector, k=limit
        ).copy()
        candidates = self._apply_catalog_filters(
            candidates,
            max_price=max_price,
            min_price=min_price,
            style=style,
            medium=medium,
            dominant_hue=dominant_hue,
            availability_only=availability_only,
        )
        candidates["semantic_similarity"] = 0.0
        candidates["combined_retrieval_score"] = candidates["retrieval_similarity"]
        return candidates

    def _build_hybrid_candidates(
        self,
        user_id: str,
        query: str,
        limit: int,
        max_price: float | None = None,
        min_price: float | None = None,
        style: str | None = None,
        medium: str | None = None,
        dominant_hue: str | None = None,
        availability_only: bool = True,
    ) -> pd.DataFrame:
        if self.assets.semantic_retriever is None:
            raise RuntimeError(
                "Semantic retriever is not loaded. Run `python scripts/build_all.py` first."
            )
        personalized = self._build_personalized_candidates(
            user_id=user_id,
            limit=limit,
            max_price=max_price,
            min_price=min_price,
            style=style,
            medium=medium,
            dominant_hue=dominant_hue,
            availability_only=availability_only,
        )
        semantic = self.assets.semantic_retriever.search(
            query=query,
            k=limit,
            max_price=max_price,
            min_price=min_price,
            style=style,
            medium=medium,
            dominant_hue=dominant_hue,
            availability_only=availability_only,
        )
        semantic = semantic[["item_id", "semantic_similarity"]].merge(
            self.assets.catalog_df, on="item_id", how="left"
        )
        semantic["retrieval_similarity"] = 0.0

        merged = pd.concat([personalized, semantic], ignore_index=True, sort=False)
        merged["retrieval_similarity"] = merged["retrieval_similarity"].fillna(0.0)
        merged["semantic_similarity"] = merged["semantic_similarity"].fillna(0.0)
        merged = (
            merged.sort_values(
                ["item_id", "retrieval_similarity", "semantic_similarity"],
                ascending=[True, False, False],
            )
            .groupby("item_id", as_index=False)
            .agg(
                {
                    **{
                        col: "first"
                        for col in merged.columns
                        if col
                        not in {
                            "item_id",
                            "retrieval_similarity",
                            "semantic_similarity",
                        }
                    },
                    "retrieval_similarity": "max",
                    "semantic_similarity": "max",
                }
            )
        )
        merged["combined_retrieval_score"] = (
            0.65 * merged["retrieval_similarity"] + 0.35 * merged["semantic_similarity"]
        )
        return merged.sort_values("combined_retrieval_score", ascending=False)

    def _build_user_context_features(
        self, user_id: str, candidates: pd.DataFrame, hybrid_mode: bool = False
    ) -> pd.DataFrame:
        user_row = self.users.loc[user_id]
        history = self._get_history_counts(user_id)
        work = candidates.copy().reset_index(drop=True)
        work["position"] = range(len(work))
        work["popularity_score_pre"] = work["popularity_prior"]
        work["retrieval_similarity"] = work.get("retrieval_similarity", 0.0)
        work["semantic_similarity"] = work.get("semantic_similarity", 0.0)
        work["affinity_score"] = (
            work["combined_retrieval_score"]
            if hybrid_mode
            else work["retrieval_similarity"]
        )
        work["budget_mean"] = float(user_row["budget_mean"])
        work["price_sensitivity"] = float(user_row["price_sensitivity"])
        work["novelty_preference"] = float(user_row["novelty_preference"])
        work["repeat_tolerance"] = float(user_row["repeat_tolerance"])
        work["activity_level"] = float(user_row["activity_level"])
        work["conversion_propensity"] = float(user_row["conversion_propensity"])
        work["save_propensity"] = float(user_row["save_propensity"])
        work["price_gap_ratio"] = (work["price"] - work["budget_mean"]) / work[
            "budget_mean"
        ].clip(lower=1.0)
        work["price_above_budget"] = (work["price"] > work["budget_mean"]).astype(int)
        work["margin_to_price"] = work["margin"] / work["price"].clip(lower=1.0)
        work["top_3_position"] = (work["position"] <= 2).astype(int)
        work["top_5_position"] = (work["position"] <= 4).astype(int)
        work["user_item_seen_before"] = (
            work["item_id"].map(history["item_seen"]).fillna(0).astype(int)
        )
        work["user_artist_seen_before"] = (
            work["artist_id"].map(history["artist_seen"]).fillna(0).astype(int)
        )
        work["user_style_seen_before"] = (
            work["style"].map(history["style_seen"]).fillna(0).astype(int)
        )
        work["user_item_not_interested_before"] = (
            work["item_id"].map(history["item_not"]).fillna(0).astype(int)
        )
        work["user_artist_not_interested_before"] = (
            work["artist_id"].map(history["artist_not"]).fillna(0).astype(int)
        )
        work["user_style_not_interested_before"] = (
            work["style"].map(history["style_not"]).fillna(0).astype(int)
        )
        work["session_rank_inverse"] = 1.0 / (work["position"] + 1)
        work["tag_count"] = (
            work["tags"]
            .fillna("")
            .apply(lambda x: len([t for t in str(x).split("|") if t]))
        )
        work["has_blue_tag"] = (
            work["tags"].fillna("").apply(lambda x: int("blue" in str(x).split("|")))
        )
        work["has_nature_tag"] = (
            work["tags"].fillna("").apply(lambda x: int("nature" in str(x).split("|")))
        )
        work["surface"] = "home_feed"
        work["device"] = str(user_row["device_pref"])
        work["country"] = str(user_row["country"])
        work["segment"] = str(user_row["segment"])
        work["hour"] = 20
        work["day_of_week"] = 5
        work["is_weekend"] = 1
        return work

    def _build_candidate_frame(self, user_id: str, limit: int = 120) -> pd.DataFrame:
        candidates = self._build_personalized_candidates(user_id=user_id, limit=limit)
        return self._build_user_context_features(user_id, candidates, hybrid_mode=False)

    def recommend(
        self,
        user_id: str,
        limit: int = 12,
        query: str | None = None,
        max_price: float | None = None,
        min_price: float | None = None,
        style: str | None = None,
        medium: str | None = None,
        dominant_hue: str | None = None,
        availability_only: bool = True,
    ) -> pd.DataFrame:
        if user_id not in self.users.index:
            raise KeyError(f"Unknown user_id: {user_id}")

        has_filters = (
            any(
                value is not None
                for value in [max_price, min_price, style, medium, dominant_hue]
            )
            or availability_only is False
        )
        retrieval_limit = max(120, limit * 10)
        if query is None and not has_filters:
            candidates = self._build_candidate_frame(
                user_id=user_id, limit=retrieval_limit
            )
            hybrid_mode = False
        elif query is None:
            raw_candidates = self._build_personalized_candidates(
                user_id=user_id,
                limit=max(300, retrieval_limit),
                max_price=max_price,
                min_price=min_price,
                style=style,
                medium=medium,
                dominant_hue=dominant_hue,
                availability_only=availability_only,
            )
            candidates = self._build_user_context_features(
                user_id, raw_candidates, hybrid_mode=False
            )
            hybrid_mode = False
        else:
            raw_candidates = self._build_hybrid_candidates(
                user_id=user_id,
                query=query,
                limit=max(120, limit * 10),
                max_price=max_price,
                min_price=min_price,
                style=style,
                medium=medium,
                dominant_hue=dominant_hue,
                availability_only=availability_only,
            )
            candidates = self._build_user_context_features(
                user_id, raw_candidates, hybrid_mode=True
            )
            hybrid_mode = True

        if candidates.empty:
            return pd.DataFrame()
        candidates["model_score"] = self.assets.ranker.score(candidates)
        reranked = apply_business_rerank(
            candidates, score_col="model_score", top_k=limit
        )
        keep = [
            "item_id",
            "title",
            "artist_display_name",
            "style",
            "medium",
            "price",
            "margin",
            "freshness",
            "retrieval_similarity",
            "semantic_similarity",
            "model_score",
            "business_score",
            "reason_codes",
            "final_rank",
        ]
        if not hybrid_mode:
            reranked["semantic_similarity"] = reranked.get("semantic_similarity", 0.0)
        return reranked[[col for col in keep if col in reranked.columns]].copy()

    def search(
        self,
        query: str,
        limit: int = 12,
        max_price: float | None = None,
        min_price: float | None = None,
        style: str | None = None,
        medium: str | None = None,
        dominant_hue: str | None = None,
        availability_only: bool = True,
    ) -> pd.DataFrame:
        if self.assets.semantic_retriever is None:
            raise RuntimeError(
                "Semantic retriever is not loaded. Run `python scripts/build_all.py` first."
            )
        return self.assets.semantic_retriever.search(
            query=query,
            k=limit,
            max_price=max_price,
            min_price=min_price,
            style=style,
            medium=medium,
            dominant_hue=dominant_hue,
            availability_only=availability_only,
        )

    def record_not_interested(
        self, user_id: str, item_id: str, reason: str | None = None
    ) -> dict:
        if user_id not in self.users.index:
            raise KeyError(f"Unknown user_id: {user_id}")
        if item_id not in self.catalog.index:
            raise KeyError(f"Unknown item_id: {item_id}")
        item = self.catalog.loc[item_id]
        event = {
            "user_id": user_id,
            "item_id": item_id,
            "artist_id": item["artist_id"],
            "style": item["style"],
            "reason": reason,
            "not_interested": 1,
        }
        self.runtime_not_interested.append(event)
        return {
            "status": "ok",
            "user_id": user_id,
            "item_id": item_id,
            "reason": reason,
            "message": "Not-interested feedback recorded in memory.",
        }

    def similar_items(self, item_id: str, k: int = 8) -> pd.DataFrame:
        return self.assets.retriever.get_similar_items(item_id=item_id, k=k)
