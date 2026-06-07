from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SEARCH_TEXT_COLUMNS = [
    "title",
    "artist_display_name",
    "style",
    "medium",
    "culture",
    "department",
    "classification",
    "object_name",
    "artist_nationality",
    "period",
    "tags",
    "dominant_hue",
    "palette",
    "description",
]

OUTPUT_COLUMNS = [
    "item_id",
    "title",
    "artist_display_name",
    "style",
    "medium",
    "price",
    "margin",
    "freshness",
    "availability",
    "tags",
    "dominant_hue",
    "palette",
    "semantic_similarity",
    "semantic_reason_codes",
]

MIN_SEMANTIC_MATCH_SCORE = 0.01


def _normalise_filter_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value.lower() if value else None


@dataclass
class SemanticCatalogRetriever:
    vectorizer: TfidfVectorizer
    text_matrix: object
    item_frame: pd.DataFrame
    text_columns: list[str]

    @classmethod
    def fit(cls, catalog_df: pd.DataFrame) -> "SemanticCatalogRetriever":
        item_frame = catalog_df.reset_index(drop=True).copy()
        text_columns = [col for col in SEARCH_TEXT_COLUMNS if col in item_frame.columns]
        searchable_text = cls._build_search_text(item_frame, text_columns)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        text_matrix = vectorizer.fit_transform(searchable_text)
        return cls(
            vectorizer=vectorizer,
            text_matrix=text_matrix,
            item_frame=item_frame,
            text_columns=text_columns,
        )

    @staticmethod
    def _build_search_text(df: pd.DataFrame, text_columns: list[str]) -> pd.Series:
        if not text_columns:
            return pd.Series([""] * len(df), index=df.index)
        return (
            df[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace("|", " ", regex=False)
        )

    def search(
        self,
        query: str,
        k: int = 12,
        max_price: float | None = None,
        min_price: float | None = None,
        style: str | None = None,
        medium: str | None = None,
        dominant_hue: str | None = None,
        availability_only: bool = True,
    ) -> pd.DataFrame:
        if query is None or not str(query).strip():
            raise ValueError("query must not be empty")

        work = self.item_frame.copy()
        mask = pd.Series(True, index=work.index)
        if availability_only and "availability" in work.columns:
            mask &= work["availability"].fillna(0).astype(int) == 1
        if min_price is not None and "price" in work.columns:
            mask &= work["price"].astype(float) >= float(min_price)
        if max_price is not None and "price" in work.columns:
            mask &= work["price"].astype(float) <= float(max_price)

        style_filter = _normalise_filter_value(style)
        if style_filter and "style" in work.columns:
            mask &= work["style"].fillna("").astype(str).str.lower() == style_filter
        medium_filter = _normalise_filter_value(medium)
        if medium_filter and "medium" in work.columns:
            mask &= work["medium"].fillna("").astype(str).str.lower() == medium_filter
        hue_filter = _normalise_filter_value(dominant_hue)
        if hue_filter and "dominant_hue" in work.columns:
            mask &= (
                work["dominant_hue"].fillna("").astype(str).str.lower() == hue_filter
            )

        available = work.loc[mask].copy()
        if available.empty:
            available["semantic_similarity"] = []
            available["semantic_reason_codes"] = []
            return available[
                [col for col in OUTPUT_COLUMNS if col in available.columns]
            ]

        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.text_matrix[available.index]).ravel()
        scores = np.clip(scores, 0.0, 1.0)
        available["semantic_similarity"] = scores
        positive = available["semantic_similarity"] >= MIN_SEMANTIC_MATCH_SCORE
        if positive.any():
            available = available.loc[positive].copy()
        available["semantic_reason_codes"] = available.apply(
            lambda row: self._reason_codes(row, query, max_price=max_price),
            axis=1,
        )
        available = available.sort_values("semantic_similarity", ascending=False).head(
            min(int(k), len(available))
        )
        return available[
            [col for col in OUTPUT_COLUMNS if col in available.columns]
        ].reset_index(drop=True)

    def _reason_codes(
        self, row: pd.Series, query: str, max_price: float | None = None
    ) -> str:
        reasons: list[str] = []
        query_terms = {term for term in query.lower().replace("|", " ").split() if term}
        row_text = " ".join(
            str(row.get(col, "")).lower().replace("|", " ") for col in self.text_columns
        )
        if query_terms and any(term in row_text for term in query_terms):
            reasons.append("query_match")
        if str(row.get("style", "")).lower() in query_terms:
            reasons.append("style_match")
        tags = str(row.get("tags", "")).lower().replace("|", " ").split()
        if query_terms.intersection(tags):
            reasons.append("tag_match")
        if str(row.get("dominant_hue", "")).lower() in query_terms:
            reasons.append("colour_match")
        if str(row.get("medium", "")).lower() in " ".join(query_terms):
            reasons.append("medium_match")
        if (
            max_price is not None
            and "price" in row
            and float(row["price"]) <= float(max_price)
        ):
            reasons.append("price_fit")
        if float(row.get("semantic_similarity", 0.0)) >= MIN_SEMANTIC_MATCH_SCORE:
            reasons.append("semantic_match")
        if not reasons:
            reasons.append("metadata_filter_match")
        return "|".join(dict.fromkeys(reasons))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "SemanticCatalogRetriever":
        return joblib.load(path)
