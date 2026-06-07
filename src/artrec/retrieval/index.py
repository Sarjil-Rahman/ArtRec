from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _matrix_from_embeddings(series: pd.Series) -> np.ndarray:
    return np.vstack(series.apply(lambda x: np.array(x, dtype=float)).tolist())


@dataclass
class CatalogRetriever:
    model: NearestNeighbors
    item_frame: pd.DataFrame

    @classmethod
    def fit(cls, catalog_df: pd.DataFrame, n_neighbors: int = 50) -> "CatalogRetriever":
        matrix = _matrix_from_embeddings(catalog_df["embedding"])
        model = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(catalog_df)), metric="cosine"
        )
        model.fit(matrix)
        return cls(model=model, item_frame=catalog_df.reset_index(drop=True).copy())

    def get_similar_items(self, item_id: str, k: int = 8) -> pd.DataFrame:
        if item_id not in set(self.item_frame["item_id"]):
            raise KeyError(f"Unknown item_id: {item_id}")
        idx = self.item_frame.index[self.item_frame["item_id"] == item_id][0]
        item_vec = np.array(self.item_frame.loc[idx, "embedding"], dtype=float).reshape(
            1, -1
        )
        distances, indices = self.model.kneighbors(
            item_vec, n_neighbors=min(k + 1, len(self.item_frame))
        )
        out = self.item_frame.iloc[indices[0]].copy()
        out["retrieval_similarity"] = 1.0 - distances[0]
        out = out[out["item_id"] != item_id].head(k)
        return out[
            [
                "item_id",
                "title",
                "artist_display_name",
                "style",
                "price",
                "freshness",
                "retrieval_similarity",
            ]
        ].reset_index(drop=True)

    def retrieve_for_user(self, user_vector: np.ndarray, k: int = 100) -> pd.DataFrame:
        distances, indices = self.model.kneighbors(
            np.array(user_vector, dtype=float).reshape(1, -1),
            n_neighbors=min(k, len(self.item_frame)),
        )
        out = self.item_frame.iloc[indices[0]].copy()
        out["retrieval_similarity"] = 1.0 - distances[0]
        return out.reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "CatalogRetriever":
        return joblib.load(path)
