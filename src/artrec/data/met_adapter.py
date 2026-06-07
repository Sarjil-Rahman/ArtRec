from __future__ import annotations
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from artrec.data.catalog import LATENT_DIM
from artrec.data.io import read_csv_with_types

MET_REQUIRED_COLUMNS = [
    "objectID",
    "title",
    "artistDisplayName",
    "department",
    "classification",
    "culture",
    "artistNationality",
    "medium",
    "objectDate",
    "isPublicDomain",
]


def _to_embedding(style: str, seed: int) -> list[float]:
    style_bank = {"Paintings": 0, "Drawings": 1, "Prints": 2, "Photographs": 3}
    vec = np.zeros(LATENT_DIM)
    idx = style_bank.get(style, 4) % LATENT_DIM
    vec[idx] = 1.0
    rnd = np.random.default_rng(seed)
    vec = vec + rnd.normal(0, 0.4, LATENT_DIM)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec.tolist()


def adapt_met_objects(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MET_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Met columns: {missing}")
    rows = []
    for _, row in df.reset_index(drop=True).iterrows():
        object_id = int(row["objectID"])
        tags_raw = row.get("tags") or []
        if isinstance(tags_raw, str):
            try:
                parsed = ast.literal_eval(tags_raw)
                if isinstance(parsed, list):
                    tags = [
                        str(x.get("term", x)) if isinstance(x, dict) else str(x)
                        for x in parsed
                    ]
                else:
                    tags = [str(tags_raw)]
            except Exception:
                tags = [
                    part.strip() for part in str(tags_raw).split("|") if part.strip()
                ]
        elif isinstance(tags_raw, list):
            tags = [
                str(x.get("term", x)) if isinstance(x, dict) else str(x)
                for x in tags_raw
            ]
        else:
            tags = []
        price = float(180 + (object_id % 350) * 2.7)
        margin_pct = 0.28 + ((object_id % 7) * 0.03)
        rows.append(
            {
                "object_id": object_id,
                "item_id": f"met_{object_id}",
                "artist_id": f"artist_{object_id % 1000:03d}",
                "artist_display_name": row.get("artistDisplayName") or "Unknown Artist",
                "title": row.get("title") or "Untitled",
                "style": row.get("classification") or "Paintings",
                "medium": row.get("medium") or "Unknown medium",
                "culture": row.get("culture") or "Unknown",
                "department": row.get("department") or "Unknown department",
                "classification": row.get("classification") or "Paintings",
                "object_date": str(row.get("objectDate") or "Unknown"),
                "price": round(price, 2),
                "quality": round(0.2 + ((object_id % 11) - 5) * 0.12, 4),
                "popularity_prior": round(0.12 + ((object_id % 13) * 0.03), 4),
                "freshness": round(0.10 + ((object_id % 9) * 0.08), 4),
                "margin": round(price * margin_pct, 2),
                "margin_pct": round(margin_pct, 4),
                "availability": 1,
                "is_public_domain": bool(row.get("isPublicDomain", False)),
                "object_name": row.get("objectName") or "Artwork",
                "artist_nationality": row.get("artistNationality") or "Unknown",
                "period": row.get("period") or "Unknown",
                "tags": "|".join(sorted({t.lower() for t in tags if t})),
                "dominant_hue": "neutral",
                "palette": "museum",
                "description": f"{row.get('title') or 'Untitled'} by {row.get('artistDisplayName') or 'Unknown Artist'}.",
                "embedding": _to_embedding(
                    str(row.get("classification") or "Paintings"), seed=object_id
                ),
                "primary_image_small": row.get("primaryImageSmall") or "",
            }
        )
    return pd.DataFrame(rows)


def load_met_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Met CSV not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV is supported in this repo")
    return read_csv_with_types(path)
