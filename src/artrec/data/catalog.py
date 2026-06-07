from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import pandas as pd
from artrec.utils.common import ensure_dir
from artrec.data.io import write_csv

LATENT_DIM = 8
STYLES = [
    "abstract",
    "impressionism",
    "modern",
    "contemporary",
    "minimalist",
    "portrait",
    "landscape",
    "surrealism",
]
MEDIUMS = [
    "Oil on canvas",
    "Acrylic on canvas",
    "Watercolor on paper",
    "Mixed media",
    "Ink on paper",
    "Gouache on board",
    "Pastel on paper",
]
CULTURES = ["American", "French", "British", "Italian", "Dutch", "Spanish", "Japanese"]
DEPARTMENTS = [
    "European Paintings",
    "Modern and Contemporary Art",
    "Drawings and Prints",
    "Asian Art",
    "Photographs",
]
CLASSIFICATIONS = ["Paintings", "Drawings", "Prints", "Photography"]
TAGS = [
    "blue",
    "nature",
    "urban",
    "gold",
    "portrait",
    "night",
    "flowers",
    "geometry",
    "movement",
    "sea",
    "dreamlike",
    "architecture",
    "minimal",
    "figure",
    "garden",
]
COLOR_PALETTES = ["warm", "cool", "neutral", "high-contrast", "earth"]
TITLE_PREFIX = [
    "Study of",
    "Composition in",
    "Memory of",
    "Dream of",
    "Portrait of",
    "View of",
    "Silence in",
    "Light over",
    "Echoes of",
    "Fragments of",
]
TITLE_SUBJECT = [
    "Blue Garden",
    "Summer Window",
    "Quiet Street",
    "Golden Figure",
    "Harbor Light",
    "Falling City",
    "Evening Field",
    "Mirror Lake",
    "Red Horizon",
    "Glass Room",
]
ARTIST_FIRST = [
    "Elena",
    "Marcus",
    "Amira",
    "Thomas",
    "Leila",
    "Jonas",
    "Sofia",
    "Noah",
    "Mila",
    "Ari",
]
ARTIST_LAST = [
    "Mercer",
    "Valette",
    "Khan",
    "Bellamy",
    "Rossi",
    "Ito",
    "Dubois",
    "Hart",
    "Morel",
    "Silva",
]


@dataclass
class CatalogConfig:
    n_items: int = 800
    n_artists: int = 150
    random_seed: int = 42


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _artist_names(n_artists: int, rnd: random.Random) -> list[str]:
    if n_artists < 0:
        raise ValueError("n_artists must be non-negative")

    base_names = [f"{first} {last}" for first in ARTIST_FIRST for last in ARTIST_LAST]
    rnd.shuffle(base_names)

    artists: list[str] = []
    suffix = 1
    while len(artists) < n_artists:
        for name in base_names:
            if len(artists) >= n_artists:
                break
            if suffix == 1:
                artists.append(name)
            else:
                artists.append(f"{name} Studio {suffix}")
        suffix += 1
    return artists


def generate_catalog(config: CatalogConfig) -> pd.DataFrame:
    rnd = random.Random(config.random_seed)
    np.random.seed(config.random_seed)
    artists = _artist_names(config.n_artists, rnd)
    artist_ids = [f"artist_{i:03d}" for i in range(config.n_artists)]
    rows = []
    for i in range(config.n_items):
        style = rnd.choice(STYLES)
        style_idx = STYLES.index(style) % LATENT_DIM
        base_vec = np.zeros(LATENT_DIM)
        base_vec[style_idx] = 1.0
        embedding = _normalize(base_vec + np.random.normal(0, 0.6, LATENT_DIM))
        title = f"{rnd.choice(TITLE_PREFIX)} {rnd.choice(TITLE_SUBJECT)}"
        medium = rnd.choice(MEDIUMS)
        culture = rnd.choice(CULTURES)
        department = rnd.choice(DEPARTMENTS)
        classification = rnd.choice(CLASSIFICATIONS)
        artist_idx = i if i < config.n_artists else rnd.randrange(config.n_artists)
        palette = rnd.choice(COLOR_PALETTES)
        price = float(np.exp(np.random.normal(np.log(220), 0.8)))
        quality = float(np.clip(np.random.normal(0.0, 1.0), -2.0, 2.0))
        popularity_prior = float(np.clip(np.random.beta(2, 8), 0.01, 0.99))
        freshness = float(np.clip(np.random.beta(2, 2), 0.0, 1.0))
        margin_pct = rnd.uniform(0.22, 0.58)
        margin = price * margin_pct
        dominant_hue = rnd.choice(
            ["blue", "red", "green", "gold", "black", "white", "violet"]
        )
        year = rnd.randint(1885, 2024)
        sample_tags = rnd.sample(TAGS, k=rnd.randint(2, 4))
        if dominant_hue not in sample_tags:
            sample_tags = sample_tags[:-1] + [dominant_hue]
        rows.append(
            {
                "object_id": 100000 + i,
                "item_id": f"item_{i:04d}",
                "artist_id": artist_ids[artist_idx],
                "artist_display_name": artists[artist_idx],
                "title": title,
                "style": style,
                "medium": medium,
                "culture": culture,
                "department": department,
                "classification": classification,
                "object_date": str(year),
                "price": round(price, 2),
                "quality": round(quality, 4),
                "popularity_prior": round(popularity_prior, 4),
                "freshness": round(freshness, 4),
                "margin": round(margin, 2),
                "margin_pct": round(margin_pct, 4),
                "availability": 0 if rnd.random() < 0.03 else 1,
                "is_public_domain": True,
                "object_name": "Painting",
                "artist_nationality": culture,
                "period": f"{year // 100 + 1}th century",
                "tags": "|".join(sorted(sample_tags)),
                "dominant_hue": dominant_hue,
                "palette": palette,
                "description": f"{title} is a {style} {medium.lower()} work with a {palette} palette and tags {', '.join(sample_tags)}.",
                "embedding": embedding.tolist(),
            }
        )
    return pd.DataFrame(rows)


def save_catalog(df: pd.DataFrame, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    path = output_dir / "catalog.csv"
    write_csv(df, path)
    return path
