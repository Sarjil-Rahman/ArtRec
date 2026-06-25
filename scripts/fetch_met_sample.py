"""Fetch a small Met sample when you run locally with internet access."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artrec.data.met_adapter import adapt_met_objects
from artrec.data.io import write_csv
from artrec.utils.logging import get_logger

logger = get_logger(__name__)
SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJECT_URL = (
    "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
)


def fetch_sample(query: str, limit: int) -> pd.DataFrame:
    resp = requests.get(SEARCH_URL, params={"hasImages": True, "q": query}, timeout=30)
    resp.raise_for_status()
    object_ids = resp.json().get("objectIDs") or []
    rows = []
    for object_id in object_ids[:limit]:
        item_resp = requests.get(OBJECT_URL.format(object_id=object_id), timeout=30)
        item_resp.raise_for_status()
        rows.append(item_resp.json())
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="landscape")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="data/raw/met_sample_catalog.csv")
    args = parser.parse_args()
    raw_df = fetch_sample(query=args.query, limit=args.limit)
    catalog_df = adapt_met_objects(raw_df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(catalog_df, out)
    logger.info("Saved Met-adapted sample catalog to %s", out)


if __name__ == "__main__":
    main()
