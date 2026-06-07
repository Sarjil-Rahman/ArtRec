from __future__ import annotations
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from artrec.data.catalog import CatalogConfig, generate_catalog, save_catalog
from artrec.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-items", type=int, default=800)
    parser.add_argument("--n-artists", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate_catalog(
        CatalogConfig(
            n_items=args.n_items, n_artists=args.n_artists, random_seed=args.seed
        )
    )
    path = save_catalog(df, Path(args.output_dir))
    logger.info("Catalog saved to %s with %s rows", path, len(df))


if __name__ == "__main__":
    main()
