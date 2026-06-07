from __future__ import annotations
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from artrec.pipeline import build_end_to_end
from artrec.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    summary = build_end_to_end(
        n_items=160, n_artists=40, n_users=80, n_days=10, seed=42
    )
    logger.info("Build summary: %s", summary)


if __name__ == "__main__":
    main()
