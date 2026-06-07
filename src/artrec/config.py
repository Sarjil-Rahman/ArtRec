from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("ARTREC_DATA_DIR", "data"))
    artifact_dir: Path = Path(os.getenv("ARTREC_ARTIFACT_DIR", "artifacts"))
    random_seed: int = int(os.getenv("ARTREC_RANDOM_SEED", "42"))
    host: str = os.getenv("ARTREC_HOST", "0.0.0.0")
    port: int = int(os.getenv("ARTREC_PORT", "8000"))


def get_settings() -> Settings:
    return Settings()
