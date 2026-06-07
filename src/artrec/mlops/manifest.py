from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import platform
import subprocess
import uuid

from artrec.models.ranker import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from artrec.utils.common import dump_json


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "fastapi", "pydantic", "joblib"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_manifest(
    data_dir: str | Path,
    artifact_dir: str | Path,
    metrics_path: str | Path,
    seed: int,
    model_type: str = "sklearn.linear_model.LogisticRegression",
) -> dict:
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    metrics_path = Path(metrics_path)
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "data_paths": {
            "catalog": str(data_dir / "raw" / "catalog.csv"),
            "users": str(data_dir / "raw" / "users.csv"),
            "impressions": str(data_dir / "raw" / "impressions.csv"),
            "train_features": str(data_dir / "processed" / "train_features.csv"),
            "test_features": str(data_dir / "processed" / "test_features.csv"),
        },
        "artifact_paths": {
            "ranker": str(artifact_dir / "ranker.joblib"),
            "retriever": str(artifact_dir / "retriever.joblib"),
            "business_metrics": str(artifact_dir / "business_metrics.json"),
            "business_report": str(artifact_dir / "business_report.md"),
            "manifest": str(artifact_dir / "manifest.json"),
        },
        "model_type": model_type,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "metrics_path": str(metrics_path),
        "config": {"random_seed": seed},
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
    }


def write_manifest(
    output_path: str | Path,
    data_dir: str | Path,
    artifact_dir: str | Path,
    metrics_path: str | Path,
    seed: int,
) -> Path:
    output_path = Path(output_path)
    manifest = build_manifest(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        metrics_path=metrics_path,
        seed=seed,
    )
    dump_json(output_path, manifest)
    return output_path
