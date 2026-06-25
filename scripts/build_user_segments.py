from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.data.io import read_csv_with_types
from artrec.evaluation.decision_intelligence import (
    build_user_segments,
    load_demo_frames,
    save_csv,
    write_user_segments_summary,
)


def _training_cutoff(data_dir: Path):
    train_path = data_dir / "processed" / "train_features.csv"
    if not train_path.exists():
        return None
    train_df = read_csv_with_types(train_path)
    if train_df.empty or "timestamp" not in train_df.columns:
        return None
    return train_df["timestamp"].max()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    catalog, users, interactions, _frame = load_demo_frames(
        data_dir, Path(args.artifact_dir)
    )
    segments = build_user_segments(
        interactions, catalog, users, cutoff=_training_cutoff(data_dir)
    )
    save_csv(segments, data_dir / "processed" / "user_segments.csv")
    write_user_segments_summary(segments, Path(args.report_dir) / "user_segments_summary.md")


if __name__ == "__main__":
    main()
