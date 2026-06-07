from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from artrec.evaluation.decision_intelligence import (
    build_uplift_by_segment,
    build_user_segments,
    load_demo_frames,
    run_ab_simulation,
    save_csv,
    write_uplift_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    catalog, users, interactions, frame = load_demo_frames(
        data_dir, Path(args.artifact_dir)
    )
    segments_path = data_dir / "processed" / "user_segments.csv"
    if segments_path.exists():
        import pandas as pd

        segments = pd.read_csv(segments_path)
    else:
        segments = build_user_segments(interactions, catalog, users)
    assigned, _metrics = run_ab_simulation(frame, top_k=args.top_k)
    uplift = build_uplift_by_segment(assigned, segments, catalog)
    save_csv(uplift, report_dir / "uplift_by_segment.csv")
    write_uplift_summary(uplift, report_dir / "uplift_summary.md")


if __name__ == "__main__":
    main()
