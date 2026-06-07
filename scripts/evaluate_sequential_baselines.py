from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from artrec.evaluation.decision_intelligence import (
    evaluate_sequential_baseline,
    load_demo_frames,
    save_csv,
    write_sequential_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    _catalog, _users, interactions, _frame = load_demo_frames(
        Path(args.data_dir), Path(args.artifact_dir)
    )
    result = evaluate_sequential_baseline(interactions, top_k=args.top_k)
    report_dir = Path(args.report_dir)
    save_csv(result.metrics, report_dir / "sequential_baseline_results.csv")
    write_sequential_summary(result.metrics, report_dir / "sequential_baseline_summary.md")


if __name__ == "__main__":
    main()
