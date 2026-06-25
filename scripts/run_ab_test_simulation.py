from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.evaluation.decision_intelligence import (
    bootstrap_lift_ci,
    load_demo_frames,
    run_ab_simulation,
    save_csv,
    summarize_lift_metrics,
    write_ab_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    _catalog, _users, _interactions, frame = load_demo_frames(
        Path(args.data_dir), Path(args.artifact_dir)
    )
    assigned, metrics = run_ab_simulation(frame, top_k=args.top_k)
    report_dir = Path(args.report_dir)
    save_csv(metrics, report_dir / "ab_test_simulation_results.csv")
    save_csv(summarize_lift_metrics(metrics), report_dir / "ab_test_lift_summary.csv")
    ci = bootstrap_lift_ci(assigned, metric="profit_proxy")
    write_ab_report(metrics, ci, report_dir / "ab_test_simulation.md")


if __name__ == "__main__":
    main()
