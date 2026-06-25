from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.evaluation.decision_intelligence import (
    build_strategy_recommendations,
    load_demo_frames,
    save_csv,
    summarize_strategy_metrics,
    write_business_tradeoff_report,
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
    catalog, _users, _interactions, frame = load_demo_frames(
        data_dir, Path(args.artifact_dir)
    )
    recs = build_strategy_recommendations(frame, top_k=args.top_k)
    metrics = summarize_strategy_metrics(recs, catalog_size=len(catalog))
    save_csv(metrics, report_dir / "business_tradeoff_metrics.csv")
    write_business_tradeoff_report(metrics, report_dir / "business_tradeoff_report.md")


if __name__ == "__main__":
    main()
