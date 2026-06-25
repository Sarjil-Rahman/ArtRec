from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.data.io import read_csv_with_types
from artrec.models.ranker import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from artrec.monitoring.drift import check_drift
from artrec.utils.common import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lightweight feature drift checks."
    )
    parser.add_argument("--reference", default="data/processed/train_features.csv")
    parser.add_argument("--current", default="data/processed/test_features.csv")
    parser.add_argument("--output", default="artifacts/drift_report.json")
    args = parser.parse_args()

    reference_df = read_csv_with_types(Path(args.reference))
    current_df = read_csv_with_types(Path(args.current))
    report = check_drift(
        reference_df, current_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    dump_json(Path(args.output), report)

    summary = report["summary"]
    print(
        json.dumps(
            {
                "reference_rows": summary["reference_rows"],
                "current_rows": summary["current_rows"],
                "alert_count": summary["alert_count"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
