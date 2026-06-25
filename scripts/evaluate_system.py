from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artrec.data.io import read_csv_with_types
from artrec.business.reporting import summarise_business_metrics, write_business_report
from artrec.utils.common import dump_json
from artrec.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    data_dir, artifact_dir = Path(args.data_dir), Path(args.artifact_dir)
    impressions_df = read_csv_with_types(data_dir / "raw" / "impressions.csv")
    conversions_df = read_csv_with_types(data_dir / "raw" / "conversions.csv")
    model_metrics = json.loads(
        (artifact_dir / "model_metrics.json").read_text(encoding="utf-8")
    )
    business_metrics = summarise_business_metrics(impressions_df, conversions_df)
    dump_json(artifact_dir / "business_metrics.json", business_metrics)
    write_business_report(
        business_metrics, model_metrics, artifact_dir / "business_report.md"
    )
    logger.info("Saved business metrics and report")


if __name__ == "__main__":
    main()
