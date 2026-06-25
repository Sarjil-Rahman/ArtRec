from __future__ import annotations
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artrec.data.io import read_csv_with_types
from artrec.simulation.behavior import BehaviorSimulator, SimulationConfig
from artrec.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/raw/catalog.csv")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--n-days", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    catalog_df = read_csv_with_types(Path(args.catalog))
    simulator = BehaviorSimulator(
        catalog_df,
        SimulationConfig(
            n_users=args.n_users, n_days=args.n_days, random_seed=args.seed
        ),
    )
    users_df, catalog_df, impressions_df, interactions_df, conversions_df = (
        simulator.run()
    )
    simulator.save_outputs(
        users_df,
        catalog_df,
        impressions_df,
        interactions_df,
        conversions_df,
        Path(args.output_dir),
    )
    logger.info(
        "Generated users=%s impressions=%s conversions=%s",
        len(users_df),
        len(impressions_df),
        len(conversions_df),
    )


if __name__ == "__main__":
    main()
