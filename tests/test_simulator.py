from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from artrec.data.catalog import CatalogConfig, generate_catalog
from artrec.business.reporting import summarise_business_metrics
from artrec.simulation.behavior import BehaviorSimulator, SimulationConfig


def test_simulator_generates_rows():
    catalog_df = generate_catalog(
        CatalogConfig(n_items=40, n_artists=10, random_seed=1)
    )
    simulator = BehaviorSimulator(
        catalog_df, SimulationConfig(n_users=20, n_days=5, random_seed=1)
    )
    users_df, catalog_df, impressions_df, interactions_df, conversions_df = (
        simulator.run()
    )
    assert len(users_df) == 20
    assert len(impressions_df) > 0
    assert "clicked" in impressions_df.columns
    assert "not_interested" in impressions_df.columns
    assert set(impressions_df["not_interested"].unique()).issubset({0, 1})


def test_simulator_marketplace_rates_are_plausible():
    catalog_df = generate_catalog(
        CatalogConfig(n_items=120, n_artists=25, random_seed=7)
    )
    simulator = BehaviorSimulator(
        catalog_df,
        SimulationConfig(n_users=50, n_days=8, random_seed=7),
    )
    _, _, impressions_df, _, conversions_df = simulator.run()

    metrics = summarise_business_metrics(impressions_df, conversions_df)
    negative_rate = impressions_df["not_interested"].mean()
    assert 0.0 <= negative_rate <= 0.15
    assert 0.03 <= metrics["ctr"] <= 0.20
    assert 0.01 <= metrics["save_rate"] <= 0.15
    assert 0.005 <= metrics["cart_rate"] <= 0.08
    assert 0.0005 <= metrics["immediate_purchase_rate"] <= 0.03
    assert metrics["immediate_purchase_rate"] < metrics["cart_rate"] < metrics["ctr"]
