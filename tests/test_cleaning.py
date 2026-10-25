"""Unit tests for the pure cleaning helpers and the end-to-end pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import claims_pipeline as cp  # noqa: E402


def test_normalize_state_strips_and_uppercases():
    assert cp.normalize_state(" ny ") == "NY"
    assert cp.normalize_state("") is None
    assert cp.normalize_state(np.nan) is None


def test_normalize_carrier_collapses_spaces_and_aliases():
    assert cp.normalize_carrier("carrier  a ") == "CARRIER A"
    assert cp.normalize_carrier("CARR C") == "CARRIER C"


def test_normalize_service_variants():
    assert cp.normalize_service("EXPRESS 2 DAY AIR") == "Express 2-Day"
    assert cp.normalize_service("ground service ") == "Ground"
    assert cp.normalize_service("Economy parcel") == "Economy"
    assert cp.normalize_service("pigeon") == "Other"


def test_categorize_comment():
    assert cp.categorize_comment("The carrier lost the package.") == "Not delivered / lost"
    assert cp.categorize_comment("We paid for express and it took longer") == "Slower than paid"
    assert cp.categorize_comment("Products arrived melted") == "Damaged / melted"
    assert cp.categorize_comment("Customer requested a refund") == "Other"


def test_amount_bucket_boundary_25_is_not_large():
    buckets = cp.amount_bucket(pd.Series([5.0, 25.0, 25.01, 150.0]))
    assert buckets.tolist() == ["$0-10", "$10-25", "$25-50", "$100+"]


def test_transit_group_caps_at_seven():
    assert cp.transit_group(2) == "2"
    assert cp.transit_group(7) == "7+"
    assert cp.transit_group(12) == "7+"
    assert cp.transit_group(np.nan) is None


def test_clean_heat_claims_drops_example_and_duplicates():
    raw = pd.DataFrame({
        "order_id": ["Example", "A1", "A1", "A2"],
        "date": ["2025-07-01"] * 4, "ice_pack": ["yes ", "Yes", "Yes", "no"],
        "transit_days": [5, 3, 3, "."], "state": ["NJ", " tx ", " tx ", "FL"],
        "carrier": ["CARRIER A"] * 4, "product_type": ["Chocolate"] * 4,
        "damage_pct": [0.3] * 4, "amount": [10, 30, 30, 20], "notes": [""] * 4,
    })
    clean = cp.clean_heat_claims(raw)
    assert len(clean) == 2
    assert clean["state"].tolist() == ["TX", "FL"]
    assert np.isnan(clean.loc[clean["order_id"] == "A2", "transit_days"].iloc[0])


def test_pipeline_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = cp.run(str(root / "data" / "sample_heat_damage_claims.csv"),
                    str(root / "data" / "sample_carrier_service_claims.csv"), str(tmp_path))
    kpis = result["heat_kpis"].iloc[0]
    assert 0 < kpis["large_claims_share_of_usd"] <= 1
    assert (tmp_path / "heat_kpis.csv").exists()
    exported = pd.read_csv(tmp_path / "service_by_category.csv")
    assert "order_id" not in exported.columns and "tracking" not in exported.columns
