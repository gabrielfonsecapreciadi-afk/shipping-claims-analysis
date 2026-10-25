"""Generate fully synthetic claim data that mimics the structure (and the mess) of
real spreadsheet exports. No real customers, orders, carriers or amounts are used.

Run:
    python src/generate_sample_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

STATE_WEIGHTS = {
    "TX": 16, "FL": 12, "CA": 10, "GA": 5, "NC": 4, "AZ": 4, "IL": 4, "NY": 4, "OH": 3,
    "TN": 3, "MO": 3, "PA": 3, "VA": 3, "NJ": 3, "MI": 3, "SC": 3, "NV": 2, "CO": 2,
    "WA": 2, "MA": 2, "OK": 2, "LA": 2, "AL": 2, "KS": 1, "NM": 1, "IN": 2,
}

CARRIER_VARIANTS = {
    "CARRIER A": ["CARRIER A", "Carrier A", "carrier a ", "CARRIER A "],
    "CARRIER B": ["CARRIER B", "Carrier B ", "CARRIER B"],
    "CARRIER C": ["CARRIER C", "carrier c", "CARR C"],
}

SERVICE_VARIANTS = {
    "Express 2-Day": ["EXPRESS 2-DAY", "Express 2 Day ", "express 2-day", "EXPRESS 2 DAY AIR"],
    "Ground": ["GROUND", "Ground ", "ground service", "GROUND SERVICE"],
    "Economy": ["ECONOMY", "Economy parcel", "economy "],
}

COMMENTS = {
    "not_delivered": [
        "The package was never delivered.",
        "Order not delivered to the customer.",
        "The carrier lost the package.",
        "Package returned to sender, no tracking updates.",
        "Tracking stuck in transit since last week.",
    ],
    "slower": [
        "We paid for express shipping and it arrived in 4 days.",
        "Shipping took longer than the service we paid for.",
        "Delivery was 2 days later than promised.",
        "Package should take 3 days but took 6.",
    ],
    "damaged": [
        "Products arrived melted and deformed.",
        "The box arrived crushed and the items were broken.",
        "Chocolate arrived in a liquid state.",
        "Package damaged, merchandise discarded.",
    ],
    "other": [
        "Customer requested a refund.",
        "Box arrived open.",
        "Customer says the order was not what they expected.",
    ],
}


def _noisy_dates(rng: np.random.Generator, n: int, start: str, end: str) -> pd.Series:
    span = (pd.Timestamp(end) - pd.Timestamp(start)).days
    offsets = (rng.triangular(0, span * 0.3, span, size=n)).astype(int)
    return pd.Timestamp(start) + pd.to_timedelta(offsets, unit="D")


def make_heat_claims(rng: np.random.Generator, n: int = 900) -> pd.DataFrame:
    states = list(STATE_WEIGHTS)
    probs = np.array(list(STATE_WEIGHTS.values()), dtype=float)
    probs /= probs.sum()
    carrier_key = rng.choice(list(CARRIER_VARIANTS), size=n, p=[0.88, 0.07, 0.05])
    df = pd.DataFrame({
        "order_id": [f"SYN-{1000 + i}" for i in range(n)],
        "date": _noisy_dates(rng, n, "2025-06-02", "2025-10-15"),
        "ice_pack": rng.choice(["Yes", "No"], size=n, p=[0.96, 0.04]),
        "transit_days": rng.choice([1, 2, 3, 4, 5, 6, 7, 9], size=n,
                                   p=[0.04, 0.22, 0.24, 0.26, 0.14, 0.05, 0.03, 0.02]).astype(object),
        "state": rng.choice(states, size=n, p=probs),
        "carrier": [rng.choice(CARRIER_VARIANTS[k]) for k in carrier_key],
        "product_type": rng.choice(["Chocolate", "Gummies", "Hard Candy"], size=n, p=[0.93, 0.04, 0.03]),
        "damage_pct": rng.choice([0.2, 0.3, 0.4, 0.5, 0.7, 1.0], size=n, p=[0.1, 0.3, 0.05, 0.25, 0.1, 0.2]),
        "amount": np.round(np.clip(rng.lognormal(2.9, 0.9, size=n), 0.7, 250), 2),
        "notes": rng.choice(["Photos received.", "Waiting for photos.", "Partial refund accepted.", ""], size=n),
    })
    # Deliberate mess, so the cleaning step has something real to do.
    df.loc[rng.random(n) < 0.04, "amount"] = np.nan
    df.loc[rng.random(n) < 0.06, "state"] = np.nan
    df.loc[rng.random(n) < 0.05, "transit_days"] = "."
    df.loc[rng.random(n) < 0.05, "ice_pack"] = np.nan
    pad = rng.random(n) < 0.05
    df.loc[pad, "state"] = df.loc[pad, "state"].astype(str).str.replace("nan", "", regex=False) + " "
    df.loc[df["state"].astype(str).str.strip() == "", "state"] = np.nan
    df = pd.concat([pd.DataFrame([{"order_id": "Example", "transit_days": 5, "state": "NJ",
                                   "carrier": "CARRIER A", "product_type": "Gummies", "damage_pct": 0.3}]),
                    df, df.sample(5, random_state=1)], ignore_index=True)
    return df


def make_service_claims(rng: np.random.Generator, n: int = 320) -> pd.DataFrame:
    services = list(SERVICE_VARIANTS)
    service_key = rng.choice(services, size=n, p=[0.40, 0.35, 0.25])
    category_probs = {
        "Express 2-Day": [0.18, 0.50, 0.21, 0.11],
        "Ground": [0.30, 0.44, 0.09, 0.17],
        "Economy": [0.52, 0.21, 0.12, 0.15],
    }
    cats = list(COMMENTS)
    rows = []
    dates = _noisy_dates(rng, n, "2025-07-20", "2025-10-20")
    for i, svc in enumerate(service_key):
        cat = rng.choice(cats, p=category_probs[svc])
        days = np.nan
        if cat == "slower" or rng.random() < 0.3:
            days = {"Express 2-Day": rng.integers(3, 7), "Ground": rng.integers(5, 9), "Economy": rng.integers(6, 13)}[svc]
        rows.append({
            "order_id": f"SYN-{5000 + i}",
            "date": dates[i],
            "shipping_service": rng.choice(SERVICE_VARIANTS[svc]),
            "tracking": f"TRK{rng.integers(10**11, 10**12)}",
            "days": days,
            "comments": rng.choice(COMMENTS[cat]),
        })
    df = pd.DataFrame(rows)
    df.loc[rng.choice(n, 2, replace=False), "date"] = pd.Timestamp("2024-08-15")  # year typos
    df = pd.concat([df, df.sample(2, random_state=3)], ignore_index=True)           # exact duplicates
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heat = make_heat_claims(rng)
    svc = make_service_claims(rng)
    heat.to_csv(OUT_DIR / "sample_heat_damage_claims.csv", index=False)
    svc.to_csv(OUT_DIR / "sample_carrier_service_claims.csv", index=False)
    print(f"Wrote {len(heat)} heat-damage rows and {len(svc)} carrier-service rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
