"""Clean, summarize and cross-check shipping claims with pandas and SQL (SQLite).

Two inputs, both exported from spreadsheets:
  1. heat-damage claims  (customer says the product arrived melted)
  2. carrier-service claims (late, lost or damaged deliveries)

Steps: clean -> summarize with pandas -> verify the same numbers with SQL -> export CSVs.

Usage:
    python src/claims_pipeline.py \
        --heat data/sample_heat_damage_claims.csv \
        --service data/sample_carrier_service_claims.csv \
        --out outputs
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

AMOUNT_BINS = [0, 10, 25, 50, 100, 10_000]
AMOUNT_LABELS = ["$0-10", "$10-25", "$25-50", "$50-100", "$100+"]
LARGE_CLAIM_THRESHOLD = 25          # "large" means strictly greater than this amount
MAX_TRANSIT_GROUP = 7               # transit times of 7 days or more are grouped as "7+"

CARRIER_ALIASES = {"CARR C": "CARRIER C", "CARR B": "CARRIER B", "CARR A": "CARRIER A"}

# Ordered rules: the first pattern that matches wins.
SERVICE_RULES = [
    (r"express|2\s*-?\s*day", "Express 2-Day"),
    (r"ground", "Ground"),
    (r"econom", "Economy"),
]
CATEGORY_RULES = [
    (r"not delivered|never (delivered|arrived)|lost|missing|returned|no tracking|stuck in transit",
     "Not delivered / lost"),
    (r"paid for|longer than|later than|should take|slower", "Slower than paid"),
    (r"melt|liquid|damag|crush|broken|deform", "Damaged / melted"),
]
CATEGORY_ORDER = ["Not delivered / lost", "Slower than paid", "Damaged / melted", "Other"]


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file and drop completely empty rows and columns."""
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df.dropna(how="all").dropna(axis=1, how="all")


# ----------------------------------------------------------------------------
# Cleaning helpers (pure functions, covered by tests)
# ----------------------------------------------------------------------------
def normalize_state(value) -> str | None:
    """' ny ' -> 'NY'; empty values stay empty."""
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def normalize_carrier(value) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    return CARRIER_ALIASES.get(text, text) or None


def normalize_service(value) -> str:
    text = str(value).strip().lower()
    for pattern, label in SERVICE_RULES:
        if re.search(pattern, text):
            return label
    return "Other"


def categorize_comment(value) -> str:
    """Keyword-based classification of a free-text comment. Approximate by design."""
    text = str(value).lower()
    for pattern, label in CATEGORY_RULES:
        if re.search(pattern, text):
            return label
    return "Other"


def amount_bucket(series: pd.Series) -> pd.Series:
    """Bucket amounts; the bins are (0,10], (10,25], (25,50], ... so exactly $25 is NOT 'large'."""
    return pd.cut(series, AMOUNT_BINS, labels=AMOUNT_LABELS).astype("object")


def transit_group(days) -> str | None:
    if pd.isna(days):
        return None
    return f"{MAX_TRANSIT_GROUP}+" if days >= MAX_TRANSIT_GROUP else str(int(days))


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def clean_heat_claims(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df[df["order_id"].astype(str).str.lower() != "example"]
    df = df.drop_duplicates(subset=["order_id"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["transit_days"] = pd.to_numeric(df["transit_days"], errors="coerce")  # '.' -> NaN
    df["state"] = df["state"].map(normalize_state)
    df["carrier"] = df["carrier"].map(normalize_carrier)
    df["ice_pack"] = df["ice_pack"].astype("string").str.strip().str.title()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["amount_bucket"] = amount_bucket(df["amount"])
    df["transit_group"] = df["transit_days"].map(transit_group)
    return df[["order_id", "date", "month", "state", "carrier", "product_type", "ice_pack",
               "transit_days", "transit_group", "damage_pct", "amount", "amount_bucket"]].reset_index(drop=True)


def clean_service_claims(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns the clean frame and a small report of what was fixed."""
    df = raw.copy()
    report = {"rows_in": len(df)}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    report["invalid_dates"] = int(df["date"].isna().sum())
    # Dates with a year far from the most common one are treated as typing errors.
    common_year = int(df["date"].dt.year.mode().iloc[0])
    wrong_year = df["date"].notna() & (df["date"].dt.year != common_year)
    df.loc[wrong_year, "date"] = df.loc[wrong_year, "date"].map(lambda x: x.replace(year=common_year))
    report["year_typos_fixed"] = int(wrong_year.sum())
    df = df.drop_duplicates(subset=["order_id", "tracking", "comments"])
    report["duplicates_removed"] = report["rows_in"] - len(df)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["service"] = df["shipping_service"].map(normalize_service)
    df["category"] = df["comments"].map(categorize_comment)
    df["days"] = pd.to_numeric(df["days"], errors="coerce")
    return df[["date", "month", "service", "category", "days"]].reset_index(drop=True), report


# ----------------------------------------------------------------------------
# Summaries with pandas
# ----------------------------------------------------------------------------
def summarize_heat(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    has_amount = df["amount"].notna()
    large = df["amount"] > LARGE_CLAIM_THRESHOLD
    total = df["amount"].sum()
    by_state = (df.groupby("state").agg(claims=("amount", "size"), usd=("amount", "sum"),
                                        avg_transit_days=("transit_days", "mean"))
                .sort_values("usd", ascending=False).round(2))
    kpis = pd.DataFrame([{
        "claims": len(df), "claims_with_amount": int(has_amount.sum()), "total_usd": round(total, 2),
        "large_claims": int(large.sum()), "large_claims_usd": round(df.loc[large, "amount"].sum(), 2),
        "large_claims_share_of_claims": round(large.sum() / has_amount.sum(), 4),
        "large_claims_share_of_usd": round(df.loc[large, "amount"].sum() / total, 4),
        "top5_states_share_of_usd": round(by_state.head(5)["usd"].sum() / total, 4),
        "claims_missing_state": int(df["state"].isna().sum()),
    }])
    group_order = [str(i) for i in range(0, MAX_TRANSIT_GROUP)] + [f"{MAX_TRANSIT_GROUP}+"]
    return {
        "heat_kpis": kpis,
        "heat_monthly": df.groupby("month").agg(claims=("amount", "size"), usd=("amount", "sum")).round(2).reset_index(),
        "heat_states": by_state.head(10).reset_index(),
        "heat_transit": (df.groupby("transit_group").agg(claims=("amount", "size"), usd=("amount", "sum"),
                                                         avg_damage=("damage_pct", "mean")).round(2)
                         .reindex(group_order).dropna(how="all").astype({"claims": int}).reset_index()),
        "heat_amount_buckets": (df.groupby("amount_bucket").agg(claims=("amount", "size"), usd=("amount", "sum"))
                                .round(2).reindex(AMOUNT_LABELS).reset_index()),
    }


def summarize_service(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    table = pd.crosstab(df["service"], df["category"]).reindex(columns=CATEGORY_ORDER, fill_value=0)
    table["total"] = table.sum(axis=1)
    return {
        "service_by_category": table.sort_values("total", ascending=False).reset_index(),
        "service_monthly": df.groupby("month").size().reset_index(name="claims"),
    }


# ----------------------------------------------------------------------------
# Cross-check with SQL
# ----------------------------------------------------------------------------
def sql_crosscheck(heat: pd.DataFrame, service: pd.DataFrame, expected: dict[str, pd.DataFrame]) -> None:
    """Re-compute key numbers with SQL; raise AssertionError if pandas and SQL disagree."""
    con = sqlite3.connect(":memory:")
    heat.assign(date=heat["date"].dt.strftime("%Y-%m-%d")).to_sql("heat_claims", con, index=False)
    service.assign(date=service["date"].dt.strftime("%Y-%m-%d")).to_sql("service_claims", con, index=False)

    total = pd.read_sql("SELECT ROUND(SUM(amount), 2) AS usd FROM heat_claims", con).iloc[0, 0]
    large = pd.read_sql(f"SELECT COUNT(*) AS n, ROUND(SUM(amount), 2) AS usd FROM heat_claims "
                        f"WHERE amount > {LARGE_CLAIM_THRESHOLD}", con)
    states = pd.read_sql("SELECT state FROM heat_claims WHERE state IS NOT NULL GROUP BY state "
                         "ORDER BY SUM(amount) DESC LIMIT 10", con)["state"].tolist()
    svc_rows = pd.read_sql("SELECT COUNT(*) AS n FROM service_claims", con).iloc[0, 0]

    k = expected["heat_kpis"].iloc[0]
    assert abs(total - k["total_usd"]) < 0.01, "total amount differs between pandas and SQL"
    assert int(large.iloc[0, 0]) == int(k["large_claims"]), "large-claim count differs"
    assert abs(large.iloc[0, 1] - k["large_claims_usd"]) < 0.01, "large-claim amount differs"
    assert states == expected["heat_states"]["state"].tolist(), "top-10 states differ"
    assert int(svc_rows) == int(expected["service_by_category"]["total"].sum()), "service claim count differs"
    con.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
HEAT_COLUMNS = ["order_id", "date", "ice_pack", "transit_days", "state", "carrier",
                "product_type", "damage_pct", "amount", "notes"]
SERVICE_COLUMNS = ["order_id", "date", "shipping_service", "tracking", "days", "comments"]


def run(heat_path: str, service_path: str, out_dir: str) -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    heat_raw, service_raw = read_table(heat_path), read_table(service_path)
    missing = (set(HEAT_COLUMNS) - set(heat_raw.columns)) | (set(SERVICE_COLUMNS) - set(service_raw.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    heat = clean_heat_claims(heat_raw)
    service, report = clean_service_claims(service_raw)
    summaries = {**summarize_heat(heat), **summarize_service(service)}
    sql_crosscheck(heat, service, summaries)

    for name, frame in summaries.items():           # exports contain no order ids or tracking numbers
        frame.to_csv(out / f"{name}.csv", index=False)

    print("OK: pandas and SQL agree on the key numbers")
    print(f"Heat claims: {len(heat)} clean rows | Service claims: {len(service)} clean rows")
    print(f"Service-file fixes: {report}")
    print("\n", summaries["heat_kpis"].T.to_string(header=False))
    print("\n", summaries["service_by_category"].to_string(index=False))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heat", required=True, help="CSV/XLSX with heat-damage claims")
    parser.add_argument("--service", required=True, help="CSV/XLSX with carrier-service claims")
    parser.add_argument("--out", default="outputs", help="folder for summary CSVs")
    args = parser.parse_args()
    run(args.heat, args.service, args.out)


if __name__ == "__main__":
    main()
