"""Draw the summary charts used in the README from the CSVs written by claims_pipeline.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs"
IMG = ROOT / "docs" / "images"

BLUE, MUTED, GRID, TEXT = "#2F6FDE", "#B9C2D0", "#E6E9EF", "#2B3140"
STACK = ["#2F6FDE", "#E8833A", "#3AA17E", "#B9C2D0"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def heat_overview() -> None:
    monthly = pd.read_csv(OUT / "heat_monthly.csv")
    states = pd.read_csv(OUT / "heat_states.csv")
    buckets = pd.read_csv(OUT / "heat_amount_buckets.csv")
    transit = pd.read_csv(OUT / "heat_transit.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    fig.suptitle("Heat-damage claims (synthetic sample data)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=TEXT)

    ax = axes[0, 0]; style(ax)
    labels = pd.to_datetime(monthly["month"] + "-01").dt.strftime("%b")
    ax.bar(labels, monthly["usd"], color=BLUE, width=0.6)
    ax.set_title("Claim amount by month (USD)")
    ax.set_ylim(0, monthly["usd"].max() * 1.15)
    for x, v in zip(labels, monthly["usd"]):
        ax.text(x, v, f"${v/1000:.1f}k", ha="center", va="bottom", fontsize=9, color=TEXT)

    ax = axes[0, 1]; ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    s = states.iloc[::-1]
    ax.barh(s["state"], s["usd"], color=BLUE, height=0.65)
    ax.set_title("Top 10 states by claim amount (USD)")
    for y, v in zip(s["state"], s["usd"]):
        ax.text(v, y, f" ${v:,.0f}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, s["usd"].max() * 1.18)

    ax = axes[1, 0]; style(ax)
    colors = [MUTED if b in ("$0-10", "$10-25") else BLUE for b in buckets["amount_bucket"]]
    ax.bar(buckets["amount_bucket"], buckets["usd"], color=colors, width=0.6)
    ax.set_title("Where the money is: amount by claim size (USD)")
    ax.set_ylim(0, buckets["usd"].max() * 1.15)
    for x, v in zip(buckets["amount_bucket"], buckets["usd"]):
        ax.text(x, v, f"${v/1000:.1f}k", ha="center", va="bottom", fontsize=9, color=TEXT)

    ax = axes[1, 1]; style(ax)
    ax.bar(transit["transit_group"].astype(str), transit["claims"], color=BLUE, width=0.6)
    ax.set_title("Claims by days in transit")
    ax.set_ylim(0, transit["claims"].max() * 1.15)
    ax.set_xlabel("Days in transit (claims only: no shipment totals available)")
    for x, v in zip(transit["transit_group"].astype(str), transit["claims"]):
        ax.text(x, v, f"{int(v)}", ha="center", va="bottom", fontsize=9, color=TEXT)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(IMG / "heat_claims_overview.png", dpi=150)
    plt.close(fig)


def service_failures() -> None:
    t = pd.read_csv(OUT / "service_by_category.csv")
    cats = [c for c in ["Not delivered / lost", "Slower than paid", "Damaged / melted", "Other"] if c in t]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    t = t.iloc[::-1]
    left = pd.Series(0, index=t.index)
    for color, cat in zip(STACK, cats):
        ax.barh(t["service"], t[cat], left=left, color=color, height=0.6, label=cat)
        left = left + t[cat]
    for y, total in zip(t["service"], t["total"]):
        ax.text(total, y, f" {total}", va="center", fontsize=9, color=TEXT)
    ax.set_title("Carrier-service claims by service and failure type (synthetic sample data)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), fontsize=9)
    ax.set_xlim(0, t["total"].max() * 1.12)
    fig.tight_layout()
    fig.savefig(IMG / "service_failures.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    heat_overview()
    service_failures()
    print(f"Charts written to {IMG}")
