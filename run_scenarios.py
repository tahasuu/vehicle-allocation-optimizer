"""
Run scenario and trade-off analysis for the vehicle-allocation MIP.

1. Scenarios: base week, hybrid demand surge (+25% on hybrids/EV),
   and a supply disruption at one distribution center (-40%).
2. Trade-off curve: total cost vs. the fair-share floor (minimum fill
   rate every dealer is guaranteed), to show what equity costs.

Outputs go to results/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import build_data
from model import solve_allocation

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

SCENARIOS = {
    "Base week": {},
    "Hybrid/EV surge +25%": {"demand_multiplier": {"Camry Hybrid": 1.25, "RAV4 Hybrid": 1.25,
                                                    "Highlander HEV": 1.25, "bZ EV": 1.25}},
    "San Antonio DC -40%": {"supply_multiplier": {"San Antonio TX": 0.60}},
}


def kpis(name, r):
    return {
        "scenario": name,
        "total_cost_$": round(r["objective"]),
        "transport_$": round(r["transport_cost"]),
        "lost_margin_$": round(r["lost_margin"]),
        "vehicles_shipped": r["vehicles_shipped"],
        "trucks": r["trucks"],
        "truck_utilization": round(r["truck_utilization"], 3),
        "overall_fill": round(r["overall_fill"], 3),
        "min_dealer_fill": round(r["min_dealer_fill"], 3),
        "mip_gap": round(r["gap"], 4),
    }


def main():
    rows = []
    for name, kw in SCENARIOS.items():
        r = solve_allocation(build_data(**kw), min_fill=0.70)
        rows.append(kpis(name, r))
        if name == "Base week":
            r["shipments"].to_csv(OUT / "base_shipments.csv", index=False)
            r["lanes"].to_csv(OUT / "base_truck_lanes.csv", index=False)
    scen = pd.DataFrame(rows)
    scen.to_csv(OUT / "scenario_kpis.csv", index=False)
    print("\nScenario KPIs (fair-share floor = 70%)\n", scen.to_string(index=False))

    # Trade-off: cost of guaranteeing every dealer a minimum fill rate
    base = build_data()
    curve = []
    for f in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        r = solve_allocation(base, min_fill=f)
        curve.append({"min_fill_floor": f, "total_cost_$": r["objective"],
                      "transport_$": r["transport_cost"], "min_dealer_fill": r["min_dealer_fill"],
                      "overall_fill": r["overall_fill"]})
    curve = pd.DataFrame(curve)
    curve.to_csv(OUT / "fairness_tradeoff.csv", index=False)
    print("\nFair-share trade-off\n", curve.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["min_fill_floor"] * 100, curve["total_cost_$"] / 1000, marker="o", color="#1f3a5f")
    ax.set_xlabel("Guaranteed minimum fill rate for every dealer (%)")
    ax.set_ylabel("Total weekly cost ($ thousands)")
    ax.set_title("What does a fairer allocation cost?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fairness_tradeoff.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(scen))
    ax.bar([i - 0.2 for i in x], scen["transport_$"] / 1000, 0.4, label="Transport", color="#1f3a5f")
    ax.bar([i + 0.2 for i in x], scen["lost_margin_$"] / 1000, 0.4, label="Lost margin", color="#c8803a")
    ax.set_xticks(list(x), scen["scenario"])
    ax.set_ylabel("$ thousands per week")
    ax.set_title("Cost breakdown by scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "scenario_costs.png", dpi=150)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
