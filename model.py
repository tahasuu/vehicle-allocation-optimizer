"""
Mixed-integer program (MIP) for weekly vehicle allocation, solved with Gurobi.

Decision variables
  x[d, m, j]  integer  vehicles of model m shipped from DC d to dealer j
  y[d, j]     integer  trucks dispatched on lane d -> j
  s[m, j]     >= 0     unfilled demand (lost sales) of model m at dealer j

Objective
  minimize  sum truck_cost[d, j] * y[d, j]          (transport)
          + sum margin[m] * s[m, j]                 (lost gross margin)

Constraints
  supply:        sum_j x[d, m, j]          <= supply[d, m]
  demand:        sum_d x[d, m, j] + s[m,j] == demand[j, m]
  truck loading: sum_m x[d, m, j]          <= truck_capacity * y[d, j]
  lot capacity:  sum_{d,m} x[d, m, j]      <= lot_capacity[j]
  fair share:    sum_{d,m} x[d, m, j]      >= min_fill * total_demand[j]   (optional)

The fair-share floor guarantees every dealer a minimum fill rate, so the
optimizer cannot starve far-away dealers just because they cost more to serve.
"""
import gurobipy as gp
from gurobipy import GRB
import pandas as pd


def solve_allocation(data, min_fill=0.0, time_limit=60, mip_gap=0.001, verbose=False):
    M, D, J = data["models"], data["dcs"], data["dealers"]
    dem, sup = data["demand"], data["supply"]
    cap, tc, mg = data["lot_capacity"], data["truck_cost"], data["margin"]
    Q = data["truck_capacity"]

    mdl = gp.Model("vehicle_allocation")
    mdl.Params.OutputFlag = int(verbose)
    mdl.Params.TimeLimit = time_limit
    mdl.Params.MIPGap = mip_gap

    x = mdl.addVars(D, M, J, vtype=GRB.INTEGER, lb=0, name="x")
    y = mdl.addVars(D, J, vtype=GRB.INTEGER, lb=0, name="y")
    s = mdl.addVars(M, J, lb=0, name="s")

    transport = gp.quicksum(tc.loc[d, j] * y[d, j] for d in D for j in J)
    lost_margin = gp.quicksum(mg[m] * s[m, j] for m in M for j in J)
    mdl.setObjective(transport + lost_margin, GRB.MINIMIZE)

    mdl.addConstrs((x.sum(d, m, "*") <= sup.loc[d, m] for d in D for m in M), "supply")
    mdl.addConstrs((x.sum("*", m, j) + s[m, j] == dem.loc[j, m] for m in M for j in J), "demand")
    mdl.addConstrs((x.sum(d, "*", j) <= Q * y[d, j] for d in D for j in J), "truck")
    mdl.addConstrs((x.sum("*", "*", j) <= cap[j] for j in J), "lot")
    if min_fill > 0:
        total_dem = dem.sum(axis=1)
        # A dealer can never be asked to exceed its lot or its own demand.
        mdl.addConstrs((x.sum("*", "*", j) >= min_fill * min(total_dem[j], cap[j]) for j in J), "fair_share")

    mdl.optimize()
    if mdl.SolCount == 0:
        raise RuntimeError(f"No solution found (status {mdl.Status})")

    ship = pd.DataFrame(
        [(d, m, j, int(round(x[d, m, j].X))) for d in D for m in M for j in J if x[d, m, j].X > 0.5],
        columns=["dc", "model", "dealer", "vehicles"],
    )
    trucks = pd.DataFrame(
        [(d, j, int(round(y[d, j].X)), data["miles"].loc[d, j]) for d in D for j in J if y[d, j].X > 0.5],
        columns=["dc", "dealer", "trucks", "miles"],
    )
    filled = ship.groupby("dealer")["vehicles"].sum().reindex(J, fill_value=0)
    # Fill rate vs. what the dealer could actually receive (its demand, capped by lot space)
    dealer_fill = filled / pd.concat([dem.sum(axis=1), cap], axis=1).min(axis=1)

    return {
        "status": mdl.Status, "objective": mdl.ObjVal, "gap": mdl.MIPGap,
        "transport_cost": transport.getValue(), "lost_margin": lost_margin.getValue(),
        "vehicles_shipped": int(ship["vehicles"].sum()), "trucks": int(trucks["trucks"].sum()),
        "truck_utilization": ship["vehicles"].sum() / (Q * trucks["trucks"].sum()),
        "overall_fill": filled.sum() / dem.values.sum(), "min_dealer_fill": dealer_fill.min(),
        "shipments": ship, "lanes": trucks, "dealer_fill": dealer_fill,
    }
