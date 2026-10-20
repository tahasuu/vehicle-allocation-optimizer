"""
Synthetic data for a vehicle-allocation problem.

Distribution centers (DCs) hold new-vehicle supply by model.
Dealers have a forecast demand by model and a limited lot capacity.
Vehicles ship by truck (fixed capacity per truck), so lanes have a
fixed per-truck cost that depends on distance.

All numbers are synthetic and for demonstration only.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
TRUCK_CAPACITY = 8          # vehicles per car-hauler truck
COST_PER_TRUCK_MILE = 3.10  # $ per truck-mile (illustrative)
FIXED_COST_PER_TRUCK = 450  # $ per dispatched truck (illustrative)

MODELS = {
    # model: (gross margin $ lost per unfilled unit, share of demand)
    "Camry Hybrid":   (2600, 0.20),
    "RAV4 Hybrid":    (3100, 0.24),
    "RAV4 Gas":       (2400, 0.12),
    "Tacoma":         (3600, 0.16),
    "Highlander HEV": (3900, 0.14),
    "bZ EV":          (2900, 0.14),
}

DCS = {
    # name: (lat, lon)
    "Georgetown KY": (38.21, -84.56),
    "San Antonio TX": (29.42, -98.49),
    "Long Beach CA": (33.77, -118.19),
}

DEALER_CITIES = {
    "Dallas TX": (32.78, -96.80), "Houston TX": (29.76, -95.37),
    "Austin TX": (30.27, -97.74), "Oklahoma City OK": (35.47, -97.52),
    "Denver CO": (39.74, -104.99), "Phoenix AZ": (33.45, -112.07),
    "Los Angeles CA": (34.05, -118.24), "San Diego CA": (32.72, -117.16),
    "Las Vegas NV": (36.17, -115.14), "Salt Lake City UT": (40.76, -111.89),
    "Chicago IL": (41.88, -87.63), "Indianapolis IN": (39.77, -86.16),
    "Columbus OH": (39.96, -83.00), "Nashville TN": (36.16, -86.78),
    "Atlanta GA": (33.75, -84.39), "Charlotte NC": (35.23, -80.84),
    "St. Louis MO": (38.63, -90.20), "Kansas City MO": (39.10, -94.58),
    "Memphis TN": (35.15, -90.05), "New Orleans LA": (29.95, -90.07),
}


def haversine_miles(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (*a, *b))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(d)) * 1.2  # 1.2 = road-circuity factor


def build_data(demand_multiplier=None, supply_multiplier=None, seed=RNG_SEED):
    """Return a dict of pandas objects describing one planning week.

    demand_multiplier: optional {model: factor} to stress specific models.
    supply_multiplier: optional {dc: factor} to simulate a disruption.
    """
    rng = np.random.default_rng(seed)
    models, dcs, dealers = list(MODELS), list(DCS), list(DEALER_CITIES)

    # Dealer size drives total weekly demand (vehicles)
    dealer_size = pd.Series(rng.integers(25, 70, len(dealers)), index=dealers)
    shares = pd.Series({m: MODELS[m][1] for m in models})
    noise = rng.uniform(0.75, 1.25, (len(dealers), len(models)))
    demand = pd.DataFrame(np.outer(dealer_size, shares) * noise, index=dealers, columns=models).round().astype(int)
    if demand_multiplier:
        for m, f in demand_multiplier.items():
            demand[m] = (demand[m] * f).round().astype(int)

    # Supply covers ~92% of base demand, split unevenly across DCs
    base_total = pd.DataFrame(np.outer(dealer_size, shares), index=dealers, columns=models).sum()
    dc_split = pd.DataFrame(rng.dirichlet([2, 2, 2], len(models)).T, index=dcs, columns=models)
    supply = (dc_split * base_total * 0.92).round().astype(int)
    if supply_multiplier:
        for d, f in supply_multiplier.items():
            supply.loc[d] = (supply.loc[d] * f).round().astype(int)

    lot_capacity = (dealer_size * 1.1).round().astype(int)
    miles = pd.DataFrame({d: [haversine_miles(DCS[d], DEALER_CITIES[j]) for j in dealers] for d in dcs}, index=dealers).T
    truck_cost = FIXED_COST_PER_TRUCK + COST_PER_TRUCK_MILE * miles
    margin = pd.Series({m: MODELS[m][0] for m in models})

    return {
        "models": models, "dcs": dcs, "dealers": dealers,
        "demand": demand, "supply": supply, "lot_capacity": lot_capacity,
        "miles": miles, "truck_cost": truck_cost, "margin": margin,
        "truck_capacity": TRUCK_CAPACITY,
    }
