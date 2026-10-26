DEATHS_PATH = "data/snow1/deaths_nd_by_house.shp"
PUMPS_PATH = "data/snow6/pumps.shp"


def load_data():
    import geopandas as gpd

    deaths = gpd.read_file(DEATHS_PATH)
    pumps = gpd.read_file(PUMPS_PATH)

    return deaths, pumps


def extract_coordinates(deaths, pumps):
    X = deaths[
        ["COORD_X", "COORD_Y"]
    ].to_numpy()

    pump_X = pumps[
        ["COORD_X", "COORD_Y"]
    ].to_numpy()

    weights = deaths["deaths"].to_numpy()

    return X, pump_X, weights


def prepare_analysis_data(deaths, pumps, use_death_weights=True):
    """Use affected houses only; optionally give each house unit weight."""
    import numpy as np

    X, pump_X, counts = extract_coordinates(deaths, pumps)
    X, pump_X, counts = (np.asarray(a, dtype=float) for a in (X, pump_X, counts))
    if (not np.isfinite(X).all() or not np.isfinite(counts).all()
            or np.any(counts < 0)):
        raise ValueError("House coordinates and death counts must be finite and nonnegative.")
    if len(pump_X) == 0 or not np.isfinite(pump_X).all():
        raise ValueError("Pump coordinates must be nonempty and finite.")
    mask = counts > 0
    X, counts = X[mask], counts[mask]
    if len(X) == 0:
        raise ValueError("At least one house with positive deaths is required.")
    weights = counts if use_death_weights else np.ones(len(counts))
    print("Analysis weights:", "death counts" if use_death_weights else "one per affected house")
    return X, pump_X, weights
