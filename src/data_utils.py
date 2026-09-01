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
