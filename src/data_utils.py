import geopandas as gpd


DEATHS_PATH = "data/snow2/deaths_by_bldg.shp"
PUMPS_PATH = "data/snow6/pumps.shp"


def load_data():
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