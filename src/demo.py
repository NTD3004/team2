"""In-memory house editing and filtering for the Jupyter demo."""
import numpy as np

from data_utils import load_data


class HouseDataEditor:
    """Keep the source data intact while preparing a reproducible selection."""

    def __init__(self, houses):
        self.original = houses.copy(deep=True).reset_index(drop=True)
        self.reset()

    def reset(self):
        self.houses = self.original.copy(deep=True)
        self.included = np.ones(len(self.houses), dtype=bool)

    def update(self, row, x, y, deaths, included=True):
        if not isinstance(row, (int, np.integer)) or not 0 <= row < len(self.houses):
            raise ValueError("Select an existing house.")
        values = np.asarray([x, y, deaths], dtype=float)
        if not np.isfinite(values).all() or deaths < 0 or deaths != int(deaths):
            raise ValueError("Coordinates must be finite; deaths must be a nonnegative integer.")
        for name, value in zip(("COORD_X", "COORD_Y", "deaths"), values):
            self.houses.at[row, name] = value
        # Keep a GeoDataFrame's geometry consistent with its coordinate columns.
        if hasattr(self.houses, "geometry"):
            from shapely.geometry import Point
            self.houses.at[row, self.houses.geometry.name] = Point(x, y)
        self.included[row] = bool(included)

    def select(self, min_deaths=0, x_range=None, y_range=None):
        mask = self.included & (self.houses["deaths"].to_numpy() >= min_deaths)
        for column, bounds in (("COORD_X", x_range), ("COORD_Y", y_range)):
            if bounds is not None:
                if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                    raise ValueError("Filter bounds must be finite and ordered from minimum to maximum.")
                mask &= self.houses[column].between(*bounds).to_numpy()
        return self.houses.loc[mask].copy()


def show_demo():
    """Display the editor in a live Jupyter notebook and return its widget."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    houses, pumps = load_data()
    editor = HouseDataEditor(houses)
    labels = [
        (f"House {row.get('ID', index)} (row {index})", index)
        for index, row in editor.houses.iterrows()
    ]
    house = widgets.Dropdown(options=labels, description="House:")
    x = widgets.FloatText(description="X (metres):")
    y = widgets.FloatText(description="Y (metres):")
    deaths = widgets.BoundedIntText(min=0, max=1000000, description="Deaths:")
    included = widgets.Checkbox(value=True, description="Include house")
    apply = widgets.Button(description="Apply house edit", button_style="primary")
    reset = widgets.Button(description="Reset all data")
    minimum = widgets.BoundedIntText(value=0, min=0, max=1000000, description="Min deaths:")
    spatial = widgets.Checkbox(value=False, description="Filter by coordinate bounds")
    xmin = widgets.FloatText(value=float(houses.COORD_X.min()), description="Min X:")
    xmax = widgets.FloatText(value=float(houses.COORD_X.max()), description="Max X:")
    ymin = widgets.FloatText(value=float(houses.COORD_Y.min()), description="Min Y:")
    ymax = widgets.FloatText(value=float(houses.COORD_Y.max()), description="Max Y:")
    preview = widgets.Button(description="Preview selection")
    method = widgets.Dropdown(options=["kmeans", "gmm", "kde"], description="Method:")
    k = widgets.BoundedIntText(value=8, min=1, max=1000, description="K:")
    weighted = widgets.Checkbox(value=True, description="Use death weights")
    n_init = widgets.BoundedIntText(value=50, min=1, max=1000, description="Initializations:")
    regularization = widgets.FloatText(value=1e-6, description="GMM reg.:")
    bandwidth = widgets.FloatText(value=50, description="KDE bandwidth:")
    run = widgets.Button(description="Run on selection", button_style="success")
    status = widgets.HTML()
    data_output = widgets.Output()
    result_output = widgets.Output()

    def selection():
        return editor.select(minimum.value,
                             (xmin.value, xmax.value) if spatial.value else None,
                             (ymin.value, ymax.value) if spatial.value else None)

    def invalidate(*_):
        result_output.clear_output()
        status.value = "Data or settings changed. Preview and run again to refresh results."

    def load_house(*_):
        row = editor.houses.iloc[house.value]
        x.value, y.value = float(row.COORD_X), float(row.COORD_Y)
        deaths.value = int(row.deaths)
        included.value = bool(editor.included[house.value])

    def preview_data(*_):
        with data_output:
            data_output.clear_output(wait=True)
            try:
                selected = selection()
                print(f"Selected {len(selected)} / {len(editor.houses)} houses; "
                      f"{int(selected.deaths.sum())} deaths; "
                      f"{int((selected.deaths > 0).sum())} affected houses.")
                print("Models use only houses with positive death counts. Table shows up to 100 rows.")
                columns = [c for c in ("ID", "COORD_X", "COORD_Y", "deaths") if c in selected]
                display(selected[columns].head(100))
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.scatter(editor.houses.COORD_X, editor.houses.COORD_Y,
                           s=8, c="lightgray", label="All houses")
                ax.scatter(selected.COORD_X, selected.COORD_Y,
                           s=8 + 5 * selected.deaths, label="Selected houses", alpha=0.7)
                row = editor.houses.iloc[house.value]
                ax.scatter(row.COORD_X, row.COORD_Y, marker="s", s=90,
                           facecolors="none", edgecolors="red", label="House being edited")
                ax.scatter(pumps.COORD_X, pumps.COORD_Y, marker="x", c="black", label="Pumps")
                ax.set(xlabel="X (metres)", ylabel="Y (metres)", title="Current house selection")
                ax.set_aspect("equal", adjustable="box")
                ax.legend()
                display(fig)
                plt.close(fig)
            except ValueError as error:
                print(error)

    def apply_edit(_):
        try:
            editor.update(house.value, x.value, y.value, deaths.value, included.value)
            invalidate()
            status.value = "House edit applied in memory. Run on selection to update model results."
            preview_data()
        except ValueError as error:
            status.value = str(error)

    def reset_data(_):
        editor.reset()
        minimum.value = 0
        spatial.value = False
        xmin.value, xmax.value = float(houses.COORD_X.min()), float(houses.COORD_X.max())
        ymin.value, ymax.value = float(houses.COORD_Y.min()), float(houses.COORD_Y.max())
        load_house()
        invalidate()
        status.value = "Original house data restored; filters cleared."
        preview_data()

    def run_model(_):
        run.disabled = True
        with result_output:
            result_output.clear_output(wait=True)
            try:
                selected = selection()
                affected = selected.loc[selected.deaths > 0]
                if affected.empty:
                    raise ValueError("Select at least one house with positive deaths.")
                if method.value != "kde" and k.value > len(affected[["COORD_X", "COORD_Y"]].drop_duplicates()):
                    raise ValueError("K cannot exceed the number of distinct affected locations.")
                data = (selected, pumps.copy(deep=True))
                status.value = "Running on the current selection…"
                if method.value == "kmeans":
                    from kmeans import run_kmeans
                    run_kmeans(k=k.value, n_init=n_init.value,
                               use_death_weights=weighted.value, data=data)
                elif method.value == "gmm":
                    from gmm import run_gmm
                    if not np.isfinite(regularization.value) or regularization.value <= 0:
                        raise ValueError("GMM regularization must be finite and positive.")
                    run_gmm(k=k.value, n_init=n_init.value, reg_covar=regularization.value,
                            use_death_weights=weighted.value, data=data)
                else:
                    from kde import run_kde
                    if not np.isfinite(bandwidth.value) or bandwidth.value <= 0:
                        raise ValueError("KDE bandwidth must be finite and positive.")
                    run_kde(bandwidth=bandwidth.value, use_death_weights=weighted.value, data=data)
                status.value = "Finished on the current selection."
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                print(error)
                status.value = "Run unsuccessful. Check the message below."
            finally:
                run.disabled = False

    house.observe(load_house, names="value")
    for control in (minimum, spatial, xmin, xmax, ymin, ymax, method, k, weighted,
                    n_init, regularization, bandwidth):
        control.observe(invalidate, names="value")
    apply.on_click(apply_edit)
    reset.on_click(reset_data)
    preview.on_click(preview_data)
    run.on_click(run_model)
    load_house()
    for control in (minimum, xmin, xmax, ymin, ymax):
        control.layout = widgets.Layout(width="260px", max_width="100%")
        control.style.description_width = "85px"
    spatial.layout = widgets.Layout(width="auto")
    spatial.style.description_width = "initial"
    filters = widgets.VBox([
        widgets.HTML("<b>Filter houses</b>"),
        minimum,
        spatial,
        widgets.HBox([xmin, xmax], layout=widgets.Layout(flex_flow="row wrap", gap="12px")),
        widgets.HBox([ymin, ymax], layout=widgets.Layout(flex_flow="row wrap", gap="12px")),
        preview,
    ], layout=widgets.Layout(
        border="1px solid #d0d0d0", padding="12px", margin="12px 0",
        width="100%", max_width="580px", gap="8px",
    ))
    panel = widgets.VBox([
        widgets.HTML("<h3>Interactive house data demo</h3>"
                     "Edit a house and click Apply. Filters affect the selection when you preview or run. "
                     "Source files are never changed. K-means also runs its existing initialization and bootstrap experiments."),
        house, widgets.HBox([x, y, deaths]), widgets.HBox([included, apply, reset]),
        filters, data_output, widgets.HBox([method, k, weighted]),
        widgets.HBox([n_init, regularization, bandwidth]), run, status, result_output,
    ])
    display(panel)
    preview_data()
    return panel
