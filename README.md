Run `docker compose up --build` initially, then `docker compose up` on subsequent runs.
Open Jupyter and run these commands in its console from the project root:

```python
# K-means: automatic K, or explicit K and initialization count
%run src/main.py kmeans
%run src/main.py kmeans --k 8 --n-init 50

# GMM: component count and covariance regularization
%run src/main.py gmm --k 8 --reg-covar 0.001 --n-init 50

# GMM bootstrap (implementation is also in src/gmm.py)
%run src/main.py gmm --bootstrap --k 8 --samples 100 --max-iter 1000

# KDE density contour map
%run src/main.py kde --kernel gaussian --bandwidth 50

# Equal influence per affected house instead of death-count weights
%run src/main.py kmeans --k 8 --no-use-death-weights
%run src/main.py gmm --k 8 --no-use-death-weights
%run src/main.py kde --no-use-death-weights
```

In a terminal with dependencies installed, replace `%run` with `python`.
Use `python src/main.py METHOD --help` for method-specific options.

Death weighting is enabled by default (`--use-death-weights`). Both modes use
only houses with positive death counts. With weighting disabled, each affected
house contributes one unit; principal clusters and bootstrap totals then measure
houses rather than deaths. Pump indices follow the input data order.

`--n-init` defaults to 50 and controls K-means restarts per fit, including
K selection, bootstrap, and GMM initialization. The K-means initialization
experiment plots this many separate single-initialization runs.
`--reg-covar` defaults to 1e-6 and adds to each GMM covariance diagonal.
GMM bootstrap uses the same controls as its reference fit; increase `--max-iter`
if fits do not converge. Bootstrap summaries explicitly exclude unsuccessful fits.
K-means runs its initialization experiment and 100 bootstrap samples automatically.

KDE already provides a filled density contour map, now with a density colorbar
and equal spatial axis scaling. KDE has no K parameter; bandwidth controls smoothing.
For comparison, use three maps with identical spatial limits and pump markers:
K-means assignments and spread ellipses, GMM assignments and covariance ellipses,
and KDE density contours. A useful next addition is to mark the KDE density maximum
and its nearest pump, alongside the principal K-means centroid and GMM mean.
These summarize different properties: a density peak versus a component center.

## Interactive house data demo

Run `docker compose up --build`, open Jupyter, and open **Demo.ipynb** from
the project root. Run its cell to display the editor. It uses
[ipywidgets](https://ipywidgets.readthedocs.io/en/latest/reference/ipywidgets.html).
For an existing local environment, install the updated `requirements.txt` first.

1. Select a house by ID/row, edit its X/Y coordinates or death count, and click
   **Apply house edit**. Uncheck **Include house** to exclude that observation.
2. Set a minimum death count and optionally enable coordinate bounds. Click
   **Preview selection** to see the selected houses, totals, and map.
3. Choose a method and its settings, then click **Run on selection**. Model runs
   use the applied edits and current filters; zero-death houses are excluded
   from fitting in both weighting modes. K and initializations apply to K-means
   and GMM; regularization applies to GMM; bandwidth applies to KDE.
4. Click **Reset all data** to restore the original houses and clear filters.

Edits are kept only in the notebook session: the source shapefiles are never
written. Pending house edits require **Apply house edit** before running.
Changing applied data, filters, or model settings clears old results.
The preview table shows at most 100 rows; fitting uses the entire selection.
K-means retains its full initialization experiment and 100 bootstrap samples,
so it can take longer than the other methods. This demo requires a live Jupyter
kernel; a static notebook export cannot execute its controls.

Validation: `python -m unittest discover -s tests -v`.
