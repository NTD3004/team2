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
