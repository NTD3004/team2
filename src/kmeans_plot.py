import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def weighted_cluster_covariance(points, weights, centroid):
    """Frequency-weighted covariance, normalized by sum of death weights."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("House coordinates must have shape (N, 2).")
    if weights.shape != (len(points),) or centroid.shape != (2,):
        raise ValueError("Weights or centroid have incompatible dimensions.")
    if (not np.isfinite(points).all() or not np.isfinite(weights).all()
            or not np.isfinite(centroid).all() or np.any(weights < 0)
            or weights.sum() <= 0):
        raise ValueError("A cluster needs finite coordinates and positive total weight.")
    differences = points - centroid
    return (differences * weights[:, None]).T @ differences / weights.sum()


def plot_clusters(
    X, pump_X, weights, labels, centroids,
    nearest_pump, principal_cluster, k
):
    """Drop-in replacement for plot_clusters in the latest kmeans.py.

    Each ellipse has semi-axis lengths 2*sqrt(lambda_1), 2*sqrt(lambda_2).
    The eigenvectors determine its orientation. This uses the same contour
    level as the GMM plot, but covariance here is calculated AFTER K-means.
    Ellipses of singleton/collinear clusters can collapse to a point/line.
    Cluster colour numbers have no intrinsic correspondence across methods.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids, dtype=float)
    pump_X = np.asarray(pump_X, dtype=float)
    if labels.shape != (len(X),) or weights.shape != (len(X),):
        raise ValueError("One label and one weight are required per house.")
    if centroids.shape != (k, 2):
        raise ValueError("centroids must have shape (k, 2).")

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab20", k)
    ax.scatter(
        X[:, 0], X[:, 1], c=labels, cmap=cmap,
        vmin=-0.5, vmax=k - 0.5, s=8 + 5 * weights, alpha=0.7,
    )

    for cluster in range(k):
        mask = labels == cluster
        covariance = weighted_cluster_covariance(
            X[mask], weights[mask], centroids[cluster]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        # Clip tiny negative eigenvalues caused by floating-point roundoff.
        eigenvalues = np.maximum(eigenvalues[order], 0)
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(
            np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        )
        ax.add_patch(Ellipse(
            xy=centroids[cluster],
            width=4 * np.sqrt(eigenvalues[0]),
            height=4 * np.sqrt(eigenvalues[1]),
            angle=angle,
            fill=False,
            edgecolor=cmap(cluster),
            linewidth=1.5,
            label="2-SD weighted spread ellipses" if cluster == 0 else None,
        ))

    ax.scatter(
        centroids[:, 0], centroids[:, 1], marker="X", s=140, c="black",
        label="Cluster centroids",
    )
    ax.scatter(
        *centroids[principal_cluster], marker="X", s=240, c="gold",
        edgecolors="black", label="Principal cluster centroid", zorder=5,
    )
    ax.scatter(
        pump_X[:, 0], pump_X[:, 1], marker="x", s=80, c="blue", label="Pumps",
    )
    ax.scatter(
        *pump_X[nearest_pump], marker="x", s=160, c="red",
        linewidths=3, label="Nearest pump", zorder=6,
    )
    for index, point in enumerate(pump_X):
        ax.annotate(
            f"P{index}", point, xytext=(4, 4), textcoords="offset points"
        )

    ax.set(
        title=f"Weighted K-means of Cholera Death Locations (K = {k})",
        xlabel="COORD_X (metres)", ylabel="COORD_Y (metres)",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.tight_layout()
    # Let run_kmeans() display all figures together using its final plt.show().
    return fig, ax
