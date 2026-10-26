import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from data_utils import load_data, prepare_analysis_data


def gaussian_kernel(u):
    return (2 * np.pi) ** (-u.shape[-1] / 2) * np.exp(
        -0.5 * np.sum(u ** 2, axis=-1)
    )


def box_kernel(u):
    return np.all(np.abs(u) <= 0.5, axis=-1).astype(float)


def estimate_density(point, X, weights, bandwidth, kernel):
    return np.sum(weights * kernel((point - X) / bandwidth)) / (
        weights.sum() * bandwidth ** X.shape[1]
    )


def validate_data(X, weights):
    X, weights = np.asarray(X, dtype=float), np.asarray(weights, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or weights.shape != (len(X),):
        raise ValueError("Expected (N,2) house coordinates and one weight per house.")
    if not np.isfinite(X).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Coordinates and weights must be finite; weights cannot be negative.")
    mask = weights > 0
    X, weights = X[mask], weights[mask]
    if len(X) < 2:
        raise ValueError("Leave-one-house-out selection needs at least two affected houses.")
    return X, weights


def select_bandwidth(X, weights, bandwidths, kernel_type="gaussian", verbose=True):
    """Maximize sum_i w_i log p_{-i,h}(x_i), removing the entire house.

    Each held-out density is normalized by W-w_i. Zero-weight houses
    contribute nothing. A box kernel with an uncovered house scores -inf;
    no density floor is inserted into the specified likelihood objective.
    """
    X, weights = validate_data(X, weights)
    bandwidths = np.asarray(bandwidths, dtype=float)
    if (bandwidths.ndim != 1 or len(bandwidths) == 0
            or not np.isfinite(bandwidths).all() or np.any(bandwidths <= 0)):
        raise ValueError("Candidate bandwidths must be finite positive numbers.")
    if kernel_type not in ("gaussian", "box"):
        raise ValueError("Unknown kernel.")
    differences = X[:, None, :] - X[None, :, :]
    squared_distances = np.sum(differences ** 2, axis=2)
    max_distances = np.max(np.abs(differences), axis=2)
    scores = []
    for h in bandwidths:
        if kernel_type == "gaussian":
            log_kernel = -np.log(2 * np.pi) - 2 * np.log(h) - squared_distances / (2 * h ** 2)
        else:
            log_kernel = np.where(max_distances <= h / 2, -2 * np.log(h), -np.inf)
        np.fill_diagonal(log_kernel, -np.inf)
        log_density = logsumexp(log_kernel + np.log(weights)[None, :], axis=1)
        log_density -= np.log(weights.sum() - weights)
        score = float(weights @ log_density)
        scores.append(score)
        if verbose:
            print(f"h = {h:.6g} metres; weighted LOO log-likelihood = {score:.6f}")
    scores = np.asarray(scores)
    if not np.isfinite(scores).any():
        raise ValueError("No candidate has finite LOO likelihood. Try larger bandwidths.")
    best = int(np.argmax(scores))
    if verbose and bandwidths[best] in (bandwidths.min(), bandwidths.max()):
        warnings.warn("Selected h is at the candidate-range boundary; consider extending the range.")
    return float(bandwidths[best]), scores


def create_grid(X, pump_X, resolution=100, padding=30):
    if resolution < 2:
        raise ValueError("Grid resolution must be at least 2.")
    points = np.vstack((X, pump_X))
    low, high = points.min(axis=0) - padding, points.max(axis=0) + padding
    return np.meshgrid(np.linspace(low[0], high[0], resolution),
                       np.linspace(low[1], high[1], resolution))


def density_on_grid(X, weights, bandwidth, kernel_type, xx, yy):
    kernel = gaussian_kernel if kernel_type == "gaussian" else box_kernel
    points = np.column_stack((xx.ravel(), yy.ravel()))
    density = np.empty(len(points))
    for start in range(0, len(points), 256):
        u = (points[start:start + 256, None, :] - X[None, :, :]) / bandwidth
        density[start:start + 256] = kernel(u) @ weights / (weights.sum() * bandwidth ** 2)
    density = density.reshape(xx.shape)
    # np.argmax selects the first grid location in a tie (common for box KDE).
    index = np.unravel_index(np.argmax(density), density.shape)
    return density, np.array([xx[index], yy[index]])


def bootstrap_kde(X, weights, pump_X, bandwidths, kernel_type, xx, yy,
                  reference_hotspot, samples=100, seed=42):
    """Resample houses, retaining their death weights, and reselect h each time.

    Repeated sampled rows remain separate bootstrap observations: leaving one
    out removes that occurrence, while other sampled copies remain in training.
    """
    rng = np.random.default_rng(seed)
    records = []
    for sample in range(samples):
        indices = rng.choice(len(X), len(X), replace=True)
        sample_X, sample_weights = X[indices], weights[indices]
        try:
            h, _ = select_bandwidth(sample_X, sample_weights, bandwidths, kernel_type, False)
            _, hotspot = density_on_grid(sample_X, sample_weights, h, kernel_type, xx, yy)
            records.append(dict(status="ok", bandwidth=h, hotspot=hotspot,
                                displacement=float(np.linalg.norm(hotspot - reference_hotspot)),
                                nearest_pump=int(np.argmin(np.linalg.norm(pump_X - hotspot, axis=1)))))
        except ValueError as error:
            records.append(dict(status="failed", error=str(error)))
        if (sample + 1) % 10 == 0 or sample == 0:
            print(f"Bootstrap {sample + 1}/{samples}: {records[-1]['status']}", flush=True)
    return records


def run_kde(kernel_type="gaussian", bandwidths=None, samples=100, seed=42, resolution=100,
            use_death_weights=True, data=None):
    if samples < 0:
        raise ValueError("Bootstrap sample count cannot be negative.")
    if bandwidths is None:
        bandwidths = np.geomspace(5, 500, 25)
    deaths, pumps = load_data() if data is None else data
    X, pump_X, weights = prepare_analysis_data(deaths, pumps, use_death_weights)
    X, weights = validate_data(X, weights)
    pump_X = np.asarray(pump_X, dtype=float)
    if (pump_X.ndim != 2 or pump_X.shape[1] != 2 or not len(pump_X)
            or not np.isfinite(pump_X).all()):
        raise ValueError("Pump coordinates must be a nonempty finite (P,2) matrix.")
    print(f"Kernel: {kernel_type}; {len(X)} affected houses; {weights.sum():g} deaths.")
    h, scores = select_bandwidth(X, weights, bandwidths, kernel_type)
    xx, yy = create_grid(X, pump_X, resolution)
    density, hotspot = density_on_grid(X, weights, h, kernel_type, xx, yy)
    distances = np.linalg.norm(pump_X - hotspot, axis=1)
    nearest = int(np.argmin(distances))
    print(f"Selected h: {h:.6g} metres")
    print("Hotspot (grid maximum):", hotspot)
    print(f"Grid spacing: {xx[0, 1] - xx[0, 0]:.3f} x {yy[1, 0] - yy[0, 0]:.3f} metres")
    print(f"Nearest pump: P{nearest}; distance: {distances[nearest]:.3f} metres")
    print("Tied grid maxima use the first grid point; increase --resolution to check sensitivity.")

    fig, ax = plt.subplots(figsize=(9, 5))
    candidates = np.asarray(bandwidths, dtype=float)
    order = np.argsort(candidates)
    # A box-kernel candidate can have -inf likelihood; leave a gap rather
    # than replacing that score with an arbitrary finite value.
    plotted_scores = np.where(np.isfinite(scores), scores, np.nan)
    ax.plot(candidates[order], plotted_scores[order], marker="o",
            label="Leave-one-house-out score")
    best_score = float(np.max(scores))
    ax.scatter(h, best_score, color="red", s=90, zorder=5)
    ax.annotate(f"Best h = {h:.3g} m\nScore = {best_score:.2f}",
                (h, best_score), xytext=(12, -36), textcoords="offset points",
                arrowprops={"arrowstyle": "->"})
    ax.axvline(h, color="red", linestyle="--", label=f"Selected h = {h:.3g} m")
    ax.set(xscale="log", xlabel="Bandwidth h (metres)", ylabel="Weighted LOO log-likelihood",
           title=f"KDE bandwidth selection ({kernel_type}; higher is better)")
    ax.grid(alpha=0.3)
    excluded = int(np.sum(~np.isfinite(scores)))
    if excluded:
        ax.text(0.02, 0.03, f"{excluded} candidates have score -infinity (not plotted)",
                transform=ax.transAxes, fontsize=9)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    contours = ax.contourf(xx, yy, density, levels=20)
    fig.colorbar(contours, ax=ax, label="Estimated density (1/m²)")
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4)
    ax.scatter(pump_X[:, 0], pump_X[:, 1], marker="x", s=80, label="Pumps")
    ax.scatter(*hotspot, marker="*", color="red", s=180, label="Grid hotspot")
    ax.scatter(*pump_X[nearest], marker="x", color="red", s=120, label="Nearest pump")
    for index, point in enumerate(pump_X):
        ax.annotate(f"P{index}", point)
    ax.set(title=f"Weighted KDE ({kernel_type}, h = {h:.3g} m)",
           xlabel="COORD_X (metres)", ylabel="COORD_Y (metres)", aspect="equal")
    ax.legend()
    fig.tight_layout()

    records = bootstrap_kde(X, weights, pump_X, bandwidths, kernel_type, xx, yy,
                            hotspot, samples, seed)
    accepted = [r for r in records if r["status"] == "ok"]
    print(f"Bootstrap successful / attempted: {len(accepted)} / {samples}")
    if len(accepted) < samples:
        print("Excluded failures:", [r['error'] for r in records if r['status'] != 'ok'])
    if accepted:
        displacements = np.array([r["displacement"] for r in accepted])
        counts = np.bincount([r["nearest_pump"] for r in accepted], minlength=len(pump_X))
        print(f"Median hotspot displacement: {np.median(displacements):.3f} metres")
        print(f"90th percentile displacement: {np.percentile(displacements, 90):.3f} metres")
        for index, count in enumerate(counts):
            print(f"P{index}: {count}/{len(accepted)} ({100 * count / len(accepted):.2f}%)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(displacements, bins=15, edgecolor="black")
        axes[0].set(xlabel="Hotspot displacement (metres)", ylabel="Successful samples")
        axes[1].bar(np.arange(len(pump_X)), counts)
        axes[1].set(xticks=np.arange(len(pump_X)), xlabel="Nearest pump index",
                    ylabel="Successful samples")
        fig.suptitle("KDE house bootstrap (bandwidth reselected per sample)")
        fig.tight_layout()
    plt.show()
    return dict(bandwidth=h, scores=scores, hotspot=hotspot, nearest_pump=nearest,
                density=density, bootstrap_records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weighted KDE with leave-one-house-out bandwidth selection.")
    parser.add_argument("--kernel", choices=["gaussian", "box"], default="gaussian")
    parser.add_argument("--bandwidths", type=float, nargs="+", default=None,
                        help="Candidate h values in metres (default: 25 log-spaced values from 5 to 500).")
    parser.add_argument("--samples", type=int, default=100, help="House bootstrap samples; 0 skips bootstrap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=100, help="Grid points per axis (default: 100).")
    args = parser.parse_args()
    run_kde(args.kernel, args.bandwidths, args.samples, args.seed, args.resolution)
