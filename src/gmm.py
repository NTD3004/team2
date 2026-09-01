import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import logsumexp

from data_utils import load_data, extract_coordinates
from kmeans import fit_kmeans


def initialize_from_kmeans(X, weights, k, reg_covar):
    """Initialize means, full covariances, and mixing coefficients."""
    # Reuse your existing function (50 K-means initializations).
    kmeans_model = fit_kmeans(X, weights, k)
    means = kmeans_model.cluster_centers_.copy()
    covariances = np.empty((k, X.shape[1], X.shape[1]))
    cluster_weights = np.zeros(k)

    for component in range(k):
        mask = kmeans_model.labels_ == component
        component_weights = weights[mask]
        cluster_weights[component] = component_weights.sum()
        if cluster_weights[component] <= 0:
            raise ValueError("K-means produced an empty component. Reduce k.")

        differences = X[mask] - means[component]
        covariance = (
            (differences * component_weights[:, None]).T @ differences
        ) / cluster_weights[component]
        covariances[component] = (
            covariance + reg_covar * np.eye(X.shape[1])
        )

    mixing_coefficients = cluster_weights / weights.sum()
    return means, covariances, mixing_coefficients


def expectation_step(X, weights, means, covariances, mixing_coefficients):
    """Compute responsibilities and weighted log-likelihood in log space.

    gamma[i,k] = pi[k] * N(X[i] | mu[k], Sigma[k]) / p(X[i]).
    House death counts do not multiply gamma directly: they enter the
    objective and the M step. Each row of gamma sums to one.
    """
    number_of_houses, dimensions = X.shape
    k = len(means)
    log_joint = np.empty((number_of_houses, k))

    for component in range(k):
        differences = X - means[component]
        # Sigma = L @ L.T. Avoid explicitly inverting Sigma.
        L = np.linalg.cholesky(covariances[component])
        transformed = np.linalg.solve(L, differences.T)
        squared_distance = np.sum(transformed ** 2, axis=0)
        log_determinant = 2 * np.log(np.diag(L)).sum()

        log_gaussian = -0.5 * (
            dimensions * np.log(2 * np.pi)
            + log_determinant
            + squared_distance
        )
        log_joint[:, component] = (
            np.log(mixing_coefficients[component]) + log_gaussian
        )

    log_density = logsumexp(log_joint, axis=1)
    responsibilities = np.exp(log_joint - log_density[:, None])
    log_likelihood = float(np.dot(weights, log_density))
    return responsibilities, log_likelihood


def maximization_step(X, weights, responsibilities, reg_covar):
    """Update all GMM parameters using weighted responsibilities."""
    weighted_responsibilities = weights[:, None] * responsibilities
    effective_deaths = weighted_responsibilities.sum(axis=0)
    if np.any(effective_deaths <= np.finfo(float).eps * weights.sum()):
        raise ValueError(
            "A component has negligible weight. Try fewer components "
            "or a larger covariance regularizer."
        )

    mixing_coefficients = effective_deaths / weights.sum()
    means = (weighted_responsibilities.T @ X) / effective_deaths[:, None]
    k, dimensions = means.shape
    covariances = np.empty((k, dimensions, dimensions))

    for component in range(k):
        differences = X - means[component]
        covariance = (
            (differences * weighted_responsibilities[:, component, None]).T
            @ differences
        ) / effective_deaths[component]
        covariances[component] = (
            covariance + reg_covar * np.eye(dimensions)
        )

    return means, covariances, mixing_coefficients


def fit_gmm(
    X, weights, k=8, max_iter=300, tol=1e-6, reg_covar=1e-6, verbose=True
):
    """Fit weighted GMM with full covariances using K-means initialization.

    X: (N,2) house coordinates; weights: (N,) positive death counts.
    tol: absolute change in average log-likelihood per death.
    reg_covar: positive diagonal regularizer in squared coordinate units.
    History includes initialization (iteration 0) and every EM update.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or len(X) == 0:
        raise ValueError("X must be a nonempty (N,2) coordinate matrix.")
    if weights.shape != (len(X),):
        raise ValueError("Provide one death weight per house.")
    if not np.isfinite(X).all() or not np.isfinite(weights).all():
        raise ValueError("Coordinates and death weights must be finite.")
    if np.any(weights <= 0):
        raise ValueError("Filter houses to strictly positive death counts first.")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(np.unique(X, axis=0)):
        raise ValueError("k must be between 1 and the number of distinct locations.")
    if max_iter < 1 or tol <= 0 or reg_covar <= 0:
        raise ValueError("max_iter, tol, and reg_covar must be positive.")

    means, covariances, mixing_coefficients = initialize_from_kmeans(
        X, weights, k, reg_covar
    )
    initial_parameters = {
        "means": means.copy(),
        "covariances": covariances.copy(),
        "mixing_coefficients": mixing_coefficients.copy(),
    }
    responsibilities, likelihood = expectation_step(
        X, weights, means, covariances, mixing_coefficients
    )
    history = [likelihood]
    converged = False
    if verbose:
        print("\nInitial K-means means:\n", means)
        print("Initial mixing coefficients:", mixing_coefficients)
        print("Initial covariance matrices:\n", covariances)
        print(f"\nIteration 0: weighted log-likelihood = {likelihood:.6f}")

    for iteration in range(1, max_iter + 1):
        means, covariances, mixing_coefficients = maximization_step(
            X, weights, responsibilities, reg_covar
        )
        responsibilities, likelihood = expectation_step(
            X, weights, means, covariances, mixing_coefficients
        )
        change = (likelihood - history[-1]) / weights.sum()
        history.append(likelihood)
        if verbose and (iteration == 1 or iteration % 10 == 0):
            print(
                f"Iteration {iteration}: weighted log-likelihood = "
                f"{likelihood:.6f}; change per death = {change:.8f}"
            )
        # Diagonal regularization is a numerical modification to plain EM;
        # do not silently report convergence after a substantial decrease.
        if change < -tol:
            warnings.warn(
                "Log-likelihood decreased beyond tolerance; inspect covariance "
                "regularization and numerical conditioning.", RuntimeWarning
            )
        if abs(change) < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            "EM reached max_iter without meeting the convergence tolerance.",
            RuntimeWarning,
        )
    return {
        "means": means,
        "covariances": covariances,
        "mixing_coefficients": mixing_coefficients,
        "responsibilities": responsibilities,
        "labels": np.argmax(responsibilities, axis=1),
        "effective_deaths": (weights[:, None] * responsibilities).sum(axis=0),
        "log_likelihood": likelihood,
        "log_likelihood_history": np.array(history),
        "initial_parameters": initial_parameters,
        "converged": converged,
        "n_iter": iteration,
    }


def plot_gmm(X, weights, pump_X, result, principal, nearest_pump):
    """Hard colours for display; component totals use soft responsibilities."""
    means = result["means"]
    k = len(means)
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab20", k)
    ax.scatter(
        X[:, 0], X[:, 1], c=result["labels"], cmap=cmap,
        vmin=-0.5, vmax=k - 0.5, s=8 + 5 * weights, alpha=0.7,
    )
    for component, (mean, covariance) in enumerate(
        zip(means, result["covariances"])
    ):
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        ax.add_patch(Ellipse(
            mean, width=4 * np.sqrt(values[0]), height=4 * np.sqrt(values[1]),
            angle=angle, fill=False, edgecolor=cmap(component), linewidth=1.5,
            label="2-SD component ellipses" if component == 0 else None,
        ))
    ax.scatter(means[:, 0], means[:, 1], marker="X", s=140, c="black",
               label="Gaussian means")
    ax.scatter(*means[principal], marker="X", s=240, c="gold",
               edgecolors="black", label="Principal component mean", zorder=5)
    ax.scatter(pump_X[:, 0], pump_X[:, 1], marker="x", s=80, c="blue",
               label="Pumps")
    ax.scatter(*pump_X[nearest_pump], marker="x", s=160, c="red",
               linewidths=3, label="Nearest pump", zorder=6)
    for index, point in enumerate(pump_X):
        ax.annotate(f"P{index}", point, xytext=(4, 4), textcoords="offset points")
    ax.set(title=f"Weighted GMM initialized by K-means (K = {k})",
           xlabel="COORD_X (metres)", ylabel="COORD_Y (metres)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    history = result["log_likelihood_history"]
    ax.plot(np.arange(len(history)), history)
    ax.set(title="Weighted EM convergence (higher is better)",
           xlabel="EM iteration (0 = K-means initialization)",
           ylabel="Weighted log-likelihood")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def run_gmm(k=8, max_iter=300, tol=1e-6, reg_covar=1e-6):
    deaths, pumps = load_data()
    X, pump_X, weights = extract_coordinates(deaths, pumps)
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    pump_X = np.asarray(pump_X, dtype=float)
    if (not np.isfinite(X).all() or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("Clean invalid coordinates/death counts before fitting.")
    if (pump_X.ndim != 2 or pump_X.shape[1] != 2 or len(pump_X) == 0
            or not np.isfinite(pump_X).all()):
        raise ValueError("Pump coordinates must be a nonempty finite (P,2) matrix.")
    mask = weights > 0
    X, weights = X[mask], weights[mask]
    print(f"Fitting K={k} components to {len(X)} houses; {weights.sum():g} deaths.")
    print("K is supplied for comparison, not independently selected for GMM.")
    result = fit_gmm(X, weights, k, max_iter, tol, reg_covar)

    principal = int(np.argmax(result["effective_deaths"]))
    distances = np.linalg.norm(pump_X - result["means"][principal], axis=1)
    nearest_pump = int(np.argmin(distances))
    hard_deaths = np.bincount(result["labels"], weights=weights, minlength=k)
    print("\nFinal GMM result")
    print("Converged:", result["converged"], "| EM iterations:", result["n_iter"])
    print("Final weighted log-likelihood:", result["log_likelihood"])
    print("Final Gaussian means:\n", result["means"])
    print("Final covariance matrices:\n", result["covariances"])
    print("Final mixing coefficients:", result["mixing_coefficients"])
    print("Effective deaths per component (soft):", result["effective_deaths"])
    print("Deaths per displayed cluster (hard):", hard_deaths)
    print("Principal component (zero-based):", principal)
    print("Principal Gaussian mean:", result["means"][principal])
    print("Principal effective death total:", result["effective_deaths"][principal])
    print("Principal share of deaths:",
          f"{100 * result['effective_deaths'][principal] / weights.sum():.2f}%")
    print("Nearest pump index (not a verified pump name):", nearest_pump)
    print("Distance from principal mean to nearest pump:",
          f"{distances[nearest_pump]:.3f} metres")
    print("The principal Gaussian mean need not be the mixture's density mode.")
    print("Pump proximity is spatial association, not evidence of causation.")

    result.update(principal_component=principal, nearest_pump=nearest_pump,
                  pump_distances=distances)
    plot_gmm(X, weights, pump_X, result, principal, nearest_pump)
    plt.show()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    args = parser.parse_args()
    run_gmm(args.k, args.max_iter, args.tol, args.reg_covar)
