import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import logsumexp

from data_utils import load_data, prepare_analysis_data
from kmeans import fit_kmeans


def initialize_from_kmeans(X, weights, k, reg_covar, n_init=50):
    """Initialize means, full covariances, and mixing coefficients."""
    # Use the configured number of K-means initializations.
    kmeans_model = fit_kmeans(X, weights, k, n_init=n_init)
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
    X, weights, k=8, max_iter=300, tol=1e-6, reg_covar=1e-6, verbose=True, n_init=50
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
        X, weights, k, reg_covar, n_init=n_init
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


def run_gmm(k=8, max_iter=300, tol=1e-6, reg_covar=1e-6,
            use_death_weights=True, n_init=50):
    deaths, pumps = load_data()
    X, pump_X, weights = prepare_analysis_data(deaths, pumps, use_death_weights)
    print(f"Fitting K={k} components to {len(X)} houses; {weights.sum():g} total weight.")
    print("K is supplied for comparison, not independently selected for GMM.")
    result = fit_gmm(X, weights, k, max_iter, tol, reg_covar, n_init=n_init)

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
    print("Effective weight per component (soft):", result["effective_deaths"])
    print("Weight per displayed cluster (hard):", hard_deaths)
    print("Principal component (zero-based):", principal)
    print("Principal Gaussian mean:", result["means"][principal])
    print("Principal effective weight total:", result["effective_deaths"][principal])
    print("Principal share of total weight:",
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



def validate_data(X, weights, pump_X, k):
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    pump_X = np.asarray(pump_X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or weights.shape != (len(X),):
        raise ValueError("Provide (N,2) coordinates and one weight per house.")
    if not np.isfinite(X).all() or not np.isfinite(weights).all():
        raise ValueError("Clean missing/invalid house coordinates and weights first.")
    if np.any(weights < 0):
        raise ValueError("Death weights cannot be negative.")
    mask = weights > 0
    X, weights = X[mask], weights[mask]
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(np.unique(X, axis=0)):
        raise ValueError("k must be positive and not exceed distinct affected locations.")
    if (pump_X.ndim != 2 or pump_X.shape[1] != 2 or len(pump_X) == 0
            or not np.isfinite(pump_X).all()):
        raise ValueError("Provide finite pump coordinates of shape (P,2).")
    return X, weights, pump_X


def summarize_principal_component(model, weights, pump_X):
    # Re-select by total SOFT death weight, never by the original label number.
    totals = np.sum(weights[:, None] * model["responsibilities"], axis=0)
    principal = int(np.argmax(totals))
    mean = model["means"][principal].copy()
    distances = np.linalg.norm(pump_X - mean, axis=1)
    pump = int(np.argmin(distances))
    return {
        "principal_component": principal,
        "principal_mean": mean,
        "effective_deaths": float(totals[principal]),
        "death_share_percent": float(100 * totals[principal] / weights.sum()),
        "nearest_pump": pump,
        "pump_distance": float(distances[pump]),
    }


def fit_checked(X, weights, k, max_iter, tol, reg_covar, n_init=50):
    """Capture fit diagnostics instead of flooding the console."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = fit_gmm(
            X, weights, k=k, max_iter=max_iter, tol=tol,
            reg_covar=reg_covar, verbose=False, n_init=n_init,
        )
    messages = [str(item.message) for item in captured]
    if not model["converged"]:
        return model, "nonconverged", messages
    if (not np.isfinite(model["log_likelihood"])
            or not np.isfinite(model["means"]).all()
            or not np.isfinite(model["responsibilities"]).all()):
        return model, "failed", messages + ["Nonfinite fitted values."]
    if any("Log-likelihood decreased" in message for message in messages):
        return model, "numerical_warning", messages
    return model, "ok", messages


def bootstrap_gmm(
    X, weights, pump_X, k=8, number_of_samples=100,
    random_state=42, max_iter=500, tol=1e-6, reg_covar=1e-6, n_init=50,
):
    """Return reference fit, per-sample records and successful-fit statistics.

    Displacements are measured from the full-data principal mean.
    Failed/nonconverged/numerically suspect fits are explicitly reported
    and excluded from numerical summaries. Frequency denominators are
    the number of successful fits, NOT silently the requested sample count.
    """
    if not isinstance(number_of_samples, (int, np.integer)) or number_of_samples < 1:
        raise ValueError("number_of_samples must be a positive integer.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tol) or not np.isfinite(reg_covar) or tol <= 0 or reg_covar <= 0:
        raise ValueError("tol and reg_covar must be finite and positive.")
    X, weights, pump_X = validate_data(X, weights, pump_X, k)
    print(f"Fitting reference GMM: {len(X)} houses, {weights.sum():g} deaths, K={k}.",
          flush=True)
    reference_model, status, messages = fit_checked(
        X, weights, k, max_iter, tol, reg_covar, n_init=n_init
    )
    if status != "ok":
        raise RuntimeError(
            f"Reference fit status: {status}. Resolve it before bootstrap. "
            "For nonconvergence, increase --max-iter. " + "; ".join(messages)
        )
    reference = summarize_principal_component(reference_model, weights, pump_X)
    print("Reference principal mean:", reference["principal_mean"])
    print("Reference nearest pump:", reference["nearest_pump"])
    print("Reference EM iterations:", reference_model["n_iter"])

    rng = np.random.default_rng(random_state)
    records = []
    for sample_number in range(1, number_of_samples + 1):
        # Same sequence as the earlier K-means bootstrap when input house
        # order, number of houses, seed, and number of samples are identical.
        indices = rng.choice(len(X), size=len(X), replace=True)
        sample_X, sample_weights = X[indices], weights[indices]
        record = {"sample": sample_number,
                  "sample_total_deaths": float(sample_weights.sum())}
        try:
            model, status, messages = fit_checked(
                sample_X, sample_weights, k, max_iter, tol, reg_covar, n_init=n_init
            )
            record.update(status=status, messages=messages, n_iter=model["n_iter"])
            if status == "ok":
                record.update(summarize_principal_component(model, sample_weights, pump_X))
                record["displacement"] = float(np.linalg.norm(
                    record["principal_mean"] - reference["principal_mean"]
                ))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            record.update(status="failed", messages=[str(error)])
        records.append(record)
        if sample_number == 1 or sample_number % 10 == 0 or record["status"] != "ok":
            print(f"Bootstrap {sample_number}/{number_of_samples}: {record['status']}",
                  flush=True)

    accepted = [record for record in records if record["status"] == "ok"]
    result = {
        "k": k, "requested_samples": number_of_samples,
        "reference": reference, "reference_model": reference_model,
        "records": records, "successful_samples": len(accepted),
        "number_of_pumps": len(pump_X),
        "centroid_displacements": np.array([r["displacement"] for r in accepted]),
        "nearest_pumps": np.array([r["nearest_pump"] for r in accepted], dtype=int),
        "principal_death_totals": np.array([r["effective_deaths"] for r in accepted]),
        "principal_death_shares": np.array([r["death_share_percent"] for r in accepted]),
    }
    print_bootstrap_summary(result)
    return result


def print_bootstrap_summary(result):
    successful = result["successful_samples"]
    requested = result["requested_samples"]
    print("\nGMM Bootstrap stability\n" + "-" * 50)
    print("Number of bootstrap samples:", requested)
    print("Successful converged fits:", successful)
    for status in ("nonconverged", "numerical_warning", "failed"):
        print(f"Fits with status {status}:",
              sum(r["status"] == status for r in result["records"]))
    if successful == 0:
        print("No successful fits: stability statistics cannot be calculated.")
        return
    if successful < requested:
        print("WARNING: summaries below use successful fits only.")
        print("Report exclusions; their results are unknown, not evidence of stability.")
        print("Increase --max-iter for nonconverged fits and rerun the same seed.")

    reference_pump = result["reference"]["nearest_pump"]
    same = int(np.sum(result["nearest_pumps"] == reference_pump))
    displacement = result["centroid_displacements"]
    totals = result["principal_death_totals"]
    shares = result["principal_death_shares"]
    print("Reference nearest pump:", reference_pump)
    print("Samples selecting the same pump:", same)
    print("Percentage selecting the same pump (successful fits):",
          f"{100 * same / successful:.2f}%")
    print(f"Same-pump / successful / attempted: {same} / {successful} / {requested}")
    print(f"Median principal-mean displacement: {np.median(displacement):.3f} metres")
    print(f"90th percentile displacement: {np.percentile(displacement, 90):.3f} metres")
    print(f"Mean principal-component effective weight total: {totals.mean():.3f}")
    # ddof=0 matches the K-means bootstrap summary already used in this project.
    print(f"Standard deviation of effective weight total: {totals.std(ddof=0):.3f}")
    print(f"Mean principal-component share of total weight: {shares.mean():.2f}%")
    print(f"Standard deviation of weight share: {shares.std(ddof=0):.2f} percentage points")
    print("\nNearest-pump frequency (successful fits):")
    for pump in range(result["number_of_pumps"]):
        count = int(np.sum(result["nearest_pumps"] == pump))
        print(f"Pump {pump}: {count}/{successful} ({100 * count / successful:.2f}%)")


def plot_bootstrap_results(result):
    if result["successful_samples"] == 0:
        return
    displacements = result["centroid_displacements"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(displacements, bins=15, edgecolor="black")
    median = np.median(displacements)
    percentile90 = np.percentile(displacements, 90)
    ax.axvline(median, color="red", linestyle="--",
               label=f"Median: {median:.3f} m")
    ax.axvline(percentile90, color="darkorange", linestyle=":",
               label=f"90th percentile: {percentile90:.3f} m")
    ax.set(title=f"GMM bootstrap: principal-mean displacement (K = {result['k']})",
           xlabel="Distance from full-data principal mean (metres)",
           ylabel="Number of successful bootstrap samples")
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = np.bincount(result["nearest_pumps"], minlength=result["number_of_pumps"])
    colours = ["crimson" if p == result["reference"]["nearest_pump"] else "steelblue"
               for p in range(len(counts))]
    ax.bar(np.arange(len(counts)), counts, color=colours)
    ax.set_xticks(np.arange(len(counts)), [f"P{p}" for p in range(len(counts))])
    for pump, count in enumerate(counts):
        ax.text(pump, count + 0.5, f"{count}", ha="center")
    ax.set_ylim(0, max(counts) * 1.15 + 1)
    ax.set(title=f"GMM bootstrap: nearest pump ({result['successful_samples']} successful fits)",
           xlabel="Pump index (red = full-data reference pump)",
           ylabel="Number of successful bootstrap samples")
    fig.tight_layout()


def run_gmm_bootstrap(
    k=8, number_of_samples=100, random_state=42,
    max_iter=500, tol=1e-6, reg_covar=1e-6, n_init=50, use_death_weights=True,
):
    deaths, pumps = load_data()
    X, pump_X, weights = prepare_analysis_data(deaths, pumps, use_death_weights)
    result = bootstrap_gmm(
        X, weights, pump_X, k=k, number_of_samples=number_of_samples,
        random_state=random_state, max_iter=max_iter, tol=tol, reg_covar=reg_covar, n_init=n_init,
    )
    plot_bootstrap_results(result)
    plt.show()
    return result
