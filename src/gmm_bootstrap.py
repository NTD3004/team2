import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data, extract_coordinates
from gmm import fit_gmm


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


def fit_checked(X, weights, k, max_iter, tol, reg_covar):
    """Capture fit diagnostics instead of flooding the console."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = fit_gmm(
            X, weights, k=k, max_iter=max_iter, tol=tol,
            reg_covar=reg_covar, verbose=False,
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
    random_state=42, max_iter=500, tol=1e-6, reg_covar=1e-6,
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
        X, weights, k, max_iter, tol, reg_covar
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
                sample_X, sample_weights, k, max_iter, tol, reg_covar
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
    print(f"Mean principal-component effective death total: {totals.mean():.3f}")
    # ddof=0 matches the K-means bootstrap summary already used in this project.
    print(f"Standard deviation of effective death total: {totals.std(ddof=0):.3f}")
    print(f"Mean principal-component share of deaths: {shares.mean():.2f}%")
    print(f"Standard deviation of death share: {shares.std(ddof=0):.2f} percentage points")
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
    max_iter=500, tol=1e-6, reg_covar=1e-6,
):
    deaths, pumps = load_data()
    X, pump_X, weights = extract_coordinates(deaths, pumps)
    result = bootstrap_gmm(
        X, weights, pump_X, k=k, number_of_samples=number_of_samples,
        random_state=random_state, max_iter=max_iter, tol=tol, reg_covar=reg_covar,
    )
    plot_bootstrap_results(result)
    plt.show()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    args = parser.parse_args()
    bootstrap_results = run_gmm_bootstrap(
        k=args.k, number_of_samples=args.samples, random_state=args.seed,
        max_iter=args.max_iter, tol=args.tol, reg_covar=args.reg_covar,
    )
