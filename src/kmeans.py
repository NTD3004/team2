import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from data_utils import load_data, prepare_analysis_data
from kmeans_plot import plot_clusters

def evaluate_bootstrap_stability(
    X,
    weights,
    pump_X,
    reference_model,
    k,
    number_of_samples=100,
    random_state=42,
    n_init=50
):
    """
    Evaluate stability by resampling houses with replacement.

    Each bootstrap sample contains the same number of houses
    as the original dataset, but some houses may occur more
    than once and some may not occur at all.
    """

    rng = np.random.default_rng(
        random_state
    )

    reference_labels = reference_model.labels_
    reference_centroids = (
        reference_model.cluster_centers_
    )

    reference_principal_cluster, _ = (
        find_principal_cluster(
            reference_labels,
            weights,
            k
        )
    )

    reference_centroid = reference_centroids[
        reference_principal_cluster
    ]

    reference_nearest_pump, _ = (
        find_nearest_pump(
            reference_centroid,
            pump_X
        )
    )

    centroid_displacements = []
    principal_death_totals = []
    nearest_pumps = []

    number_of_houses = len(X)

    for bootstrap_number in range(
        number_of_samples
    ):
        # Resample house indices with replacement.
        sample_indices = rng.choice(
            number_of_houses,
            size=number_of_houses,
            replace=True
        )

        bootstrap_X = X[sample_indices]
        bootstrap_weights = weights[
            sample_indices
        ]

        bootstrap_model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            random_state=(
                random_state
                + bootstrap_number
            )
        )

        bootstrap_model.fit(
            bootstrap_X,
            sample_weight=bootstrap_weights
        )

        bootstrap_labels = (
            bootstrap_model.labels_
        )

        bootstrap_centroids = (
            bootstrap_model.cluster_centers_
        )

        principal_cluster, cluster_deaths = (
            find_principal_cluster(
                bootstrap_labels,
                bootstrap_weights,
                k
            )
        )

        principal_centroid = (
            bootstrap_centroids[
                principal_cluster
            ]
        )

        nearest_pump, _ = find_nearest_pump(
            principal_centroid,
            pump_X
        )

        displacement = np.linalg.norm(
            principal_centroid
            - reference_centroid
        )

        centroid_displacements.append(
            displacement
        )

        principal_death_totals.append(
            cluster_deaths[
                principal_cluster
            ]
        )

        nearest_pumps.append(
            nearest_pump
        )

    centroid_displacements = np.array(
        centroid_displacements
    )

    principal_death_totals = np.array(
        principal_death_totals
    )

    nearest_pumps = np.array(
        nearest_pumps
    )

    same_pump_count = np.sum(
        nearest_pumps
        == reference_nearest_pump
    )

    same_pump_percentage = (
        same_pump_count
        / number_of_samples
        * 100
    )

    print("\nBootstrap stability")
    print("-" * 50)

    print(
        "Number of bootstrap samples:",
        number_of_samples
    )

    print(
        "Reference nearest pump:",
        reference_nearest_pump
    )

    print(
        "Samples selecting the same pump:",
        same_pump_count
    )

    print(
        "Percentage selecting the same pump:",
        f"{same_pump_percentage:.2f}%"
    )

    print(
        "Median centroid displacement:",
        f"{np.median(centroid_displacements):.3f} metres"
    )

    print(
        "90th percentile centroid displacement:",
        f"{np.percentile(centroid_displacements, 90):.3f} metres"
    )

    print(
        "Mean principal-cluster weight total:",
        f"{principal_death_totals.mean():.3f}"
    )

    print(
        "Standard deviation of weight total:",
        f"{principal_death_totals.std():.3f}"
    )

    print("\nNearest-pump frequency:")

    unique_pumps, pump_counts = np.unique(
        nearest_pumps,
        return_counts=True
    )

    for pump_index, count in zip(
        unique_pumps,
        pump_counts
    ):
        percentage = (
            count
            / number_of_samples
            * 100
        )

        print(
            f"Pump {pump_index}: "
            f"{count}/{number_of_samples} "
            f"({percentage:.2f}%)"
        )

    return {
        "reference_pump": reference_nearest_pump,
        "nearest_pumps": nearest_pumps,
        "centroid_displacements": (
            centroid_displacements
        ),
        "principal_death_totals": (
            principal_death_totals
        )
    }

def demonstrate_initializations(
    X,
    weights,
    k,
    number_of_runs=50
):
    """
    Run K-means using different initial centroid sets
    and plot the resulting objective values.
    """

    objective_values = []

    print("\nK-means initialization experiment")
    print("-" * 50)

    for run in range(number_of_runs):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=1,
            random_state=run
        )

        model.fit(
            X,
            sample_weight=weights
        )

        objective = model.inertia_

        objective_values.append(
            objective
        )

        print(
            f"Initialization {run + 1:2d}: "
            f"J = {objective:.3f}"
        )

    objective_values = np.array(
        objective_values
    )

    best_run_index = int(
        np.argmin(objective_values)
    )

    best_run_number = best_run_index + 1

    minimum_J = objective_values[
        best_run_index
    ]

    maximum_J = np.max(
        objective_values
    )

    mean_J = np.mean(
        objective_values
    )

    standard_deviation_J = np.std(
        objective_values
    )

    percentage_difference = (
        (maximum_J - minimum_J)
        / minimum_J
        * 100
    )

    print("-" * 50)

    print(
        "Best initialization:",
        best_run_number
    )

    print(
        "Smallest J:",
        round(minimum_J, 3)
    )

    print(
        "Largest J:",
        round(maximum_J, 3)
    )

    print(
        "Mean J:",
        round(mean_J, 3)
    )

    print(
        "Standard deviation of J:",
        round(standard_deviation_J, 3)
    )

    print(
        "Largest J is higher than smallest J by:",
        f"{percentage_difference:.2f}%"
    )

    # Create the graph.
    initialization_numbers = np.arange(
        1,
        number_of_runs + 1
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        initialization_numbers,
        objective_values,
        marker="o",
        markersize=4,
        linewidth=1,
        color="steelblue",
        label="Objective value for each initialization"
    )

    # Highlight the best initialization.
    plt.scatter(
        best_run_number,
        minimum_J,
        color="red",
        s=100,
        zorder=5,
        label=(
            f"Best initialization: "
            f"{best_run_number}"
        )
    )

    # Draw the minimum-J reference line.
    plt.axhline(
        y=minimum_J,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Minimum J = "
            f"{minimum_J:,.0f}"
        )
    )

    # Draw the mean reference line.
    plt.axhline(
        y=mean_J,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Mean J = "
            f"{mean_J:,.0f}"
        )
    )

    plt.title(
        f"Sensitivity of Weighted K-means "
        f"to Initialization (k = {k})"
    )

    plt.xlabel(
        "Initialization number"
    )

    plt.ylabel(
        "Weighted objective value J"
    )

    plt.xticks(
        np.arange(
            1,
            number_of_runs + 1,
            2
        )
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    return objective_values


def fit_kmeans(X, weights, k, n_init=50):
    """
    Fit weighted K-means for a specified number of clusters.
    """

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        random_state=42
    )

    model.fit(
        X,
        sample_weight=weights
    )

    return model


def calculate_average_radius(X, weights, model):
    """
    Calculate the weighted average Euclidean distance
    from each house to its assigned cluster centroid.
    """

    labels = model.labels_
    centroids = model.cluster_centers_

    # Select the assigned centroid for every house.
    assigned_centroids = centroids[labels]

    # Euclidean distance from every house to its centroid.
    distances = np.linalg.norm(
        X - assigned_centroids,
        axis=1
    )

    return np.average(
        distances,
        weights=weights
    )


def reduction_per_cluster(
    old_radius,
    new_radius,
    old_k,
    new_k
):
    """
    Calculate the average decrease in radius for each
    additional cluster introduced.
    """

    return (
        old_radius - new_radius
    ) / (new_k - old_k)


def select_k(
    X,
    weights,
    max_k=32,
    small_change=2.0,
    n_init=50
):
    """
    First evaluate k = 1, 2, 4, 8, ...
    If doubling k produces only a small improvement,
    search more carefully inside the preceding interval.
    """

    models = {}
    radii = {}

    # K cannot exceed the number of observations.
    max_k = min(max_k, len(X))

    # Construct k = 1, 2, 4, 8, ...
    k_values = [1]

    while k_values[-1] < max_k:
        next_k = min(
            2 * k_values[-1],
            max_k
        )

        if next_k == k_values[-1]:
            break

        k_values.append(next_k)

    search_range = None

    print("\nEvaluating k using trial and error")
    print("-" * 50)

    for index, current_k in enumerate(k_values):
        model = fit_kmeans(
            X,
            weights,
            current_k,
            n_init=n_init
        )

        models[current_k] = model

        radii[current_k] = calculate_average_radius(
            X,
            weights,
            model
        )

        print(
            f"k = {current_k}, "
            f"average radius = "
            f"{radii[current_k]:.3f} metres"
        )

        if index == 0:
            continue

        previous_k = k_values[index - 1]

        reduction = reduction_per_cluster(
            radii[previous_k],
            radii[current_k],
            previous_k,
            current_k
        )

        print(
            f"Reduction from {previous_k} to {current_k} "
            f"= {reduction:.3f} metres "
            f"per additional cluster"
        )

        if reduction < small_change:
            # Trial and error: if little improvement occurs between
            # v and 2v, search between v/2 and v.
            left = max(1, previous_k // 2)
            right = previous_k

            search_range = (left, right)
            break

    if search_range is None:
        print(
            f"\nNo sufficiently small reduction was found "
            f"up to k = {max_k}."
        )

        print(
            "The trial and error procedure does not automatically "
            "justify a value of k in this range."
        )

        return None, models, radii

    left, right = search_range

    print(
        f"\nRefining the search between "
        f"k = {left} and k = {right}"
    )

    # Refine the interval.
    while right - left > 1:
        middle = (left + right) // 2

        if middle not in models:
            models[middle] = fit_kmeans(
                X,
                weights,
                middle,
                n_init=n_init
            )

            radii[middle] = calculate_average_radius(
                X,
                weights,
                models[middle]
            )

        if right not in models:
            models[right] = fit_kmeans(
                X,
                weights,
                right,
                n_init=n_init
            )

            radii[right] = calculate_average_radius(
                X,
                weights,
                models[right]
            )

        reduction = reduction_per_cluster(
            radii[middle],
            radii[right],
            middle,
            right
        )

        print(
            f"Compare k = {middle} with k = {right}: "
            f"{reduction:.3f} metres "
            f"per additional cluster"
        )

        # Little change means that the appropriate value
        # lies in the lower half of the interval.
        if reduction < small_change:
            right = middle
        else:
            left = middle

    recommended_k = right

    print(
        "\nTrial and error recommended k:",
        recommended_k
    )

    return recommended_k, models, radii


def find_principal_cluster(labels, weights, k):
    """
    Define the principal cluster as the cluster containing
    the greatest total number of deaths.
    """

    cluster_deaths = np.zeros(k)

    for cluster in range(k):
        cluster_deaths[cluster] = np.sum(
            weights[labels == cluster]
        )

    principal_cluster = int(
        np.argmax(cluster_deaths)
    )

    return principal_cluster, cluster_deaths


def find_nearest_pump(point, pump_X):
    """
    Calculate the distance from a point to every pump
    and return the index of the nearest pump.
    """

    distances = np.linalg.norm(
        pump_X - point,
        axis=1
    )

    nearest_pump = int(
        np.argmin(distances)
    )

    return nearest_pump, distances


def plot_k_selection(
    radii,
    recommended_k,
    final_k
):
    """
    Plot the weighted average radius for all evaluated
    values of k.

    A red line shows the trial and error recommendation.
    A green line shows a manually supplied value.
    """

    k_values = sorted(radii.keys())

    radius_values = [
        radii[current_k]
        for current_k in k_values
    ]

    plt.figure(figsize=(8, 6))

    plt.plot(
        k_values,
        radius_values,
        marker="o",
        label="Weighted average radius"
    )

    if recommended_k is not None:
        plt.axvline(
            recommended_k,
            color="red",
            linestyle="--",
            linewidth=2,
            label=(
                f"Trial and error recommendation: "
                f"k = {recommended_k}"
            )
        )

    if (
        final_k is not None
        and final_k != recommended_k
    ):
        plt.axvline(
            final_k,
            color="green",
            linestyle=":",
            linewidth=2,
            label=(
                f"Manually selected: "
                f"k = {final_k}"
            )
        )

    plt.title(
        "Evaluation of Candidate Values of k"
    )

    plt.xlabel(
        "Number of clusters (k)"
    )

    plt.ylabel(
        "Weighted average radius (metres)"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


def run_kmeans(k=None, use_death_weights=True, n_init=50, data=None):
    """
    Run the complete weighted K-means analysis.

    If k is supplied manually, that value is used for
    the final clustering. Otherwise, the value recommended
    by the trial and error procedure is used.
    """

    deaths, pumps = load_data() if data is None else data

    X, pump_X, weights = prepare_analysis_data(deaths, pumps, use_death_weights)

    # This is the value suggested by the trial and error procedure.
    recommended_k, models, radii = select_k(
        X,
        weights,
        max_k=32,
        small_change=2.0,
        n_init=n_init
    )

    # A manually supplied k takes precedence.
    if k is not None:
        final_k = k
    else:
        final_k = recommended_k

    print("\nK selection summary")
    print("-" * 50)
    print("Trial and error recommended k:", recommended_k)
    print("Manually requested k:", k)
    print("Final k used for clustering:", final_k)

    plot_k_selection(
        radii,
        recommended_k,
        final_k
    )

    if final_k is None:
        print(
            "\nNo final value of k is available. "
            "Supply a value using --k."
        )

        plt.show()
        return

    if final_k > len(X):
        raise ValueError(
            "k cannot exceed the number of houses "
            "with positive death counts."
        )

    # Reuse a previously fitted model when possible.
    if final_k in models:
        model = models[final_k]
    else:
        model = fit_kmeans(
            X,
            weights,
            final_k,
            n_init=n_init
        )

    labels = model.labels_
    centroids = model.cluster_centers_

    principal_cluster, cluster_deaths = (
        find_principal_cluster(
            labels,
            weights,
            final_k
        )
    )

    principal_centroid = centroids[
        principal_cluster
    ]

    nearest_pump, pump_distances = (
        find_nearest_pump(
            principal_centroid,
            pump_X
        )
    )

    total_deaths = np.sum(weights)

    principal_deaths = cluster_deaths[
        principal_cluster
    ]

    principal_percentage = (
        principal_deaths
        / total_deaths
        * 100
    )

    print("\nFinal clustering result")
    print("-" * 50)
    print("Selected k:", final_k)

    print(
        "Weight in each cluster:",
        cluster_deaths
    )

    print(
        "Principal cluster:",
        principal_cluster
    )

    print(
        "Weight in principal cluster:",
        principal_deaths
    )

    print(
        "Percentage of total weight:",
        f"{principal_percentage:.2f}%"
    )

    print(
        "Principal centroid:",
        principal_centroid
    )

    print(
        "Nearest pump index:",
        nearest_pump
    )

    print(
        "Distance to nearest pump:",
        f"{pump_distances[nearest_pump]:.3f} metres"
    )

    plot_clusters(
        X=X,
        pump_X=pump_X,
        weights=weights,
        labels=labels,
        centroids=centroids,
        nearest_pump=nearest_pump,
        principal_cluster=principal_cluster,
        k=final_k
    )

    objective_values = demonstrate_initializations(
        X=X,
        weights=weights,
        k=final_k,
        number_of_runs=n_init
    )

    bootstrap_results = (
        evaluate_bootstrap_stability(
            X=X,
            weights=weights,
            pump_X=pump_X,
            reference_model=model,
            k=final_k,
            number_of_samples=100,
            random_state=42,
            n_init=n_init
        )
    )

    # Count how often each pump is nearest across bootstrap samples.
    nearest_pumps = np.asarray(
        bootstrap_results["nearest_pumps"],
        dtype=int
    )

    pump_counts = np.bincount(
        nearest_pumps,
        minlength=len(pump_X)
    )

    reference_pump = bootstrap_results["reference_pump"]

    colors = [
        "crimson" if index == reference_pump else "steelblue"
        for index in range(len(pump_X))
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        np.arange(len(pump_X)),
        pump_counts,
        color=colors
    )

    ax.set_xticks(
        np.arange(len(pump_X)),
        [f"P{index}" for index in range(len(pump_X))]
    )

    for index, count in enumerate(pump_counts):
        ax.text(
            index,
            count + 0.5,
            str(count),
            ha="center"
        )

    ax.set_ylim(0, pump_counts.max() * 1.15 + 1)

    ax.set_title(
        f"K-means bootstrap: nearest pump "
        f"({len(nearest_pumps)} samples)"
    )

    ax.set_xlabel(
        "Pump index (red = full-data reference pump)"
    )

    ax.set_ylabel(
        "Number of bootstrap samples"
    )

    fig.tight_layout()

    plt.figure(figsize=(8, 5))

    plt.hist(
        bootstrap_results[
            "centroid_displacements"
        ],
        bins=15,
        edgecolor="black"
    )

    plt.axvline(
        np.median(
            bootstrap_results[
                "centroid_displacements"
            ]
        ),
        color="red",
        linestyle="--",
        label="Median displacement"
    )

    plt.title(
        f"Bootstrap Stability of Principal Centroid "
        f"(k = {final_k})"
    )

    plt.xlabel(
        "Displacement from original centroid (metres)"
    )

    plt.ylabel(
        "Number of bootstrap samples"
    )

    percentile_90 = np.percentile(
        bootstrap_results["centroid_displacements"],
        90
    )

    plt.axvline(
        x=percentile_90,
        color="darkorange",
        linestyle=":",
        linewidth=1.5,
        label=f"90th percentile: {percentile_90:.3f} m"
    )

    plt.legend()
    plt.tight_layout()

    # Display the K-selection graph and cluster map together.
    plt.show()