import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from data_utils import load_data, extract_coordinates


def fit_kde(X, weights, bandwidth=50):
    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=bandwidth
    )

    kde.fit(X, sample_weight=weights)

    return kde


def create_grid(X, pump_X, resolution=100, padding=30):
    x_min = min(X[:, 0].min(), pump_X[:, 0].min()) - padding
    x_max = max(X[:, 0].max(), pump_X[:, 0].max()) + padding

    y_min = min(X[:, 1].min(), pump_X[:, 1].min()) - padding
    y_max = max(X[:, 1].max(), pump_X[:, 1].max()) + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.column_stack([
        xx.ravel(),
        yy.ravel()
    ])

    return xx, yy, grid_points


def evaluate_kde(kde, grid_points, shape):
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)

    return density.reshape(shape)


def plot_kde(xx, yy, density, X, pump_X):
    plt.contourf(xx, yy, density)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        s=8
    )

    plt.scatter(
        pump_X[:, 0],
        pump_X[:, 1],
        marker="x",
        s=80
    )

    plt.title("KDE of Cholera Death Locations")
    plt.xlabel("COORD_X")
    plt.ylabel("COORD_Y")

    plt.show()


def run_kde():
    deaths, pumps = load_data()

    X, pump_X, weights = extract_coordinates(
        deaths,
        pumps
    )

    kde = fit_kde(
        X,
        weights,
        bandwidth=50
    )

    xx, yy, grid_points = create_grid(
        X,
        pump_X
    )

    density = evaluate_kde(
        kde,
        grid_points,
        xx.shape
    )

    plot_kde(
        xx,
        yy,
        density,
        X,
        pump_X
    )