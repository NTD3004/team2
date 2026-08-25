import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from data_utils import load_data, extract_coordinates

def estimate_density(point, X, weights, bandwidth, kernel):
    D = X.shape[1]

    # (x - x_n) / h for every x_n
    u = (point - X) / bandwidth

    # Evaluate k(u_n) for all observations
    kernel_values = kernel(u)

    # Weighted sum
    total = np.sum(weights * kernel_values)

    return total / (
        np.sum(weights) * bandwidth ** D
    )

def gaussian_kernel(u):
    D = u.shape[1]

    squared_distance = np.sum(u ** 2, axis=1)

    return (
        (2 * np.pi) ** (-D / 2)
        * np.exp(-0.5 * squared_distance)
    )

def box_kernel(u):
    return np.all(np.abs(u) <= 0.5, axis=1).astype(float)

def create_grid(X, pump_X, resolution=100, padding=30):
    x_min = min(X[:, 0].min(), pump_X[:, 0].min()) - padding
    x_max = max(X[:, 0].max(), pump_X[:, 0].max()) + padding

    y_min = min(X[:, 1].min(), pump_X[:, 1].min()) - padding
    y_max = max(X[:, 1].max(), pump_X[:, 1].max()) + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    return xx, yy

def run_kde(kernel_type="gaussian"):
    print("Running KDE with kernel:", kernel_type)
    deaths, pumps = load_data()

    X, pump_X, weights = extract_coordinates(
        deaths,
        pumps
    )

    xx, yy = create_grid(
        X,
        pump_X
    )

    density = np.zeros(xx.shape)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([
                xx[i, j],
                yy[i, j]
            ])

            if (kernel_type == "gaussian"):
                density[i, j] = estimate_density(
                    point,
                    X,
                    weights,
                    bandwidth=50,
                    kernel=gaussian_kernel
                )
            else:
                density[i, j] = estimate_density(
                    point,
                    X,
                    weights,
                    bandwidth=50,
                    kernel=box_kernel
                )

    
    plt.contourf(xx, yy, density)

    plt.scatter(X[:, 0], X[:, 1], s=8)
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


    