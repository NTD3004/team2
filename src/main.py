import argparse

from kmeans import run_kmeans
from gmm import run_gmm


def positive_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "The value must be a positive integer."
        )

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "The value must be a positive integer."
        )

    return value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "method",
        choices=["kmeans", "gmm"],
        help="Method to run"
    )

    parser.add_argument(
        "--k",
        type=positive_integer,
        default=None,
        help=(
            "Number of clusters/components. "
            "If omitted: automatic selection for K-means; "
            "8 components for GMM."
        )
    )

    parser.add_argument(
        "--max-iter",
        type=positive_integer,
        default=300,
        help="Maximum number of EM iterations for GMM"
    )

    args = parser.parse_args()

    if args.method == "kmeans":
        run_kmeans(k=args.k)

    elif args.method == "gmm":
        selected_k = 8 if args.k is None else args.k

        run_gmm(
            k=selected_k,
            max_iter=args.max_iter
        )


if __name__ == "__main__":
    main()