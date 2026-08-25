import argparse

from kde import run_kde


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "method",
        choices=["kde", "base", "kmeans"],
        help="Method to run"
    )

    parser.add_argument(
        "--kernel",
        choices=["gaussian", "box"],
        default="gaussian",
        help="Kernel used by KDE (default: gaussian)"
    )

    args = parser.parse_args()

    if args.method == "kde":
        run_kde(args.kernel)

    # elif args.method == "base":
    #     run_baseline()

    # elif args.method == "kmeans":
    #     run_kmeans()


if __name__ == "__main__":
    main()