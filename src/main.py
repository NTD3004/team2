import argparse
import math


def positive_integer(value):
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("The value must be a positive integer.")


def positive_float(value):
    try:
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("The value must be finite and positive.")


def main():
    parser = argparse.ArgumentParser(description="Analyze cholera death locations.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--use-death-weights", action=argparse.BooleanOptionalAction, default=True,
        help="Use death counts (default); --no-use-death-weights gives each affected house unit weight.",
    )
    methods = parser.add_subparsers(dest="method", required=True)
    kmeans = methods.add_parser("kmeans", parents=[common])
    kmeans.add_argument("--k", type=positive_integer, default=None,
                        help="Number of clusters; omitted means automatic selection.")
    kmeans.add_argument("--n-init", type=positive_integer, default=50,
                        help="K-means initializations per fit (also bootstrap); default: 50.")
    gmm = methods.add_parser("gmm", parents=[common])
    gmm.add_argument("--k", type=positive_integer, default=8)
    gmm.add_argument("--n-init", type=positive_integer, default=50,
                     help="Initializations for the K-means seed of each GMM fit.")
    gmm.add_argument("--reg-covar", type=positive_float, default=1e-6,
                     help="Covariance diagonal regularization in squared coordinate units.")
    gmm.add_argument("--max-iter", type=positive_integer, default=300)
    gmm.add_argument("--tol", type=positive_float, default=1e-6)
    gmm.add_argument("--bootstrap", action="store_true", help="Run bootstrap stability analysis.")
    gmm.add_argument("--samples", type=positive_integer, default=100)
    gmm.add_argument("--seed", type=int, default=42)
    kde = methods.add_parser("kde", parents=[common])
    kde.add_argument("--bandwidth", type=positive_float, default=50,
                     help="Kernel bandwidth in coordinate units (metres); default: 50.")
    kde.add_argument("--kernel", choices=["gaussian", "box"], default="gaussian")
    args = parser.parse_args()
    if args.method == "kmeans":
        from kmeans import run_kmeans
        run_kmeans(k=args.k, use_death_weights=args.use_death_weights, n_init=args.n_init)
    elif args.method == "gmm":
        from gmm import run_gmm, run_gmm_bootstrap
        options = dict(k=args.k, use_death_weights=args.use_death_weights,
                       n_init=args.n_init, reg_covar=args.reg_covar,
                       max_iter=args.max_iter, tol=args.tol)
        if args.bootstrap:
            run_gmm_bootstrap(**options, number_of_samples=args.samples, random_state=args.seed)
        else:
            run_gmm(**options)
    else:
        from kde import run_kde
        run_kde(kernel_type=args.kernel, use_death_weights=args.use_death_weights,
                bandwidth=args.bandwidth)


if __name__ == "__main__":
    main()
