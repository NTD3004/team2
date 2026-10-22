## First run this
docker compose up --build

## Then run this
docker compose up

## Choose console in the notebook browser
## Then run one of these commands

# Kmeans
%run src/main.py kmeans

# Kmeans with k
%run src/main.py kmeans --k 8

# GMM
%run src/gmm.py --k 8

# KDE (Gaussian kernel)
%run src/kde.py

# KDE (box kernel)
%run src/kde.py --kernel box

# KDE with custom candidate bandwidths (metres) and bootstrap count
%run src/kde.py --bandwidths 10 20 30 50 75 100 150 250 500 --samples 100

# KDE bandwidth selection without bootstrap
%run src/kde.py --samples 0

KDE selects h by maximizing weighted leave-one-house-out log-likelihood.
The bandwidth-selection graph plots this score against h and highlights the
maximum with a red marker and dashed line. Higher is better; this uses the
likelihood maximum rather than a K-means-style elbow rule.
Each held-out house and all its death weight are removed from training;
the remaining density is normalized by the remaining death total.
Defaults: Gaussian kernel, 25 log-spaced candidate bandwidths from 5 to 500
metres, 100 house bootstrap samples, seed 42, and a 100-by-100 density grid.
Extend the candidate range if the selected h is at its boundary. For box
kernels, any held-out positive-weight house with zero density makes that
candidate's score negative infinity; use larger candidates if all fail.
The hotspot is the maximum on the density grid, with the first grid point
chosen for ties. Use `--resolution 200` (or higher) to check grid sensitivity.
Bootstrap resamples houses with their death weights, reselects h on each
sample, and reports hotspot displacement and nearest-pump frequencies.
Any failed samples are reported and excluded from summary denominators.

# GMM bootstrap
%run src/gmm_bootstrap.py --k 8 --samples 100

# GMM bootstrap with # of iterations (If some fits do not converge)
%run src/gmm_bootstrap.py --k 8 --samples 100 --max-iter 1000


