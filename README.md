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

# GMM bootstrap
%run src/gmm_bootstrap.py --k 8 --samples 100

# GMM bootstrap with # of iterations (If some fits do not converge)
%run src/gmm_bootstrap.py --k 8 --samples 100 --max-iter 1000


