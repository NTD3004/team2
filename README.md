## First run this
docker compose up --build

## Then run this
docker compose up

## Choose console in the notebook browser
## Then run one of these commands

# Gaussian KDE
%run src/main.py kde --kernel gaussian

# Box KDE
%run src/main.py kde --kernel box

