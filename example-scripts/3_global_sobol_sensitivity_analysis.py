import multiprocessing
import random
import sys
from itertools import combinations

import numpy as np
import torch
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.special import binom

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import get_minmax, read_labels
from gpytGPE.utils.plotting import gsa_box, gsa_donut

EMUL_TYPE = "full"  # possible choices are: "full", "best"
WATCH_METRIC = "R2Score"
N = 1000
N_DRAWS = 1000
SEED = 8
THRE = 0.01


def main():
    # ================================================================
    # Making the code reproducible
    # ================================================================
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # GPE loading
    # ================================================================
    emul_type = EMUL_TYPE
    idx_feature = sys.argv[2]
    metric = WATCH_METRIC
    path = sys.argv[1].rstrip("/") + "/" + idx_feature + "/"

    if emul_type == "best":
        metric_score_list = np.loadtxt(path + metric + "_cv.txt", dtype=float)
        if metric == "R2Score":
            best_split = np.argmax(metric_score_list)
        else:
            best_split = np.argmin(metric_score_list)
        path = path + f"{best_split}/"

    X_train = np.loadtxt(path + "X_train.txt", dtype=float)
    y_train = np.loadtxt(path + "y_train.txt", dtype=float)
    filename = "gpe.pth"
    emul = GPEmul.load(path + filename, X_train, y_train)

    # ================================================================
    # Estimating Sobol' sensitivity indices
    # ================================================================
    label = read_labels("data/ylabels.txt")

    n = N
    n_draws = N_DRAWS

    D = X_train.shape[1]
    I = get_minmax(X_train)

    index_i = read_labels("data/xlabels.txt")
    index_ij = [
        "({}, {})".format(c[0], c[1]) for c in combinations(index_i, 2)
    ]

    problem = {"num_vars": D, "names": index_i, "bounds": I}

    X_sobol = saltelli.sample(
        problem, n, calc_second_order=True
    )  # n x (2D + 2) | if calc_second_order == False --> n x (D + 2)
    Y = emul.sample(X_sobol, n_draws=n_draws)

    ST = np.zeros((0, D), dtype=float)
    S1 = np.zeros((0, D), dtype=float)
    S2 = np.zeros((0, int(binom(D, 2))), dtype=float)

    for i in range(n_draws):
        S = sobol.analyze(
            problem,
            Y[i],
            calc_second_order=True,
            parallel=True,
            n_processors=multiprocessing.cpu_count(),
            seed=seed,
        )
        total_order, first_order, (_, second_order) = sobol.Si_to_pandas_dict(
            S
        )

        ST = np.vstack((ST, total_order["ST"].reshape(1, -1)))
        S1 = np.vstack((S1, first_order["S1"].reshape(1, -1)))
        S2 = np.vstack((S2, np.array(second_order["S2"]).reshape(1, -1)))

    np.savetxt(path + "STi.txt", ST, fmt="%.6f")
    np.savetxt(path + "Si.txt", S1, fmt="%.6f")
    np.savetxt(path + "Sij.txt", S2, fmt="%.6f")

    # ================================================================
    # Plotting
    # ================================================================
    thre = THRE
    ylab = label[int(idx_feature)]

    gsa_donut(path, thre, index_i, ylab, savefig=True)
    gsa_box(path, thre, index_i, index_ij, ylab, savefig=True)


if __name__ == "__main__":
    main()
