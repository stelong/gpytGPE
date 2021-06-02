import random
import sys
from itertools import combinations

import numpy as np
import torch
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.special import binom
from scipy.stats import norm

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import get_minmax, read_labels
from gpytGPE.utils.plotting import gsa_box, gsa_donut, gsa_heat, gsa_network

EMUL_TYPE = "full"  # possible choices are: "full", "best"
N = 1000
N_BOOTSTRAP = 100
N_DRAWS = 1000
SEED = 8
THRE = 0.01
WATCH_METRIC = "R2Score"
HIGHEST_IS_BEST = True


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
    metric_name = WATCH_METRIC
    metric_type = HIGHEST_IS_BEST
    loadpath = sys.argv[1].rstrip("/") + "/"
    savepath = sys.argv[3].rstrip("/") + "/" + idx_feature + "/"

    if emul_type == "best":
        metric_score_list = np.loadtxt(
            savepath + metric_name + "_cv.txt", dtype=float
        )
        if metric_type:
            best_split = np.argmax(metric_score_list)
        else:
            best_split = np.argmin(metric_score_list)
        savepath = savepath + f"{best_split}/"

    X_train = np.loadtxt(savepath + "X_train.txt", dtype=float)
    y_train = np.loadtxt(savepath + "y_train.txt", dtype=float)

    emul = GPEmul.load(X_train, y_train, savepath)

    # ================================================================
    # Estimating Sobol' sensitivity indices
    # ================================================================
    n = N
    n_draws = N_DRAWS

    d = X_train.shape[1]
    I = get_minmax(X_train)

    index_i = read_labels(loadpath + "xlabels.txt")
    index_ij = [list(c) for c in combinations(index_i, 2)]

    problem = {"num_vars": d, "names": index_i, "bounds": I}

    X = saltelli.sample(problem, n, calc_second_order=True)
    Y = emul.sample(X, n_draws=n_draws)

    conf_level = 0.95
    z = norm.ppf(0.5 + conf_level / 2)
    n_bootstrap = N_BOOTSTRAP

    ST = np.zeros((0, d), dtype=float)
    S1 = np.zeros((0, d), dtype=float)
    S2 = np.zeros((0, int(binom(d, 2))), dtype=float)

    ST_std = np.zeros((0, d), dtype=float)
    S1_std = np.zeros((0, d), dtype=float)
    S2_std = np.zeros((0, int(binom(d, 2))), dtype=float)

    for i in range(n_draws):
        S = sobol.analyze(
            problem,
            Y[i],
            calc_second_order=True,
            num_resamples=n_bootstrap,
            conf_level=conf_level,
            print_to_console=False,
            parallel=False,
            n_processors=None,
            seed=seed,
        )
        T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

        ST = np.vstack((ST, T_Si["ST"].reshape(1, -1)))
        S1 = np.vstack((S1, first_Si["S1"].reshape(1, -1)))
        S2 = np.vstack((S2, np.array(second_Si["S2"]).reshape(1, -1)))

        ST_std = np.vstack((ST_std, T_Si["ST_conf"].reshape(1, -1) / z))
        S1_std = np.vstack((S1_std, first_Si["S1_conf"].reshape(1, -1) / z))
        S2_std = np.vstack(
            (S2_std, np.array(second_Si["S2_conf"]).reshape(1, -1) / z)
        )

    np.savetxt(savepath + "STi.txt", ST, fmt="%.6f")
    np.savetxt(savepath + "Si.txt", S1, fmt="%.6f")
    np.savetxt(savepath + "Sij.txt", S2, fmt="%.6f")

    np.savetxt(savepath + "STi_std.txt", ST_std, fmt="%.6f")
    np.savetxt(savepath + "Si_std.txt", S1_std, fmt="%.6f")
    np.savetxt(savepath + "Sij_std.txt", S2_std, fmt="%.6f")

    # ================================================================
    # Plotting
    # ================================================================
    thre = THRE
    ylabels = read_labels(loadpath + "ylabels.txt")
    ylabel = ylabels[int(idx_feature)]

    gsa_box(ST, S1, S2, index_i, index_ij, ylabel, savepath, correction=thre)
    gsa_donut(ST, S1, index_i, ylabel, savepath, correction=thre)
    gsa_heat(ST, S1, index_i, ylabel, savepath, correction=thre)
    gsa_network(ST, S1, S2, index_i, index_ij, ylabel, savepath, correction=thre)


if __name__ == "__main__":
    main()
