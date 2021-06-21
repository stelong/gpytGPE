import sys

import numpy as np

from gpytGPE.utils.design import read_labels
from gpytGPE.utils.plotting import correct_index

CRITERION = "STi"  # possible choices are: "Si", "STi"
EMUL_TYPE = "full"  # possible choices are: "full", "best"
THRE = 0.01
WATCH_METRIC = "R2Score"
HIGHEST_IS_BEST = True


def main():
    loadpath = sys.argv[1].rstrip("/") + "/"

    index_i = read_labels(loadpath + "xlabels.txt")
    label = read_labels(loadpath + "ylabels.txt")
    features = np.loadtxt(loadpath + "features_idx_list.txt", dtype=int)

    criterion = CRITERION

    if criterion == "Si":
        tag = "first-order"
    elif criterion == "STi":
        tag = "total"

    if features.shape == ():
        features_list = [features]
    else:
        features_list = list(features)

    msg = f"\nParameters ranking will be performed according to {tag} effects on selected features:\n"
    for idx in features_list:
        msg += f" {label[idx]}"
    print(msg)

    emul_type = EMUL_TYPE
    metric_name = WATCH_METRIC
    metric_type = HIGHEST_IS_BEST
    thre = THRE

    loadpath_sobol = sys.argv[2].rstrip("/") + "/"
    r_dct = {key: [] for key in index_i}

    for idx in features_list:
        path = loadpath_sobol + f"{idx}/"

        if emul_type == "best":
            metric_score_list = np.loadtxt(
                path + metric_name + "_cv.txt", dtype=float
            )
            if metric_type:
                best_split = np.argmax(metric_score_list)
            else:
                best_split = np.argmin(metric_score_list)
            path += f"{best_split}/"

        S = np.loadtxt(path + criterion + ".txt", dtype=float)
        S = correct_index(S, thre)

        mean = np.array([S[:, i].mean() for i in range(len(index_i))])
        ls = list(np.argsort(mean))

        for i, idx in enumerate(ls):
            if mean[idx] != 0:
                r_dct[index_i[idx]].append(i)
            else:
                r_dct[index_i[idx]].append(0)

    for key in r_dct.keys():
        r_dct[key] = -np.sum(r_dct[key])

    tuples = sorted(r_dct.items(), key=lambda tup: tup[1])

    R = {}
    c = 0
    for i, t in enumerate(tuples):
        if t[1] == 0:
            if c == 0:
                i0 = i + 1
                if i < len(index_i) - 1:
                    c += 1
            else:
                c += 1
            rank = f"#{i0}_{c}"
        else:
            rank = f"#{i+1}"

        R[rank] = t[0]

    print(
        f"\nParameters ranking from the most to the least important is:\n {R}"
    )


if __name__ == "__main__":
    main()
