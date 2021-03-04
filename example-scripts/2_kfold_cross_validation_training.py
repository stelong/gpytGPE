import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.concurrent import execute_task_in_parallel

FOLD = 5
SEED = 8
WATCH_METRIC = "R2Score"


def cv(X_train, y_train, X_val, y_val, split, savepath, metric):
    print(f"\nSplit {split}...")

    savepath += f"{split}" + "/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    np.savetxt(savepath + "X_train.txt", X_train, fmt="%g")
    np.savetxt(savepath + "y_train.txt", y_train, fmt="%g")
    np.savetxt(savepath + "X_val.txt", X_val, fmt="%g")
    np.savetxt(savepath + "y_val.txt", y_val, fmt="%g")

    emul = GPEmul(X_train, y_train)
    emul.train(X_val, y_val, savepath=savepath, watch_metric=metric)
    emul.save()

    return emul.metric_score, emul.best_epoch


def main():
    # ================================================================
    # Making the code reproducible
    # ================================================================
    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # Loading the dataset
    # ================================================================
    loadpath = sys.argv[1].rstrip("/") + "/"
    idx_feature = sys.argv[2]

    X = np.loadtxt(loadpath + "X.txt", dtype=float)
    y = np.loadtxt(loadpath + "Y.txt", dtype=float)[:, int(idx_feature)]

    # ================================================================
    # GPE training with cross-validation
    # ================================================================
    fold = FOLD
    metric = WATCH_METRIC
    savepath = sys.argv[3].rstrip("/") + "/" + idx_feature + "/"

    kf = KFold(n_splits=fold, shuffle=False, random_state=None)

    inputs = {
        split: (
            X[idx_train],
            y[idx_train],
            X[idx_val],
            y[idx_val],
            split,
            savepath,
            metric,
        )
        for split, (idx_train, idx_val) in enumerate(kf.split(X))
    }
    results = execute_task_in_parallel(cv, inputs)

    metric_score_list = [results[i][0] for i in range(fold)]
    np.savetxt(
        savepath + metric + "_cv.txt", np.array(metric_score_list), fmt="%g"
    )

    # ================================================================
    # GPE training using the entire dataset
    # ================================================================
    best_epoch_list = [results[i][1] for i in range(fold)]
    n_epochs = int(np.around(np.mean(best_epoch_list), decimals=0))

    np.savetxt(savepath + "X_train.txt", X, fmt="%g")
    np.savetxt(savepath + "y_train.txt", y, fmt="%g")

    emul = GPEmul(X, y)
    emul.train(
        [],
        [],
        max_epochs=n_epochs,
        patience=n_epochs,
        savepath=savepath,
        straight_to_the_end=True,
        watch_metric=metric,
    )
    emul.save()


if __name__ == "__main__":
    main()
