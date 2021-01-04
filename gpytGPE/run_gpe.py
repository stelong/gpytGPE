#!/usr/bin/env python3
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.concurrent import execute_task_in_parallel

FOLD = 3
LEARNING_RATE = 0.1
MAX_EPOCHS = 1000
METRIC = "R2Score"
N_RESTARTS = 1
PATIENCE = 20
SEED = 8


def cv(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    n_restarts,
    lr,
    max_epochs,
    metric,
    patience,
    split,
    path,
):
    print(f"\nSplit {split}...")

    savepath = path + f"{split}" + "/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    np.savetxt(savepath + "X_train.txt", X_train, fmt="%.6f")
    np.savetxt(savepath + "y_train.txt", y_train, fmt="%.6f")
    np.savetxt(savepath + "X_val.txt", X_val, fmt="%.6f")
    np.savetxt(savepath + "y_val.txt", y_val, fmt="%.6f")

    emul = GPEmul(X_train, y_train, device, learn_noise=False, scale_data=True)
    emul.train(
        X_val,
        y_val,
        n_restarts,
        lr,
        max_epochs,
        patience,
        savepath=savepath,
        save_losses=False,
        watch_metric=metric,
    )
    emul.print_stats()
    emul.save()

    return emul.metric_score, emul.best_epoch


def main():
    # ========================================
    # Making the code reproducible
    # ========================================
    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ========================================
    # Loading the dataset
    # ========================================
    tag = sys.argv[1]
    idx_feature = sys.argv[2]

    path_in = "../data/" + tag + "/"
    X = np.loadtxt(path_in + "X.txt", dtype=float)
    y = np.loadtxt(path_in + "Y.txt", dtype=float)[:, int(idx_feature)]

    # ========================================
    # GPE training with cross-validation
    # ========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = FOLD
    lr = LEARNING_RATE
    max_epochs = MAX_EPOCHS
    metric = METRIC
    n_restarts = N_RESTARTS
    patience = PATIENCE

    path_out = "../tmp/" + tag + "/" + idx_feature + "/"

    kf = KFold(n_splits=fold, shuffle=True, random_state=seed)

    inputs = {
        split: (
            X[idx_train],
            y[idx_train],
            X[idx_val],
            y[idx_val],
            device,
            n_restarts,
            lr,
            max_epochs,
            metric,
            patience,
            split,
            path_out,
        )
        for split, (idx_train, idx_val) in enumerate(kf.split(X))
    }
    results = execute_task_in_parallel(cv, inputs)

    metric_score_list = [results[i][0] for i in range(fold)]
    np.savetxt(
        path_out + metric + "_cv.txt", np.array(metric_score_list), fmt="%.6f"
    )

    # ========================================
    # GPE training using the entire dataset
    # ========================================
    best_epoch_list = [results[i][1] for i in range(fold)]
    n_epochs = int(np.round(np.mean(best_epoch_list), 0))

    np.savetxt(path_out + "X_train.txt", X, fmt="%.6f")
    np.savetxt(path_out + "y_train.txt", y, fmt="%.6f")

    emul = GPEmul(X, y, device, learn_noise=False, scale_data=True)
    emul.train(
        [],
        [],
        n_restarts,
        lr,
        max_epochs=n_epochs,
        patience=n_epochs,
        savepath=path_out,
        save_losses=False,
        watch_metric=metric,
    )
    emul.print_stats()
    emul.save()


if __name__ == "__main__":
    main()
