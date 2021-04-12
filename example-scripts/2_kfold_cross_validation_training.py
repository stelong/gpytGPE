import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from sklearn.model_selection import KFold

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.concurrent import execute_task_in_parallel
from gpytGPE.utils.metrics import IndependentStandardError as ISE

FOLD = 5
SEED = 8
METRICS_DCT = {
    "MSE": torchmetrics.MeanSquaredError(),
    "R2Score": torchmetrics.R2Score(),
}  # you can expand this dictionary with other metrics from torchmetrics you are intrested in monitoring
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
    emul.train([], [], savepath=savepath, watch_metric=metric)
    emul.save()

    y_pred_mean, y_pred_std = emul.predict(X_val)

    score = metric(emul.tensorize(y_pred_mean), emul.tensorize(y_val))
    ise = ISE(
        emul.tensorize(y_val),
        emul.tensorize(y_pred_mean),
        emul.tensorize(y_pred_std),
    )

    return (score, ise), emul.best_epoch


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
    # GPE training with K-fold cross-validation
    # ================================================================
    fold = FOLD
    metric_name = WATCH_METRIC
    metric = METRICS_DCT[metric_name]
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

    # in case "cv" function cannot be run in parallel, comment line 90 and uncomment the following block
    # results = []
    # for key in inputs.keys():
    #     results.append(cv(*inputs[key]))

    metric_list = [results[i][0][0] for i in range(fold)]
    np.savetxt(
        savepath + metric_name + "_cv.txt",
        np.array(metric_list),
        fmt="%.6f",
    )
    ise_list = [results[i][0][1] for i in range(fold)]
    np.savetxt(savepath + "ISE_cv.txt", np.array(ise_list), fmt="%.6f")

    # ================================================================
    # GPE training using the entire dataset
    # ================================================================
    best_epoch_list = [results[i][1] for i in range(fold)]
    n_epochs = int(np.max(best_epoch_list))

    np.savetxt(savepath + "X_train.txt", X, fmt="%g")
    np.savetxt(savepath + "y_train.txt", y, fmt="%g")

    emul = GPEmul(X, y)
    emul.train(
        [],
        [],
        max_epochs=n_epochs,
        savepath=savepath,
        watch_metric=metric,
    )
    emul.save()


if __name__ == "__main__":
    main()
