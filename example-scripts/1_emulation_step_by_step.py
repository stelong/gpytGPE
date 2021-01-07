import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import read_labels
from gpytGPE.utils.metrics import R2Score
from gpytGPE.utils.plotting import plot_dataset

LEARNING_RATE = 0.1
MAX_EPOCHS = 1000
METRIC = "R2Score"
N_RESTARTS = 10
PATIENCE = 20
SEED = 8


def main():
    # ================================================================
    # (0) Making the code reproducible
    # ================================================================
    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # (1) Loading and visualising dataset
    # ================================================================
    path_in = sys.argv[1].rstrip("/") + "/"
    X = np.loadtxt(path_in + "X.txt", dtype=float)
    Y = np.loadtxt(path_in + "Y.txt", dtype=float)

    xlabels = read_labels(path_in + "xlabels.txt")
    ylabels = read_labels(path_in + "ylabels.txt")
    plot_dataset(X, Y, xlabels, ylabels)

    # ================================================================
    # (2) Building example training and validation datasets
    # ================================================================
    idx_feature = sys.argv[2]
    print(f"\n{ylabels[int(idx_feature)]} feature selected for emulation.")

    y = np.copy(Y[:, int(idx_feature)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # ================================================================
    # (3) Training GPE
    # ================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = LEARNING_RATE
    max_epochs = MAX_EPOCHS
    metric = METRIC
    n_restarts = N_RESTARTS
    patience = PATIENCE

    path_out = sys.argv[3].rstrip("/") + "/" + idx_feature + "/"
    Path(path_out).mkdir(parents=True, exist_ok=True)

    np.savetxt(path_out + "X_train.txt", X_train, fmt="%.6f")
    np.savetxt(path_out + "y_train.txt", y_train, fmt="%.6f")
    np.savetxt(path_out + "X_val.txt", X_val, fmt="%.6f")
    np.savetxt(path_out + "y_val.txt", y_val, fmt="%.6f")

    emul = GPEmul(X_train, y_train, device, learn_noise=False, scale_data=True)
    emul.train(
        X_val,
        y_val,
        n_restarts,
        lr,
        max_epochs,
        patience,
        savepath=path_out,
        save_losses=False,
        watch_metric=metric,
    )

    # ================================================================
    # (4) Saving trained GPE
    # ================================================================
    filename = "gpe.pth"
    emul.save(filename=filename)

    # ================================================================
    # (5) Loading already trained GPE
    # ================================================================
    # NOTE: you need exactely the same training dataset used in (3)
    # ================================================================
    emul = GPEmul.load(path_out, X_train, y_train, filename=filename)

    # ================================================================
    # (6) Testing trained GPE at new input points (inference)
    # ================================================================
    # NOTE: we will use the validation dataset used in (3) as an example
    # ================================================================
    X_test = X_val
    y_test = y_val

    y_pred_mean, y_pred_std = emul.predict(X_test)
    r2s = R2Score(emul.tensorize(y_test), emul.tensorize(y_pred_mean))
    print(f"\nAccuracy on testing dataset: R2Score = {r2s:.6f}.")

    # ================================================================
    # (7) Plotting predictions vs observations
    # ================================================================
    height = 9.36111
    width = 5.91667
    fig, axis = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 3))

    l = np.argsort(
        y_pred_mean
    )  # let's sort predicted values for a better visualisation
    ci = 3

    axis.scatter(
        np.arange(len(l)),
        y_test[l],
        facecolors="none",
        edgecolors="C0",
        label="observed",
    )
    axis.scatter(
        np.arange(len(l)),
        y_pred_mean[l],
        facecolors="C0",
        s=16,
        label="predicted",
    )
    axis.errorbar(
        np.arange(len(l)),
        y_pred_mean[l],
        yerr=ci * y_pred_std[l],
        c="C0",
        ls="none",
        lw=0.5,
        label=f"uncertainty ({ci} STD)",
    )
    axis.set_title(f"R2Score = {r2s:.6f}", fontsize=12)
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_ylabel(ylabels[int(idx_feature)], fontsize=12)
    axis.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
