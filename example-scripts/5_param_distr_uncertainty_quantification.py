import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import gamma, norm, uniform
from sklearn.model_selection import train_test_split

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import read_labels

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
    # (1) Loading the dataset
    # ================================================================
    loadpath = sys.argv[1].rstrip("/") + "/"
    X = np.loadtxt(loadpath + "X.txt", dtype=float)
    Y = np.loadtxt(loadpath + "Y.txt", dtype=float)
    xlabels = read_labels(loadpath + "xlabels.txt")
    ylabels = read_labels(loadpath + "ylabels.txt")

    # ================================================================
    # (2) Building an example training and test sets
    # ================================================================
    idx_feature = int(sys.argv[2])
    active_feature = ylabels[idx_feature]
    print(f"\n{active_feature} feature selected for emulation.")

    y = np.copy(Y[:, idx_feature])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # ================================================================
    # (4) Training a Gaussian Process Emulator (GPE)
    # ================================================================
    savepath = sys.argv[4].rstrip("/") + f"/{idx_feature}/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    np.savetxt(savepath + "X_train.txt", X_train, fmt="%.6g")
    np.savetxt(savepath + "y_train.txt", y_train, fmt="%.6g")
    np.savetxt(savepath + "X_test.txt", X_test, fmt="%.6g")
    np.savetxt(savepath + "y_test.txt", y_test, fmt="%.6g")

    emul = GPEmul(X_train, y_train)
    emul.train([], [], savepath=savepath)
    emul.save()

    # ================================================================
    # (6) Performing uncertainty quantification for single input parameter
    # ================================================================
    loadpath = savepath
    emul = GPEmul.load(X_train, y_train, loadpath)
    y_pred_mean, y_pred_std = emul.predict(X_test)

    idx_param = int(sys.argv[3])
    active_param = xlabels[idx_param]
    print(
        f"\n{active_param} parameter selected for uncertainty quantification."
    )

    a, b = X_test[:, idx_param].min(), X_test[:, idx_param].max()
    idx = np.argmin(np.abs(X_test[:, idx_param] - 0.5 * (a + b)))
    x0 = X_test[idx, :]
    y0_mean, y0_std = y_pred_mean[idx], y_pred_std[idx]

    n_samples = 10000

    # x = gamma.rvs(x0[idx_param], scale=(b-a)/3, size=n_samples)
    # x = uniform.rvs(loc=a, scale=b-a, size=n_samples)
    x = norm.rvs(loc=x0[idx_param], scale=(b - a) / 3, size=n_samples)

    X_uq = np.tile(x0, (n_samples, 1))
    X_uq[:, idx_param] = x
    y_mean_param, y_std_param = emul.predict(X_uq)

    M = np.hstack(
        (
            x.reshape(-1, 1),
            100
            * np.hstack(
                (
                    y_mean_param.reshape(-1, 1) / y0_mean,
                    y_std_param.reshape(-1, 1) / y0_std,
                )
            ),
        )
    )
    df = pd.DataFrame(
        data=M,
        columns=[
            f"{active_param} distribution",
            "mean (% of control)\ndistribution",
            "std (% of control)\ndistribution",
        ],
    )

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(1, 3, figsize=(2 * width, 2 * height / 4))

    for col, axis in zip(df.columns, axes.flat):
        sns.histplot(df[col], kde=True, ax=axis)
    axes[0].axvline(
        x0[idx_param],
        c="r",
        ls="--",
        label=f"control:\n{active_param}={x0[idx_param]:.4f}",
    )
    axes[0].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
