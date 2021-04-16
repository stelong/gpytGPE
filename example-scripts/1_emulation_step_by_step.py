import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from sklearn.model_selection import train_test_split

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import read_labels
from gpytGPE.utils.metrics import IndependentStandardError as ISE
from gpytGPE.utils.plotting import plot_dataset

SEED = 8
METRICS_DCT = {
    "MSE": torchmetrics.MeanSquaredError(),
    "R2Score": torchmetrics.R2Score(),
}  # you can expand this dictionary with other metrics from torchmetrics you are intrested in monitoring
WATCH_METRIC = "R2Score"


def main():
    # ================================================================
    # (0) Making the code reproducible
    # ================================================================
    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # (1) Loading and visualising the dataset
    # ================================================================
    loadpath = sys.argv[1].rstrip("/") + "/"
    X = np.loadtxt(loadpath + "X.txt", dtype=float)
    Y = np.loadtxt(loadpath + "Y.txt", dtype=float)

    savepath = sys.argv[3].rstrip("/") + "/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    xlabels = read_labels(loadpath + "xlabels.txt")
    ylabels = read_labels(loadpath + "ylabels.txt")
    plot_dataset(X, Y, xlabels, ylabels, savepath=savepath)

    # ================================================================
    # (2) Building an example training, validation and test sets
    # ================================================================
    idx_feature = sys.argv[2]
    print(f"\n{ylabels[int(idx_feature)]} feature selected for emulation.")

    y = np.copy(Y[:, int(idx_feature)])

    X_, X_test, y_, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_, y_, test_size=0.2, random_state=seed
    )

    savepath += idx_feature + "/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    np.savetxt(savepath + "X_train.txt", X_train, fmt="%.6g")
    np.savetxt(savepath + "y_train.txt", y_train, fmt="%.6g")
    np.savetxt(savepath + "X_val.txt", X_val, fmt="%.6g")
    np.savetxt(savepath + "y_val.txt", y_val, fmt="%.6g")
    np.savetxt(savepath + "X_test.txt", X_test, fmt="%.6g")
    np.savetxt(savepath + "y_test.txt", y_test, fmt="%.6g")

    # ================================================================
    # (3) Training a Gaussian Process Emulator (GPE)
    # ================================================================
    metric_name = WATCH_METRIC
    metric = METRICS_DCT[
        metric_name
    ]  # initialise the chosen regression metric from torchmetrics

    # There are two possible ways of training a GPE:
    #
    # (A) either against a validation set (e.g. by validation loss)
    emul_a = GPEmul(X_train, y_train)
    emul_a.train(
        X_val, y_val, savepath=savepath, watch_metric=metric, save_losses=True
    )
    #
    # (B) or against nothing (e.g. by training loss)
    emul_b = GPEmul(X_train, y_train)
    emul_b.train(
        [], [], savepath=savepath, watch_metric=metric, save_losses=False
    )

    # ================================================================
    # (4) Saving a trained GPE
    # ================================================================
    emul_a.save(filename="gpe_a.pth")
    emul_b.save(filename="gpe_b.pth")

    # ================================================================
    # (5) Loading an already trained GPE
    # ================================================================
    # NOTE: you need exactly the same training dataset used in (3)
    # ================================================================
    loadpath = savepath
    emul_a = GPEmul.load(X_train, y_train, loadpath, filename="gpe_a.pth")
    emul_b = GPEmul.load(X_train, y_train, loadpath, filename="gpe_b.pth")
    emlators = [emul_a, emul_b]
    tags = ["with_val=True", "with_val=False"]

    # ================================================================
    # (6) Testing the trained GPE on the held out data
    # ================================================================
    mean_list = []
    std_list = []
    score_list = []
    ise_list = []

    for i, emul in enumerate(emlators):
        y_pred_mean, y_pred_std = emul.predict(X_test)
        mean_list.append(y_pred_mean)
        std_list.append(y_pred_std)

        score = metric(emul.tensorize(y_pred_mean), emul.tensorize(y_test))
        score_list.append(score)
        ise = ISE(
            emul.tensorize(y_test),
            emul.tensorize(y_pred_mean),
            emul.tensorize(y_pred_std),
        )
        ise_list.append(ise)
        print(f"\nStatistics on test set for GPE trained {tags[i]}:")
        print(f"  {metric_name} = {score:.4f}")
        print(f"  ISE = {ise:.2f} %\n")

    # ================================================================
    # (7) Plotting mean predictions + uncertainty vs observations
    # ================================================================
    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 4))

    ci = 2  # ~95% confidance interval

    inf_bound = []
    sup_bound = []
    for i, (m, s) in enumerate(zip(mean_list, std_list)):
        l = np.argsort(m)  # for the sake of a better visualisation
        inf_bound.append((m - ci * s).min())  # same
        sup_bound.append((m + ci * s).max())  # same

        axes[i].scatter(
            np.arange(1, len(l) + 1),
            y_test[l],
            facecolors="none",
            edgecolors="C0",
            label="observed",
        )
        axes[i].scatter(
            np.arange(1, len(l) + 1),
            m[l],
            facecolors="C0",
            s=16,
            label="predicted",
        )
        axes[i].errorbar(
            np.arange(1, len(l) + 1),
            m[l],
            yerr=ci * s[l],
            c="C0",
            ls="none",
            lw=0.5,
            label=f"uncertainty ({ci} STD)",
        )

        axes[i].set_xticks([])
        axes[i].set_xticklabels([])
        axes[i].set_ylabel(ylabels[int(idx_feature)], fontsize=12)
        axes[i].set_title(
            f"GPE {tags[i]} | {metric_name} = {score_list[i]:.4f} | ISE = {ise_list[i]:.2f} %",
            fontsize=12,
        )
        axes[i].legend(loc="upper left")

    axes[0].set_ylim([np.min(inf_bound), np.max(sup_bound)])
    axes[1].set_ylim([np.min(inf_bound), np.max(sup_bound)])

    fig.tight_layout()
    plt.savefig(
        savepath + "inference_on_testset.pdf", bbox_inches="tight", dpi=1000
    )


if __name__ == "__main__":
    main()
