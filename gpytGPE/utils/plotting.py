import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def get_col(color_name=None):
    """Material Design color palettes (only '100' and '900' variants).
    Help: call with no arguments to see the list of available colors, these are also returned into a list
    Kwarg:
            - color_name: string representing the color's name
    Output:
            - color: list of two elements
                    [0] = lightest color '100'-variant (RGB-triplet in [0, 1])
                    [1] = darkest color '900'-variant (RGB-triplet in [0, 1])
    """
    colors = {
        "red": [[255, 205, 210], [183, 28, 28]],
        "pink": [[248, 187, 208], [136, 14, 79]],
        "purple": [[225, 190, 231], [74, 20, 140]],
        "deep_purple": [[209, 196, 233], [49, 27, 146]],
        "indigo": [[197, 202, 233], [26, 35, 126]],
        "blue": [[187, 222, 251], [13, 71, 161]],
        "light_blue": [[179, 229, 252], [1, 87, 155]],
        "cyan": [[178, 235, 242], [0, 96, 100]],
        "teal": [[178, 223, 219], [0, 77, 64]],
        "green": [[200, 230, 201], [27, 94, 32]],
        "light_green": [[220, 237, 200], [51, 105, 30]],
        "lime": [[240, 244, 195], [130, 119, 23]],
        "yellow": [[255, 249, 196], [245, 127, 23]],
        "amber": [[255, 236, 179], [255, 111, 0]],
        "orange": [[255, 224, 178], [230, 81, 0]],
        "deep_orange": [[255, 204, 188], [191, 54, 12]],
        "brown": [[215, 204, 200], [62, 39, 35]],
        "gray": [[245, 245, 245], [33, 33, 33]],
        "blue_gray": [[207, 216, 220], [38, 50, 56]],
    }
    if not color_name:
        print("\n=== Colors available are:")
        for key, _ in colors.items():
            print("- " + key)
        return list(colors.keys())
    else:
        color = [
            [colors[color_name][i][j] / 255 for j in range(3)]
            for i in range(2)
        ]
        return color


def interp_col(color, n):
    """Linearly interpolate a color.
    Args:
            - color: list with two elements:
                    color[0] = lightest color variant (get_col('color_name')[0])
                    color[1] = darkest color variant (get_col('color_name')[1]).
            - n: number of desired output colors (n >= 2).
    Output:
            - lsc: list of n linearly scaled colors.
    """
    c = [
        np.interp(list(range(1, n + 1)), [1, n], [color[0][i], color[1][i]])
        for i in range(3)
    ]
    lsc = [[c[0][i], c[1][i], c[2][i]] for i in range(n)]
    return lsc


def check_distr_mean(A, threshold):
    mean = []
    std = []
    for a in A.T:
        mean.append(a.mean())
        std.append(a.std())
    l = []
    for i, (m, s) in enumerate(zip(mean, std)):
        if m < threshold or m - 3 * s < 0:
            l.append(i)
    return l


def check_distr_lquart(A, threshold):
    lower_quartile = []
    for a in A.T:
        lower_quartile.append(np.percentile(a, 25))
    l = []
    for i, q1 in enumerate(lower_quartile):
        if q1 < threshold:
            l.append(i)
    return l


def correct_index(
    A, threshold, criterion="mean"
):  # NOTE: when using criterion="lquart", np.median has to be used instead of np.mean in gsa_donut, gsa_heat, gsa_network
    if criterion == "mean":
        l = check_distr_mean(A, threshold)
    elif criterion == "lquart":
        l = check_distr_lquart(A, threshold)
    else:
        raise ValueError(
            "Not a valid criterion! Available criteria are: 'mean' or 'lquart'."
        )
    A[:, l] = np.zeros((A.shape[0], len(l)), dtype=float)
    return A


def angle(p, c):
    [dx, dy] = p - c

    if dx == 0:
        if dy > 0:
            return 0.5 * np.pi
        else:
            return 1.5 * np.pi
    elif dx > 0:
        if dy >= 0:
            return np.arctan(dy / dx)
        else:
            return 2.0 * np.pi + np.arctan(dy / dx)
    elif dx < 0:
        return np.pi + np.arctan(dy / dx)


def gsa_box(ST, S1, S2, index_i, index_ij, ylabel, savepath, correction=None):
    if correction is not None:
        ST = correct_index(ST, correction)
        S1 = correct_index(S1, correction)
        S2 = correct_index(S2, correction)

    df_ST = pd.DataFrame(data=ST, columns=index_i)
    df_S1 = pd.DataFrame(data=S1, columns=index_i)
    df_S2 = pd.DataFrame(
        data=S2,
        columns=["(" + elem[0] + ", " + elem[1] + ")" for elem in index_ij],
    )

    plt.style.use("seaborn")
    gs = grsp.GridSpec(2, 2)
    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(2 * width, 2 * height / 2))

    ax0 = fig.add_subplot(gs[0, 0])
    sns.boxplot(ax=ax0, data=df_S1)
    ax0.set_ylim(0, 1)
    ax0.set_title("First-order effect", fontweight="bold", fontsize=12)
    ax0.set_xticklabels(
        ax0.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    ax1 = fig.add_subplot(gs[0, 1])
    sns.boxplot(ax=ax1, data=df_ST)
    ax1.set_ylim(0, 1)
    ax1.set_title("Total effect", fontweight="bold", fontsize=12)
    ax1.set_xticklabels(
        ax1.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    ax2 = fig.add_subplot(gs[1, :])
    sns.boxplot(ax=ax2, data=df_S2)
    ax2.set_ylim(0, 1)
    ax2.set_title("Second-order effect", fontweight="bold", fontsize=12)
    ax2.set_xticklabels(
        ax2.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    fig.tight_layout()
    plt.savefig(savepath + ylabel + "_box.pdf", bbox_inches="tight", dpi=1000)


def gsa_donut(ST, S1, index_i, ylabel, savepath, correction=None):
    if correction is not None:
        ST = correct_index(ST, correction)
        S1 = correct_index(S1, correction)

    ST_mean = np.mean(ST, axis=0)
    S1_mean = np.mean(S1, axis=0)

    sum_s1 = S1_mean.sum()
    sum_st = ST_mean.sum()
    ho = sum_st - sum_s1
    x_si = np.array(list(S1_mean) + [ho])
    x_sti = ST_mean

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 4))

    c = "blue"
    colors = interp_col(get_col(c), len(index_i))
    colors += [interp_col(get_col("gray"), 6)[2]]

    wedges, _ = axes[0].pie(
        x_si,
        radius=1,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    axes[0].set_title("S1", fontsize=12, fontweight="bold")

    axes[1].pie(
        x_sti,
        radius=1,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    axes[1].set_title("ST", fontsize=12, fontweight="bold")

    plt.figlegend(
        wedges, index_i + ["higher-order int."], ncol=5, loc="lower center"
    )
    plt.savefig(
        savepath + ylabel + "_donut.pdf", bbox_inches="tight", dpi=1000
    )


def gsa_heat(ST, S1, index_i, ylabel, savepath, correction=None):
    if correction is not None:
        ST = correct_index(ST, correction)
        S1 = correct_index(S1, correction)

    ST_mean = np.mean(ST, axis=0).reshape(1, -1)
    S1_mean = np.mean(S1, axis=0).reshape(1, -1)

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 8))

    df = pd.DataFrame(data=S1_mean, index=[ylabel], columns=index_i)
    h1 = sns.heatmap(
        df,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidth=0.1,
        cbar_kws={"shrink": 0.8},
        ax=axes[0],
    )
    axes[0].set_title("S1", fontsize=12, fontweight="bold")
    axes[0].tick_params(left=False, bottom=False)
    h1.set_xticklabels(h1.get_xticklabels(), rotation=45, va="top")
    h1.set_yticklabels(h1.get_yticklabels(), rotation=0, ha="right")

    df = pd.DataFrame(data=ST_mean, index=[ylabel], columns=index_i)
    ht = sns.heatmap(
        df,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidth=0.1,
        cbar_kws={"shrink": 0.8},
        ax=axes[1],
    )
    axes[1].set_title("ST", fontsize=12, fontweight="bold")
    axes[1].tick_params(left=False, bottom=False)
    ht.set_xticklabels(ht.get_xticklabels(), rotation=45, va="top")
    ht.set_yticklabels(ht.get_yticklabels(), rotation=0, ha="right")

    plt.savefig(savepath + ylabel + "_heat.pdf", bbox_inches="tight", dpi=1000)


def gsa_network(
    ST, S1, S2, index_i, index_ij, ylabel, savepath, correction=None
):
    if correction is not None:
        ST = correct_index(ST, correction)
        S1 = correct_index(S1, correction)
        S2 = correct_index(S2, correction)

    ST_mean = np.mean(ST, axis=0)
    S1_mean = np.mean(S1, axis=0)
    S2_mean = np.mean(S2, axis=0)

    maximum = np.max([ST_mean.max(), S1_mean.max(), S2_mean.max()])
    ST_mean /= maximum
    S1_mean /= maximum
    S2_mean /= maximum

    min_size = 0
    max_size = 200
    foreground_node_size = [
        min_size + (max_size - min_size) * k for k in list(S1_mean)
    ]
    backgroud_node_size = [
        min_size + (max_size - min_size) * k for k in list(ST_mean)
    ]
    edge_width = [
        np.sqrt((min_size + (max_size - min_size) * k) / np.pi)
        for k in list(S2_mean)
    ]

    Sources = list(list(zip(*index_ij))[0])
    Targets = list(list(zip(*index_ij))[1])
    Weights = list(S2_mean)

    G = nx.Graph()
    for s, t, w in zip(Sources, Targets, Weights):
        G.add_edges_from([(s, t)], w=w)

    Pos = nx.circular_layout(G)

    c = "blue"
    colors = interp_col(get_col(c), 5)
    width = 5.91667
    fig, axis = plt.subplots(1, 1, figsize=(width, width))

    nx.draw_networkx_nodes(
        G,
        Pos,
        node_size=backgroud_node_size,
        node_color=len(index_i) * [colors[4]],
        ax=axis,
    )
    nx.draw_networkx_nodes(
        G,
        Pos,
        node_size=foreground_node_size,
        node_color=len(index_i) * [colors[0]],
        ax=axis,
    )
    nx.draw_networkx_edges(
        G,
        Pos,
        width=edge_width,
        edge_color=len(index_ij) * [colors[2]],
        alpha=0.8,
        ax=axis,
    )

    center = [0.0, 0.0]
    radius = 1.0

    names = nx.draw_networkx_labels(
        G, Pos, font_size=12, font_family="DejaVu Sans", ax=axis
    )
    for node, text in names.items():
        position = (
            1.2 * radius * np.cos(angle(Pos[node], center)),
            1.2 * radius * np.sin(angle(Pos[node], center)),
        )
        text.set_position(position)
        text.set_clip_on(False)

    axis.axis("equal")
    axis.set_axis_off()
    fig.tight_layout()
    plt.savefig(
        savepath + ylabel + "_network.pdf", bbox_inches="tight", dpi=1000
    )


def plot_dataset(Xdata, Ydata, xlabels, ylabels, savepath):
    """Plot Y high-dimensional dataset by pairwise plotting its features against each X dataset's feature.
    Args:
            - Xdata: n*m1 matrix
            - Ydata: n*m2 matrix
            - xlabels: list of m1 strings representing the name of X dataset's features
            - ylabels: list of m2 strings representing the name of Y dataset's features.
    """
    height = 9.36111
    width = 5.91667
    sample_dim = Xdata.shape[0]
    in_dim = Xdata.shape[1]
    out_dim = Ydata.shape[1]
    fig, axes = plt.subplots(
        nrows=out_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(2 * width, 2 * height / 3),
    )
    for i, axis in enumerate(axes.flatten()):
        axis.scatter(
            Xdata[:, i % in_dim], Ydata[:, i // in_dim], fc="C0", ec="C0"
        )
        inf = min(Xdata[:, i % in_dim])
        sup = max(Xdata[:, i % in_dim])
        mean = 0.5 * (inf + sup)
        delta = sup - mean
        if i // in_dim == out_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim])
            axis.set_xlim(left=inf - 0.3 * delta, right=sup + 0.3 * delta)
        if i % in_dim == 0:
            axis.set_ylabel(ylabels[i // in_dim])
    plt.suptitle("Sample dimension = {} points".format(sample_dim))
    plt.savefig(savepath + "X_vs_Y.png", bbox_inches="tight", dpi=300)
