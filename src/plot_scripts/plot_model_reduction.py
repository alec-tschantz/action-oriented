import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import core
from core.config import *

plt.rc("text", usetex=True)
plt.rcParams["axes.edgecolor"] = "#333F4B"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333F4B"
plt.rcParams["ytick.color"] = "#333F4B"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rc("ytick", labelsize=12)
plt.rc("xtick", labelsize=12)


def create_heatmap(matrix, title, save_path, color_bar=False):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.7, 4))

    x_labels = [r"$s_{t-1}^{neg}$", r"$s_{t-1}^{pos}$"]
    y_labels = [r"$s_{t}^{neg}$", r"$s_{t}^{pos}$"]
    g1 = sns.heatmap(
        matrix[0, :, :] * 100,
        cmap="OrRd",
        ax=ax1,
        vmin=0.0,
        vmax=100.0,
        linewidth=2.5,
        annot=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=color_bar,
    )
    g2 = sns.heatmap(
        matrix[1, :, :] * 100,
        cmap="OrRd",
        ax=ax2,
        vmin=0.0,
        vmax=100.0,
        linewidth=2.5,
        annot=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=color_bar,
    )
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0, fontsize=14)
    g1.set_xticklabels(g1.get_xticklabels(), fontsize=14)
    g2.set_yticklabels(g2.get_yticklabels(), rotation=0, fontsize=14)
    g2.set_xticklabels(g2.get_xticklabels(), fontsize=14)

    f.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    pruned = np.load(PRUNED_PATH)

    create_heatmap(pruned[FULL_ID, :, :, :], AGENT_NAMES[FULL_ID], FULL_PRUNED)
    create_heatmap(pruned[INST_ID, :, :, :], AGENT_NAMES[INST_ID], INST_PRUNED)
    create_heatmap(pruned[EPIS_ID, :, :, :], AGENT_NAMES[EPIS_ID], EPIS_PRUNED)
    create_heatmap(pruned[RAND_ID, :, :, :], AGENT_NAMES[RAND_ID], RAND_PRUNED)
    create_heatmap(pruned[RAND_ID, :, :, :], AGENT_NAMES[RAND_ID], COLOR_BAR_PRUNED, color_bar=True)

    full_total = np.sum(pruned[FULL_ID, :, :, :])
    inst_total = np.sum(pruned[INST_ID, :, :, :])
    epis_total = np.sum(pruned[EPIS_ID, :, :, :])
    rand_total = np.sum(pruned[RAND_ID, :, :, :])

    colors = core.get_color_palette()
    x_ticks = range(N_AGENTS)
    x_labels = AGENT_NAMES
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar(
        range(N_AGENTS),
        [full_total, inst_total, epis_total, rand_total],
        color=colors,
        edgecolor="grey",
        align="center",
    )
    plt.xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_smart_bounds(True)
    ax.spines["bottom"].set_smart_bounds(True)

    f.savefig(TOTAL_PRUNED, dpi=600, bbox_inches="tight")
    plt.show()

    # model does not need to know about what happens when you run in negative gradients, or tumble in positive gradients
    # we are looking at the learned model - what redundant priors does it entail *in the presence of action*?
