import matplotlib.pyplot as plt
import numpy as np
import core
from core.config import *

plt.rcParams["axes.edgecolor"] = "#333F4B"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333F4B"
plt.rcParams["ytick.color"] = "#333F4B"

if __name__ == "__main__":
    failed_raw = np.load("data/failed.npy")
    failed_sum = np.sum(failed_raw, axis=1)

    colors = core.get_color_palette()
    x_ticks = range(N_AGENTS)
    x_labels = AGENT_NAMES
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar(range(N_AGENTS), failed_sum, color=colors, edgecolor="grey", align="center")
    plt.xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_smart_bounds(True)
    ax.spines["bottom"].set_smart_bounds(True)

    f.savefig(FAILED_MODELS, dpi=600, bbox_inches="tight")
    plt.show()
