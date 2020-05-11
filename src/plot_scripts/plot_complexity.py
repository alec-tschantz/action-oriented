import numpy as np
import matplotlib.pyplot as plt
from core.config import *

TICK_SIZE = 14

FONT_SIZE = 16
plt.rc("ytick", labelsize=16)
plt.rc("xtick", labelsize=16)
plt.rc("legend", fontsize=13)

plt.rcParams["axes.edgecolor"] = "#333F4B"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333F4B"
plt.rcParams["ytick.color"] = "#333F4B"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

plt.rc("text", usetex=True)


def plot_complexity(matrix, title, save_path):
    colors = get_color_palette()
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))
    positions = [-0.3, -0.1, 0.1, 0.3]
    x_axis = np.array([0.0, 1.0])

    x_labels_a = [
        r"$P_{\theta}(s_{t}|s_{t-1}^{neg}, u_{t-1}^{tumble})$",
        r"$P_{\theta}(s_{t}|s_{t-1}^{pos}, u_{t-1}^{tumble})$",
    ]

    x_labels_b = [
        r"$P_{\theta}(s_{t}|s_{t-1}^{neg}, u_{t-1}^{run})$",
        r"$P_{\theta}(s_{t}|s_{t-1}^{pos}, u_{t-1}^{run})$",
    ]

    for i in range(N_AGENTS):
        values = matrix[i, :, :]
        mean_values = np.mean(values, axis=1)
        _ = np.std(values, axis=1) / np.sqrt(N_AVERAGES)

        x_values = x_axis + positions[i]
        ax1.bar(
            x_values,
            mean_values[:2],
            align="center",
            width=0.2,
            color=colors[i],
            edgecolor="white",
            label=AGENT_NAMES[i],
        )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_smart_bounds(True)
        ax1.spines["bottom"].set_smart_bounds(True)
        ax1.set_xticks(x_axis)
        ax1.set_xticklabels(x_labels_a)
        ax1.tick_params(axis="x", which="major", pad=15)

        ax2.bar(
            x_values,
            mean_values[2:],
            align="center",
            width=0.2,
            color=colors[i],
            edgecolor="white",
            label=AGENT_NAMES[i],
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_smart_bounds(True)
        ax2.spines["bottom"].set_smart_bounds(True)
        ax2.set_xticks(x_axis)
        ax2.set_xticklabels(x_labels_b)
        ax2.tick_params(axis="x", which="major", pad=15)

    legend = plt.legend()
    f.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    complexity = np.load(COMPLEXITY_PATH)
    plot_complexity(complexity, "Change in distributions", COMPLEXITY)
    reversed_complexity = np.load(REVERSED_COMPLEXITY_PATH)
    plot_complexity(
        reversed_complexity, "Change in distributions (reversed prior)", REVERSED_COMPLEXITY
    )
