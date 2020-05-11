import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc("legend", fontsize=16)

plt.rcParams["axes.edgecolor"] = "#333F4B"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333F4B"
plt.rcParams["ytick.color"] = "#333F4B"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rc("ytick", labelsize=17)
plt.rc("xtick", labelsize=17)

if __name__ == "__main__":
    T = 2500
    buffer = 25
    Tb = int(T / buffer)

    x_ticks = np.arange(0, Tb, 20)
    labels = [int(i) * 25 for i in x_ticks]

    x = np.load("data/metrics.npy", allow_pickle=True)
    change_x = x[0][0]
    t = x[1]
    efe_run = x[2]
    efe_tumble = x[3]
    util_tumble = x[4]
    epi_tumble = x[5]
    util_run = x[6]
    epi_run = x[7]

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 7)

    np.save("data/metrics", x)

    ax.plot(t, efe_run, lw=5, label="EFE Run")
    ax.plot(t, efe_tumble, lw=5, label="EFE Tumble")

    ax.plot(t, util_tumble, lw=4, linestyle="--", label="Instrumental Tumble ")
    ax.plot(t, epi_tumble, lw=4, linestyle="--", label="Epistemic Tumble")

    ax.plot(t, util_run, lw=4, linestyle="--", label="Instrumental Run")
    ax.plot(t, epi_run, lw=4, linestyle="--", label="Epistemic Run")

    plt.axvline(x=0, color="#7b7b7b", linestyle="-.", lw=3)
    plt.axvline(x=change_x, color="#7b7b7b", linestyle="-.", lw=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = plt.legend()
    frame = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")

    plt.xticks(x_ticks)
    ax.set_xticklabels(labels)
    ax.spines["left"].set_smart_bounds(True)
    ax.spines["bottom"].set_smart_bounds(True)

    ax.set_xlabel("Number of time steps", {"size": 19})
    ax.set_ylabel("Bits", {"size": 19})

    fig.savefig("figs/metrics.pdf", dpi=600, bbox_inches="tight")
    plt.show()
