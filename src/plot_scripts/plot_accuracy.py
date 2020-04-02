import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import *

TICK_SIZE = 16
LEGEND_SIZE = 14
LABEL_SIZE = 20
FIG_SIZE = [9, 7]

FONT_SIZE = 18
plt.rc('ytick', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('legend', fontsize=FONT_SIZE)

plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'


N_PLOT_EPOCHS = 40
N_AVERAGES = 300


def convert_steps_to_ticks(_steps, _epochs):
    _steps = _steps[0:_epochs]
    _steps = _steps - 20
    _steps = [item for item in _steps.astype(str)]
    _x_ticks = (np.arange(0, _epochs, 10))
    _labels = [_steps[int(i)] for i in _x_ticks]
    return _x_ticks, _labels


if __name__ == "__main__":
    steps = np.load(STEPS_PATH)
    raw_accuracy = np.load(ACCURACY_PATH)
    x_range = range(N_PLOT_EPOCHS)

    colors = get_color_palette()
    x_ticks, x_labels = convert_steps_to_ticks(steps, N_PLOT_EPOCHS)

    fig, ax = plt.subplots()
    fig.set_size_inches(FIG_SIZE[0], FIG_SIZE[1])

    for agent_id in range(N_AGENTS):
        mean = 2.2 - np.mean(raw_accuracy[agent_id, 0:N_PLOT_EPOCHS, :], axis=1)
        std = np.std(raw_accuracy[agent_id, 0:N_PLOT_EPOCHS, :], axis=1) / np.sqrt(N_AVERAGES)
        high = mean + std
        low = mean - std

        ax.plot(x_range, mean, color=colors[agent_id], lw=2.5, label=AGENT_NAMES[agent_id])
        ax.fill_between(x_range, high, low, color=colors[agent_id], alpha=0.2, linewidth=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    legend = plt.legend()
    frame = legend.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')

    plt.xlabel("Number of learning steps", {'size': LABEL_SIZE})
    plt.ylabel("(-ve) KL from true model", {'size': LABEL_SIZE})

    plt.xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    plt.savefig(ACCURACY, dpi=600, bbox_inches='tight')
    plt.show()
