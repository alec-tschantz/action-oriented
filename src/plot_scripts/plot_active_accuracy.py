import numpy as np
import core
from core.config import *

FONT_SIZE = 20
plt.rc('ytick', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('legend', fontsize=FONT_SIZE)

plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

if __name__ == "__main__":
    active_accuracy = np.load(ACTIVE_ACCURACY_PATH)
    passive_accuracy = np.load(PASSIVE_ACCURACY_PATH)

    colors = core.get_color_palette()
    positions = [-0.24, -0.08, 0.08, 0.24]
    x = np.array([0, 1])
    x_ticks = (np.arange(2))

    f, ax = plt.subplots(1, 1, figsize=(10, 7))

    for agent_id in range(N_AGENTS):
        avg = np.round(np.mean(passive_accuracy[agent_id, :]), 1)
        sem = np.std(passive_accuracy[agent_id, :]) / np.sqrt(N_AVERAGES)
        plt.bar(0 + positions[agent_id], avg, width=0.16, color=colors[agent_id], align='center',
                label=AGENT_NAMES[agent_id], edgecolor='white')

    for agent_id in range(N_AGENTS):
        avg = np.round(np.mean(active_accuracy[agent_id, :]), 1)
        sem = np.std(active_accuracy[agent_id, :]) / np.sqrt(N_AVERAGES)
        plt.bar(1 + positions[agent_id], avg, width=0.16, color=colors[agent_id],
                align='center', edgecolor='white')

    plt.xticks(x_ticks, fontsize=FONT_SIZE)
    ax.set_xticklabels(['Passive error', 'Active error'])

    legend = plt.legend()
    frame = legend.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_ylabel('Total M.S.E error', {"size": FONT_SIZE}, labelpad=10)
    f.savefig(ACTIVE_ACCURACY, dpi=600, bbox_inches='tight')
    plt.show()
