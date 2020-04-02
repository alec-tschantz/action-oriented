import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import *

plt.rc('text', usetex=True)


def create_heatmap(matrix, title, save_path, color_bar=True):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.7, 4))

    x_labels = [r"$s_{t-1}^{neg}$", r"$s_{t-1}^{pos}$"]
    y_labels = [r"$s_{t}^{neg}$", r"$s_{t}^{pos}$"]
    g1 = sns.heatmap(matrix[0, :, :]*100, cmap="OrRd", ax=ax1, vmin=0.0, vmax=70.0, linewidth=2.5, annot=True,
                     xticklabels=x_labels, yticklabels=y_labels, cbar=color_bar)
    g2 = sns.heatmap(matrix[1, :, :]*100, cmap="OrRd", ax=ax2, vmin=0.0, vmax=70.0, linewidth=2.5, annot=True,
                     xticklabels=x_labels, yticklabels=y_labels, cbar=color_bar)
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0, fontsize=14)
    g1.set_xticklabels(g1.get_xticklabels(), fontsize=14)
    g2.set_yticklabels(g2.get_yticklabels(), rotation=0, fontsize=14)
    g2.set_xticklabels(g2.get_xticklabels(), fontsize=14)

    f.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    states = np.load(STATES_PATH)
    create_heatmap(states[FULL_ID, :, :, :], AGENT_NAMES[FULL_ID], FULL_STATES, color_bar=False)
    create_heatmap(states[INST_ID, :, :, :], AGENT_NAMES[INST_ID], INST_STATES, color_bar=False)
    create_heatmap(states[EPIS_ID, :, :, :], AGENT_NAMES[EPIS_ID], EPIS_STATES, color_bar=False)
    create_heatmap(states[RAND_ID, :, :, :], AGENT_NAMES[RAND_ID], RAND_STATES, color_bar=False)
    create_heatmap(states[RAND_ID, :, :, :], AGENT_NAMES[RAND_ID], COLOR_BAR, color_bar=True)

    reversed_states = np.load(REVERSED_STATES_PATH)
    create_heatmap(reversed_states[FULL_ID, :, :, :], AGENT_NAMES[FULL_ID] + " (Reversed prior)", FULL_STATES_REVERSED,
                   color_bar=False)
    create_heatmap(reversed_states[INST_ID, :, :, :], AGENT_NAMES[INST_ID] + " (Reversed prior)", INST_STATES_REVERSED,
                   color_bar=False)
    create_heatmap(reversed_states[EPIS_ID, :, :, :], AGENT_NAMES[EPIS_ID] + " (Reversed prior)", EPIS_STATES_REVERSED,
                   color_bar=False)
    create_heatmap(reversed_states[RAND_ID, :, :, :], AGENT_NAMES[RAND_ID] + " (Reversed prior)", RAND_STATES_REVERSED,
                   color_bar=False)
