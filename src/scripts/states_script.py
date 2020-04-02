import numpy as np
import src.core as core
from src.core.config import *


def get_state_dist(agent_id, reverse_prior=False):
    print("> Processing agent: {}".format(AGENT_NAMES[agent_id]))
    state_ensemble = np.zeros([N_AVERAGES, N_CONTROL, N_STATES, N_STATES])
    for sample in range(N_AVERAGES):
        if sample % 50 == 0:
            print("> Processing average [{}/{}]".format(sample, N_AVERAGES))
        mdp = core.get_mdp(agent_id, reverse_prior=reverse_prior)
        mdp, states_trial = core.learn_trial(mdp, TEST_TRIAL_LEN, record_states=True)
        state_ensemble[sample, :, :, :] = states_trial

    states_sum = np.sum(state_ensemble, axis=0)
    states_dist = states_sum / np.sum(states_sum)
    return np.round(states_dist, 3)


if __name__ == "__main__":
    print("\n> Processing state ensemble")
    states = np.zeros([N_AGENTS, N_CONTROL, N_STATES, N_STATES])
    states[FULL_ID, :, :, :] = get_state_dist(FULL_ID)
    states[INST_ID, :, :, :] = get_state_dist(INST_ID)
    states[EPIS_ID, :, :, :] = get_state_dist(EPIS_ID)
    states[RAND_ID, :, :, :] = get_state_dist(RAND_ID)

    print("\n> Processing state ensemble (reversed prior)")
    reversed_states = np.zeros([N_AGENTS, N_CONTROL, N_STATES, N_STATES])
    reversed_states[FULL_ID, :, :, :] = get_state_dist(FULL_ID, reverse_prior=True)
    reversed_states[INST_ID, :, :, :] = get_state_dist(INST_ID, reverse_prior=True)
    reversed_states[EPIS_ID, :, :, :] = get_state_dist(EPIS_ID, reverse_prior=True)
    reversed_states[RAND_ID, :, :, :] = get_state_dist(RAND_ID, reverse_prior=True)

    np.save(STATES_PATH, states)
    np.save(REVERSED_STATES_PATH, reversed_states)
    print("> Data saved")
