import numpy as np
from core.config import *
from core.mdp import MDP


def get_mdp(agent_id, reverse_prior=False):
    a = np.eye(N_OBS)
    b = np.random.rand(N_CONTROL, N_STATES, N_STATES)
    c = np.zeros([N_OBS, 1])

    if reverse_prior:
        c[0] = 1
    else:
        c[PRIOR_ID] = 1

    kwargs = {}
    if agent_id == FULL_ID:
        kwargs = {"alpha": ALPHA, "beta": 1, "lr": LR}
    elif agent_id == INST_ID:
        kwargs = {"alpha": ALPHA, "beta": 0, "lr": LR}
    elif agent_id == EPIS_ID:
        kwargs = {"alpha": 0, "beta": 1, "lr": LR}
    elif agent_id == RAND_ID:
        kwargs = {"alpha": 0, "beta": 0, "lr": LR}

    mdp = MDP(a, b, c, **kwargs)
    return mdp


def get_true_model():
    b = np.zeros([N_CONTROL, N_STATES, N_STATES])
    b[TUMBLE, :, :] = np.array([[0.5, 0.5], [0.5, 0.5]])
    b[RUN, :, :] = np.array([[1, 0], [0, 1]])
    b += np.exp(-16)
    return b
