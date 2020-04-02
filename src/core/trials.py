import numpy as np
from core.env import Environment
from core.config import *


def learn_trial(mdp, n_steps, record_states=False):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    states = np.zeros([N_CONTROL, N_STATES, N_STATES])

    for step in range(n_steps):
        prev_obv = obv
        action = mdp.step(obv)
        obv = env.act(action)
        mdp.update(action, obv, prev_obv)
        if record_states:
            states[action, obv, prev_obv] += 1

    if record_states:
        return mdp, states
    return mdp


def test_distance(mdp, steps):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)

    for _ in range(steps):
        action = mdp.step(obv)
        obv = env.act(action)

    return (env.distance() - env.source_size) + 1


def test_passive_accuracy(mdp, n_steps):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    acc = 0

    for _ in range(n_steps):
        random_action = np.random.choice([0, 1])
        pred, t_pred = mdp.predict_obv(random_action, obv)
        _ = mdp.step(obv)
        obv = env.act(random_action)
        acc += diff(t_pred, pred)

    return acc


def test_active_accuracy(mdp, n_steps):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    acc = 0

    for _ in range(n_steps):
        action = mdp.step(obv)
        pred, t_pred = mdp.predict_obv(action, obv)
        acc += diff(t_pred, pred)
        obv = env.act(action)

    return acc


def diff(p, q):
    return np.mean(np.square(p - q))
