import numpy as np
from scipy.special import gamma
import pprint
import src.core as core
from src.core.config import *


def ln_beta_fn(vec):
    numerator = 1
    for a in vec:
        numerator = numerator * gamma(a)
    denominator = sum(vec)
    denominator = gamma(denominator)
    return np.log(numerator / denominator)


def calc_df(prior, posterior, reduced_prior):
    prior = np.squeeze(prior)
    posterior = np.squeeze(posterior)
    posterior_reduced = posterior + reduced_prior - prior

    term_1 = ln_beta_fn(posterior)
    term_2 = ln_beta_fn(reduced_prior)
    term_3 = ln_beta_fn(prior)
    term_4 = ln_beta_fn(posterior_reduced)

    df = term_1 + term_2 - term_3 - term_4
    return df


def calc_dfs(prior, posterior):
    p_0 = np.exp(-16)
    _dfs = np.zeros([N_CONTROL, N_STATES, N_STATES])

    for u in range(N_CONTROL):
        for s_t0 in range(N_STATES):
            _prior = prior[u, s_t0, :]
            _posterior = posterior[u, s_t0, :]

            for s_t1 in range(N_STATES):
                reduced_prior = np.copy(_prior)
                reduced_prior[s_t1] = p_0
                df = calc_df(_prior, _posterior, reduced_prior)
                _dfs[u, s_t0, s_t1] = df
    return _dfs


def perform_model_reduction(agent_id):
    print("> Processing agent {}".format(AGENT_NAMES[agent_id]))
    _dfs = np.zeros([N_CONTROL, N_STATES, N_STATES, N_AVERAGES])
    _pruned_priors = np.zeros([N_CONTROL, N_STATES, N_STATES, N_AVERAGES])

    for n in range(N_AVERAGES):
        if n % 10 == 0:
            print("> Processing average [{}/{}]".format(n, N_AVERAGES))
        mdp = core.get_mdp(agent_id)
        mdp = core.learn_trial(mdp, TEST_TRIAL_LEN * 4)
        prior = np.copy(mdp.Ba)
        mdp = core.learn_trial(mdp, MODEL_REDUCTION_TRIAL_LEN)
        _trial_dfs = calc_dfs(prior, mdp.Ba)
        _trial_pruned = np.zeros([N_CONTROL, N_STATES, N_STATES])
        _trial_pruned[_trial_dfs < 0.0] = 1
        _dfs[:, :, :, n] = _trial_dfs
        _pruned_priors[:, :, :, n] = _trial_pruned

    return _dfs, _pruned_priors


if __name__ == "__main__":
    print("> Processing model reduction")
    pruned = np.zeros([N_AGENTS, N_CONTROL, N_STATES, N_STATES])

    dfs_full, pruned_full = perform_model_reduction(FULL_ID)
    dfs_mean = np.mean(dfs_full[:, :, :, :], axis=-1)
    pruned_sum = np.sum(pruned_full[:, :, :, :], axis=-1)
    pprint.pprint(dfs_mean)
    pprint.pprint(pruned_sum / N_AVERAGES)
    print(np.sum(pruned_sum) / N_AVERAGES)
    pruned[FULL_ID, :, :, :] = pruned_sum / N_AVERAGES

    dfs_inst, pruned_inst = perform_model_reduction(INST_ID)
    dfs_mean = np.mean(dfs_inst[:, :, :, :], axis=-1)
    pruned_sum = np.sum(pruned_inst[:, :, :, :], axis=-1)
    pprint.pprint(dfs_mean)
    pprint.pprint(pruned_sum / N_AVERAGES)
    print(np.sum(pruned_sum) / N_AVERAGES)
    pruned[INST_ID, :, :, :] = pruned_sum / N_AVERAGES

    dfs_epis, pruned_epis = perform_model_reduction(EPIS_ID)
    dfs_mean = np.mean(dfs_epis[:, :, :, :], axis=-1)
    pruned_sum = np.sum(pruned_epis[:, :, :, :], axis=-1)
    pprint.pprint(dfs_mean)
    pprint.pprint(pruned_sum / N_AVERAGES)
    print(np.sum(pruned_sum) / N_AVERAGES)
    pruned[EPIS_ID, :, :, :] = pruned_sum / N_AVERAGES

    dfs_rand, pruned_rand = perform_model_reduction(RAND_ID)
    dfs_mean = np.mean(dfs_rand[:, :, :, :], axis=-1)
    pruned_sum = np.sum(pruned_rand[:, :, :, :], axis=-1)
    pprint.pprint(dfs_mean)
    pprint.pprint(pruned_sum / N_AVERAGES)
    print(np.sum(pruned_sum) / N_AVERAGES)
    pruned[RAND_ID, :, :, :] = pruned_sum / N_AVERAGES

    np.save(PRUNED_PATH, pruned)
    print("> Data saved")

    # maybe increase trial len
