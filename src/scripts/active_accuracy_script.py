import numpy as np
import src.core as core
from src.core.config import *

TRAIN_STEPS = 300
TEST_STEPS = 100


if __name__ == "__main__":
    passive_accuracy = np.zeros([4, N_AVERAGES])
    active_accuracy = np.zeros([4, N_AVERAGES])

    for n in range(N_AVERAGES):

        if n % 20 == 0:
            print("> Processing average {}".format(n))

            full = core.get_mdp(FULL_ID)
            inst = core.get_mdp(INST_ID)
            epis = core.get_mdp(EPIS_ID)
            rand = core.get_mdp(RAND_ID)

            full = core.learn_trial(full, TRAIN_STEPS)
            inst = core.learn_trial(inst, TRAIN_STEPS)
            epis = core.learn_trial(epis, TRAIN_STEPS)
            rand = core.learn_trial(rand, TRAIN_STEPS)

            passive_accuracy[FULL_ID, n] = core.test_passive_accuracy(full, TEST_STEPS)
            passive_accuracy[INST_ID, n] = core.test_passive_accuracy(inst, TEST_STEPS)
            passive_accuracy[EPIS_ID, n] = core.test_passive_accuracy(epis, TEST_STEPS)
            passive_accuracy[RAND_ID, n] = core.test_passive_accuracy(rand, TEST_STEPS)

            active_accuracy[FULL_ID, n] = core.test_active_accuracy(full, TEST_STEPS)
            active_accuracy[INST_ID, n] = core.test_active_accuracy(inst, TEST_STEPS)
            active_accuracy[EPIS_ID, n] = core.test_active_accuracy(epis, TEST_STEPS)
            active_accuracy[RAND_ID, n] = core.test_active_accuracy(rand, TEST_STEPS)

    np.save(ACTIVE_ACCURACY_PATH, active_accuracy)
    np.save(PASSIVE_ACCURACY_PATH, passive_accuracy)
    print("> Data saved")
