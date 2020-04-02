import numpy as np


class MDP(object):

    def __init__(self, a, b, c,
                 lr=0.1,
                 alpha=1,
                 beta=1):

        self.A = a
        self.B = b
        self.C = c

        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.p0 = np.exp(-16)

        if np.size(self.C, 1) > np.size(self.C, 0):
            self.C = self.C.T

        self.Ns = self.A.shape[1]
        self.No = self.A.shape[0]
        self.Nu = self.B.shape[0]

        self.A = self.A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

        self.B = self.B + self.p0
        for u in range(self.Nu):
            self.B[u] = self.normdist(self.B[u])
        self.Ba = np.copy(self.B)
        self.wB = 0
        self.calc_wb()

        self.true_B = self.get_true_model()

        self.C = self.C + self.p0
        self.C = self.normdist(self.C)

        self.sQ = np.zeros([self.Ns, 1])
        self.uQ = np.zeros([self.Nu, 1])
        self.EFE = np.zeros([self.Nu, 1])

        self.action_range = np.arange(0, self.Nu)
        self.obv = 0
        self.action = 0

    def reset(self, obv):
        self.obv = obv
        likelihood = self.lnA[obv, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
        self.action = int(np.random.choice(self.action_range))

    def step(self, obv):
        self.obv = obv
        self.infer_sQ(obv)
        self.evaluate_efe()
        self.infer_uq()
        return self.act()

    def infer_sQ(self, obv):
        likelihood = self.lnA[obv, :]
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B[self.action], self.sQ)
        prior = np.log(prior)
        self.sQ = self.softmax(likelihood + prior)

    def evaluate_efe(self):
        self.EFE = np.zeros([self.Nu, 1])

        for u in range(self.Nu):
            fs = np.dot(self.B[u], self.sQ)
            fo = np.dot(self.A, fs)
            fo = self.normdist(fo + self.p0)

            utility = (np.sum(fo * np.log(fo / self.C), axis=0)) * self.alpha
            utility = utility[0]
            surprise = self.bayesian_surprise(u, fs) * self.beta

            self.EFE[u] -= utility
            self.EFE[u] += surprise

    def infer_uq(self):
        self.uQ = self.softmax(self.EFE)

    def update(self, action, new, previous):
        self.Ba[action, new, previous] += self.lr
        b = np.copy(self.Ba[action])
        self.B[action] = self.normdist(b)
        self.calc_wb()

    def calc_expectation(self):
        for u in range(self.Nu):
            b = np.copy(self.Ba[u])
            self.B[u] = self.normdist(b)
        self.calc_wb()

    def calc_wb(self):
        wb_norm = np.copy(self.Ba)
        wb_avg = np.copy(self.Ba)

        for u in range(self.Nu):
            for s in range(self.Ns):
                wb_norm[u, :, s] = np.divide(1.0, np.sum(wb_norm[u, :, s]))
                wb_avg[u, :, s] = np.divide(1.0, (wb_avg[u, :, s]))

        self.wB = wb_norm - wb_avg

    def act(self):
        hu = max(self.uQ)
        options = np.where(self.uQ == hu)[0]
        self.action = int(np.random.choice(options))
        return self.action

    def bayesian_surprise(self, u, fs):
        surprise = 0
        wb = self.wB[u, :, :]
        for st in range(self.Ns):
            for s in range(self.Ns):
                surprise += fs[st] * wb[st, s] * self.sQ[s]
        return -surprise

    def predict_obv(self, action, obv):
        _obv = np.zeros([2, 1]) + self.p0
        _obv[obv] = 1
        fs = np.dot(self.B[action], _obv)
        fo = np.dot(self.A, fs)
        fo = self.normdist(fo + self.p0)

        tfs = np.dot(self.true_B[action], _obv)
        tfo = np.dot(self.A, tfs)
        tfo = self.normdist(tfo + self.p0)
        return fo, tfo

    @staticmethod
    def entropy(fs):
        fs = fs[:, 0]
        return -np.sum(fs * np.log(fs), axis=0)

    @staticmethod
    def softmax(x):
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x):
        return np.dot(x, np.diag(1 / np.sum(x, 0)))
        
    @staticmethod
    def get_true_model():
        b = np.zeros([2, 2, 2])
        b[0, :, :] = np.array([[0.5, 0.5], [0.5, 0.5]])
        b[1, :, :] = np.array([[1, 0], [0, 1]])
        b += np.exp(-16)
        return b
