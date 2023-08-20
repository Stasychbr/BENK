import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.metrics.pairwise import rbf_kernel
from numba import jit, float32, int64
import warnings


def sf_to_t(S, T):
    t_diff = T[1:] - T[:-1]
    integral = S[:, :-1] * t_diff[None, :]
    return T[0] + np.sum(integral, axis=-1)

# W and T already sorted
@jit(float32[:, ::1](float32[:, ::1], int64[::1]), nopython=True)
def nw_helper_func(W, D):
    k = W.shape[0]
    n = W.shape[1]
    S = np.empty((k, n), np.float32)
    if D[0] == 1:
        S[:, 0] = 1 - W[:, 0]
    else:
        S[:, 0] = 1
    for i in range(1, n):
        if D[i] == 0:
            S[:, i] = 1 * S[:, i - 1]
            continue
        weight_sum = np.zeros(k, dtype=np.float32)
        for j in range(i):
            weight_sum += W[:, j]
        cur_S = 1 - W[:, i] / (1 - weight_sum)
        for j in range(k):
            if not np.isfinite(cur_S[j]) or cur_S[j] < 0:
                cur_S[j] = 1
        S[:, i] = S[:, i - 1] * cur_S
    return S


class NWSurv():
    # random state is only for the interface consistency
    def __init__(self, gamma=None, random_state=None):
        self.gamma = 1 if gamma is None else gamma

    def predict(self, x):
        np.seterr(invalid='ignore', divide='ignore')
        W = rbf_kernel(x.astype(np.float64), self.x_train.astype(np.float64), gamma=self.gamma)
        W = W / np.sum(W, axis=-1, keepdims=True)
        W[np.any(np.logical_not(np.isfinite(W)), axis=-1), :] = 1 / self.x_train.shape[0]
        S = nw_helper_func(W.astype(np.float32), self.delta)
        if np.any(np.logical_not(np.isfinite(S))):
            raise ValueError('nan or inf in S')
        return sf_to_t(S, self.T)

    def fit(self, x, y):
        self.T = y['time']
        sort_args = np.argsort(self.T)
        self.T = self.T[sort_args]
        self.x_train = x[sort_args]
        self.delta = y['censor'].astype(np.int64)[sort_args]
        return self

    def get_params(self, deep=False):
        return {'gamma': self.gamma}

    def set_params(self, gamma):
        self.gamma = gamma
        return self


class MyCox(CoxnetSurvivalAnalysis):
    # random state is only for the interface consistency
    def __init__(self, *, n_alphas=1, alphas=None, alpha_min_ratio="auto", l1_ratio=0.75, penalty_factor=None,
                 normalize=False, copy_X=True, tol=1e-5, max_iter=100000, verbose=False, fit_baseline_model=True,
                 random_state=None):
        self.random_state = random_state
        super().__init__(n_alphas=n_alphas, alphas=alphas, alpha_min_ratio=alpha_min_ratio, l1_ratio=l1_ratio, penalty_factor=penalty_factor,
                         normalize=normalize, copy_X=copy_X, tol=tol, max_iter=max_iter, verbose=verbose, fit_baseline_model=fit_baseline_model)

    def fit(self, x, y):
        self.T = np.sort(y['time'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super().fit(x, y)
        return self

    def predict(self, x):
        t = self.T
        baseline_model = self._get_baseline_model(None)
        sf_list = self._predict_survival_function(baseline_model, super().predict(x), False)
        res = np.empty((x.shape[0], t.shape[0]))
        for i, sf in enumerate(sf_list):
            domain = sf.domain
            cur_t = np.clip(t, *domain)
            res[i] = np.clip(sf(cur_t), 0, 1000)
        return sf_to_t(res, t)


class MySurvForest(RandomSurvivalForest):

    def fit(self, x, y):
        self.T = np.sort(y['time'])
        super().fit(x, y)
        return self

    def predict(self, x):
        t = self.T
        sf_list = super().predict_survival_function(x)
        res = np.empty((x.shape[0], t.shape[0]))
        for i, sf in enumerate(sf_list):
            domain = sf.domain
            cur_t = np.clip(t, *domain)
            res[i] = sf(cur_t)
        return sf_to_t(res, t)
