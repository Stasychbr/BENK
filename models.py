import numpy as np
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sklearn.metrics.pairwise import rbf_kernel
from itertools import product
import pandas as pd

def sf_to_t(S, T):
    t_diff = T[1:] - T[:-1]
    integral = S[:, :-1] * t_diff[None, :]
    # assert(integral.ndim == 1 or integral.ndim == 2 and integral.shape[1] == 1)
    return T[0] + np.sum(integral, axis=-1)

class NWSurv():
    def __init__(self, gamma):
        self.gamma_list = gamma
        self.gamma = None

    def predict(self, x, t):
        n = self.x_train.shape[0]
        W = rbf_kernel(x, self.x_train, gamma=self.gamma)
        W = W / np.sum(W, axis=-1)[:, None]
        S = 1
        for i in range(n):
            if self.delta[i] == 0:
                continue
            cur_S = 1 - W[:, i] / (1 - np.sum(W * (self.T[None, :] < self.T[i, None]), axis=1))
            cur_S[np.logical_not(np.isfinite(cur_S))] = 0
            cur_S = np.tile(cur_S[:, None], (1, t.shape[0]))
            cur_S[np.tile((self.T[i] > t)[None, :], (x.shape[0], 1))] = 1
            S *= cur_S
        if np.any(np.logical_not(np.isreal(S))):
            raise ValueError('nan or inf in S')
        return sf_to_t(S, t)
        
    def fit(self, x, t, delta):
        assert(t.shape[0] == delta.shape[0])
        self.x_train = x
        self.T = np.sort(t)
        self.delta = delta
        if self.gamma is not None:
            return self
        if all(map(lambda o: o is not None, (self.val_x, self.val_T, self.val_delta))):
            min_loss = float('inf')
            for g in self.gamma_list:
                self.gamma = g
                pred = self.predict(self.val_x, self.T)
                cur_loss = np.mean((pred - self.val_T) ** 2)
                if cur_loss <= min_loss:
                    min_loss = cur_loss
                    best_gamma = g
            self.gamma = best_gamma
        else:
            self.gamma = 1
        return self
    
    def set_val(self, val_x, val_t, val_delta):
        self.val_x = val_x
        self.val_T = val_t
        self.val_delta = val_delta
    
    def get_params(self):
        return {'gamma': self.gamma}
    
    def set_params(self, params):
        self.gamma = params['gamma']

class MyCox():
    def __init__(self, alpha):
        # self.cox = CoxPHSurvivalAnalysis(n_iter=200)
        # self.n_iter = 200
        self.pen_list = alpha
        self.penalizer = None
        self.val_df = None
        self.params = None
    
    def fit(self, x, t, delta):
        assert(t.shape[0] == delta.shape[0])
        df_fit = pd.DataFrame(x)
        df_fit = df_fit.assign(**{'event': delta.astype(int), 'time': t})
        if self.params is not None:
            self.cox = CoxPHFitter(**self.params)
            self.cox.fit(df_fit, 'time', 'event')
            return self
        if self.val_df is not None:
            max_C_ind = 0
            for c in self.pen_list:
                cox = CoxPHFitter(penalizer=c)
                cox.fit(df_fit, 'time', 'event')
                cur_C_ind = cox.score(self.val_df,  'concordance_index')
                if cur_C_ind >= max_C_ind:
                    max_C_ind = cur_C_ind
                    self.cox = cox
                    self.penalizer = c
        else:
            self.cox = CoxPHFitter()
            self.cox.fit(df_fit, 'time', 'event')
        return self

    def get_params(self):
        return {'penalizer': self.penalizer}
    
    def set_params(self, params):
        self.params = params

    def set_val(self, val_x, val_t, val_delta):
        val_df = pd.DataFrame(val_x)
        self.val_df = val_df.assign(**{'event': val_delta.astype(int), 'time': val_t})

    def predict(self, x, t):
        # sf_frame = self.cox.predict_survival_function(x, t)
        # res = [sf(t) for sf in sf_list]
        return self.cox.predict_expectation(x).to_numpy()

class MySurvForest():
    def __init__(self, trees, depth, leaf_samples, max_features = None):
        self.trees = trees
        self.depth = depth
        self.leaf_samples = leaf_samples
        self.max_features = max_features
        self.val_x = None
        self.val_str_array = None
        self.params = None
        self.n_jobs = 8
    
    def fit(self, x, t, delta):
        assert(t.shape[0] == delta.shape[0])
        str_array = np.ndarray(shape=(t.shape[0]), dtype=[('censor', '?'), ('time', 'f4')])
        str_array['censor'] = delta.astype(bool)
        str_array['time'] = t
        if self.params is not None:
            self.forest = RandomSurvivalForest(**self.params)
            self.forest.fit(x, str_array)
            return self
        if self.val_x is not None and self.val_str_array is not None:
            max_C_ind = 0
            for n_estimators, max_depth, min_samples_leaf in product(self.trees, self.depth, self.leaf_samples):
                forest = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=self.max_features, n_jobs=self.n_jobs)
                forest.fit(x, str_array)
                cur_C_ind = forest.score(self.val_x, self.val_str_array)
                if cur_C_ind >= max_C_ind:
                    max_C_ind = cur_C_ind
                    self.forest = forest
                    self.n_estims = n_estimators
                    self.max_depth = max_depth
                    self.min_samples_leaf = min_samples_leaf
        else:
            # print('No validation set was provided!')
            self.forest = RandomSurvivalForest()
            self.forest.fit(x, str_array)
    
        self.forest.fit(x, str_array)
        return self

    def get_params(self):
        return {'n_estimators': self.n_estims, 'max_depth': self.max_depth, 'min_samples_leaf': self.min_samples_leaf, 'max_features': self.max_features, 'n_jobs': self.n_jobs}
    
    def set_params(self, params):
        self.params = params

    def set_val(self, val_x, val_t, val_delta):
        self.val_x = val_x
        self.val_str_array = np.ndarray(shape=(val_t.shape[0]), dtype=[('censor', '?'), ('time', 'f4')])
        self.val_str_array['censor'] = val_delta.astype(bool)
        self.val_str_array['time'] = val_t

    def predict(self, x, t):
        sf_list = self.forest.predict_survival_function(x)
        res = [sf(t) for sf in sf_list]
        return sf_to_t(np.array(res), t)