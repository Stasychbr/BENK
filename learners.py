import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from numpy.lib import recfunctions as rfn

N_JOBS = 6


def cens_scorer(Y_true, T_pred):
    T = Y_true['time']
    D = Y_true['censor']
    uncens_idx = D == 1
    if np.max(uncens_idx) == False:
        return float('inf')
    return np.mean((T[uncens_idx] - T_pred[uncens_idx]) ** 2)


class TValLearner():
    def __init__(self, estimator1, estimator2, val_set_c, val_labels_c, val_set_t, val_labels_t, cv_grid, n_splits, random_state=None, do_cv=True) -> None:
        self.estim_control = estimator1
        self.estim_treat = estimator2
        self.val_set_c = val_set_c
        self.val_labels_c = val_labels_c
        self.val_set_t = val_set_t
        self.val_labels_t = val_labels_t
        self.cv_grid = cv_grid
        self.do_cv = do_cv
        if do_cv:
            self.cv_strat = KFold(n_splits, random_state=random_state, shuffle=True)
            self.scorer = make_scorer(cens_scorer, greater_is_better=False)

    def fit(self, X, Y, W):
        control_idx = W == 0
        X_train_control = X[control_idx]
        X_train_control = np.concatenate((X_train_control, self.val_set_c), axis=0)
        Y_train_control = Y[control_idx]
        Y_train_control = rfn.stack_arrays((Y_train_control, self.val_labels_c), usemask=False)
        if self.do_cv:
            grid_search = RandomizedSearchCV(self.estim_control, self.cv_grid,
                                             scoring=self.scorer, refit=True, cv=self.cv_strat, n_jobs=N_JOBS)
            grid_search.fit(X_train_control, Y_train_control)
            self.estim_control = grid_search.best_estimator_
        else:
            self.estim_control.set_params(**self.cv_grid)
            if hasattr(self.estim_control, 'n_jobs'):
                self.estim_control.n_jobs = N_JOBS
            self.estim_control = self.estim_control.fit(X_train_control, Y_train_control)
        treat_idx = W == 1
        X_train_treat = X[treat_idx]
        X_train_treat = np.concatenate((X_train_treat, self.val_set_t), axis=0)
        Y_train_treat = Y[treat_idx]
        Y_train_treat = rfn.stack_arrays((Y_train_treat, self.val_labels_t), usemask=False)
        self.estim_treat.set_params(**self.estim_control.get_params())
        self.estim_treat.fit(X_train_treat, Y_train_treat)
        self.is_fitted = True
        if self.do_cv:
            return grid_search.best_params_

    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X) - self.estim_control.predict(X)

    def predict_control(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_control.predict(X)

    def predict_treat(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X)


class SValLearner():
    def __init__(self, estimator1, val_set_c, val_labels_c, val_set_t, val_labels_t, cv_grid, n_splits, random_state=None, do_cv=True) -> None:
        self.estim = estimator1
        self.val_set_c = np.concatenate((val_set_c, np.zeros((val_set_c.shape[0], 1))), axis=1)
        self.val_labels_c = val_labels_c
        self.val_set_t = np.concatenate((val_set_t, np.ones((val_set_t.shape[0], 1))), axis=1)
        self.val_labels_t = val_labels_t
        self.cv_grid = cv_grid
        self.cv_strat = KFold(n_splits, random_state=random_state, shuffle=True)
        self.scorer = make_scorer(cens_scorer, greater_is_better=False)
        self.do_cv = do_cv

    def fit(self, X, Y, W):
        X_train = np.concatenate((X, W[:, np.newaxis]), axis=1)
        X_train = np.concatenate((X_train, self.val_set_c, self.val_set_t), axis=0)
        Y_train = rfn.stack_arrays((Y, self.val_labels_c, self.val_labels_t), usemask=False)
        if self.do_cv:
            grid_search = RandomizedSearchCV(self.estim, self.cv_grid,
                                             scoring=self.scorer, refit=True, cv=self.cv_strat, n_jobs=N_JOBS)
            grid_search.fit(X_train, Y_train)
            self.estim = grid_search.best_estimator_
        else:
            self.estim.set_params(**self.cv_grid)
            if hasattr(self.estim, 'n_jobs'):
                self.estim.n_jobs = N_JOBS
            self.estim = self.estim.fit(X_train, Y_train)
        self.is_fitted = True
        if self.do_cv:
            return grid_search.best_params_

    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat) - self.estim.predict(X_control)

    def predict_control(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_control)

    def predict_treat(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat)


class XValLearner():
    def __init__(self, estimator1, estimator2, estimator3, estimator4, val_set_c, val_labels_c, val_set_t, val_labels_t, cv_grid_base, cv_grid_reg, n_splits, random_state=None, do_cv=True) -> None:
        self.mu_0 = estimator1
        self.mu_1 = estimator2
        self.tau_0 = estimator3
        self.tau_1 = estimator4
        self.val_set_c = val_set_c
        self.val_labels_c = val_labels_c
        self.val_set_t = val_set_t
        self.val_labels_t = val_labels_t
        self.cv_grid_base = cv_grid_base
        self.cv_grid_reg = cv_grid_reg
        self.cv_strat_base = KFold(n_splits, random_state=random_state, shuffle=True)
        self.cv_strat_reg = KFold(n_splits, random_state=random_state, shuffle=True)
        self.scorer_base = make_scorer(cens_scorer, greater_is_better=False)
        self.scorer_reg = make_scorer(mean_squared_error, greater_is_better=False)
        self.do_cv = do_cv

    def fit(self, X, Y, W):
        self.g = np.sum(W) / W.shape[0]

        cnt_idx = W == 0
        X_0, Y_0 = X[cnt_idx], Y[cnt_idx]
        X_0 = np.concatenate((X_0, self.val_set_c), axis=0)
        Y_0 = rfn.stack_arrays((Y_0, self.val_labels_c), usemask=False)
        tr_idx = W == 1
        X_1, Y_1 = X[tr_idx], Y[tr_idx]
        X_1 = np.concatenate((X_1, self.val_set_t), axis=0)
        Y_1 = rfn.stack_arrays((Y_1, self.val_labels_t), usemask=False)

        if self.do_cv:
            grid_search_base = RandomizedSearchCV(self.mu_0, self.cv_grid_base, scoring=self.scorer_base,
                                                  refit=True, cv=self.cv_strat_base, n_jobs=N_JOBS)
            grid_search_base.fit(X_0, Y_0)
            self.mu_0 = grid_search_base.best_estimator_
        else:
            self.mu_0.set_params(**self.cv_grid_base)
            if hasattr(self.mu_0, 'n_jobs'):
                self.mu_0.n_jobs = N_JOBS
            self.mu_0 = self.mu_0.fit(X_0, Y_0)
        self.mu_1.set_params(**self.mu_0.get_params())
        self.mu_1.fit(X_1, Y_1)

        D_0 = self.mu_1.predict(X_0) - Y_0['time']
        D_1 = Y_1['time'] - self.mu_0.predict(X_1)

        if self.do_cv:
            grid_search_reg = RandomizedSearchCV(self.tau_0, self.cv_grid_reg, scoring=self.scorer_reg,
                                                 refit=True, cv=self.cv_strat_reg, n_jobs=N_JOBS)
            grid_search_reg.fit(X_0, D_0)
            self.tau_0 = grid_search_reg.best_estimator_
        else:
            self.tau_0.set_params(**self.cv_grid_reg)
            if hasattr(self.tau_0, 'n_jobs'):
                self.tau_0.n_jobs = N_JOBS
            self.tau_0 = self.tau_0.fit(X_0, D_0)
        self.tau_1.set_params(**self.tau_0.get_params())
        self.tau_1.fit(X_1, D_1)

        self.is_fitted = True
        if self.do_cv:
            return grid_search_base.best_params_, grid_search_reg.best_params_

    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.g * self.tau_0.predict(X) + (1 - self.g) * self.tau_1.predict(X)


def make_t_learner(val_set_c, val_labels_c, base_cls, cv_grid, n_splits, random_state, do_cv):
    val_set_t = np.empty((0, val_set_c.shape[1]))
    val_labels_t = np.ndarray(shape=(0), dtype=[('censor', '?'), ('time', 'f4')])
    return TValLearner(base_cls(random_state=random_state), base_cls(random_state=random_state),
                       val_set_c, val_labels_c, val_set_t, val_labels_t, cv_grid, n_splits, random_state, do_cv)


def make_s_learner(val_set_c, val_labels_c, base_cls, cv_grid, n_splits, random_state, do_cv):
    val_set_t = np.empty((0, val_set_c.shape[1]))
    val_labels_t = np.ndarray(shape=(0), dtype=[('censor', '?'), ('time', 'f4')])
    return SValLearner(base_cls(random_state=random_state), val_set_c, val_labels_c,
                       val_set_t, val_labels_t, cv_grid, n_splits, random_state, do_cv)


def make_x_learner(val_set_c, val_labels_c, base_cls, reg_cls, cv_grid_base, cv_grid_reg, n_splits, random_state, do_cv):
    val_set_t = np.empty((0, val_set_c.shape[1]))
    val_labels_t = np.ndarray(shape=(0), dtype=[('censor', '?'), ('time', 'f4')])
    return XValLearner(base_cls(random_state=random_state), base_cls(random_state=random_state), reg_cls(random_state=random_state),
                       reg_cls(random_state=random_state), val_set_c, val_labels_c, val_set_t, val_labels_t,
                       cv_grid_base, cv_grid_reg, n_splits, random_state, do_cv)
