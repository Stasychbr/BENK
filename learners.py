import numpy as np

class TValLearner():
    # val_set is (x, T, delta)
    def __init__(self, estimator1, estimator2, val_set_c, val_set_t) -> None:
        self.estim_control = estimator1
        self.estim_treat = estimator2
        self.val_set_c = val_set_c
        self.val_set_t = val_set_t

    def fit(self, X, T, delta, W):
        control_idx = W == 0
        X_train_control = X[control_idx]
        T_train_control = T[control_idx]
        delta_train_control = delta[control_idx]
        self.estim_control.set_val(*self.val_set_c)
        self.estim_control.fit(X_train_control, T_train_control, delta_train_control)
        treat_idx = W == 1
        X_train_treat = X[treat_idx]
        T_train_treat = T[treat_idx]
        delta_train_treat = delta[treat_idx]
        if self.val_set_t is None:
            self.estim_treat.set_params(self.estim_control.get_params())
        else:
            self.estim_treat.set_val(*self.val_set_t)
        self.estim_treat.fit(X_train_treat, T_train_treat, delta_train_treat)
        self.is_fitted = True
    
    def predict(self, X, T):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X, T) - self.estim_control.predict(X, T)

    def predict_control(self, X, T):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_control.predict(X, T)

    def predict_treat(self, X, T):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X, T)

class SValLearner():
    def __init__(self, estimator, val_set_c, val_set_t) -> None:
        self.estim = estimator
        val_x = np.concatenate((val_set_c[0], np.zeros((val_set_c[0].shape[0], 1))), axis=1)
        self.val_set_c = (val_x, val_set_c[1], val_set_c[2])
        # self.val_set_t = np.concatenate((val_set_t, np.ones((val_set_t.shape[0], 1))), axis=1)
        # self.val_labels_t = val_labels_t

    def fit(self, X, T, delta, W):
        X_train = np.concatenate((X, W[:, np.newaxis]), axis=1)
        self.estim.set_val(*self.val_set_c)
        self.estim.fit(X_train, T, delta)
        self.is_fitted = True
    
    def predict(self, X, T):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat, T) - self.estim.predict(X_control, T)

    def predict_control(self, X, T):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_control, T)

    def predict_treat(self, X, T):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat, T)

class XValLearner():
    def __init__(self, estimator1, estimator2, estimator3, estimator4, val_set_c, val_set_t) -> None:
        self.mu_0 = estimator1
        self.mu_1 = estimator2
        self.tau_0 = estimator3
        self.tau_1 = estimator4
        self.val_set_c = val_set_c
        self.val_set_t = val_set_t

    def fit(self, X, T, delta, W):
        self.g = np.sum(W) / W.shape[0]
        
        cnt_idx = W == 0
        X_0, T_0, d_0 = X[cnt_idx], T[cnt_idx], delta[cnt_idx]
        tr_idx = W == 1
        X_1, T_1, d_1 = X[tr_idx], T[tr_idx], delta[tr_idx]
        
        self.mu_0.set_val(*self.val_set_c)
        self.mu_0.fit(X_0, T_0, d_0)
        # self.mu_1.set_val(self.val_set_t, self.val_labels_t)
        self.mu_1.set_params(self.mu_0.get_params())
        self.mu_1.fit(X_1, T_1, d_1)
        
        T_pred = np.sort(T)
        D_0 = self.mu_1.predict(X_0, T_pred) - T_0.ravel()
        # T_1_sorted = np.sort(T_1)
        D_1 = T_1.ravel() - self.mu_0.predict(X_1, T_pred)
        
        # self.tau_0.set_val(self.val_set_c, self.mu_1.predict(self.val_set_c) - self.val_labels_c)
        self.tau_0.set_params(self.mu_0.get_params())
        self.tau_0.fit(X_0, D_0, d_0)
        # self.tau_1.set_val(self.val_set_t, self.mu_0.predict(self.val_set_t) - self.val_labels_t)
        self.tau_1.set_params(self.mu_0.get_params())
        self.tau_1.fit(X_1, D_1, d_1)

        self.is_fitted = True

    def predict(self, X, T):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.g * self.tau_0.predict(X, T) + (1 - self.g) * self.tau_1.predict(X, T)