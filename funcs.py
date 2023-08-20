from abc import ABC
import numpy as np
from math import gamma


class func(ABC):
    def __init__(self, b, l, nu) -> None:
        super().__init__()
        self.b, self.l, self.nu = b, l, nu
        self.gamma = gamma(1 / nu + 1)

    def calc_T(self, tau):
        return np.power(
            -np.log(np.random.uniform(0, 1, tau.shape[0])) /
            (self.l * np.exp(self.b * tau)), 1 / self.nu)

    def calc_expected(self, tau):
        return np.power(self.l * np.exp(self.b * tau), -1 / self.nu) * self.gamma


class circle_func(func):
    def __init__(self, m, t_bnd, b, l, nu) -> None:
        assert m % 2 == 0
        super().__init__(b, l, nu)
        self.m = m
        circles_n = m // 2
        t_range = t_bnd[1] - t_bnd[0]
        self.split = [(i + 1) * t_range / circles_n for i in range(circles_n)]
        self.split[-1] = t_bnd[1]

    def calc_x(self, tau):
        X = np.empty((tau.shape[0], self.m))
        last_border = 0
        for i, cur_border in enumerate(self.split):
            cur_range = cur_border - last_border
            scale = 2 * np.pi / cur_range
            shift = -last_border
            t_mask = np.logical_and(last_border < tau, tau < cur_border).astype(np.float32)
            X[:, 2 * i] = np.sin((shift + tau) * scale) * t_mask
            X[:, 2 * i + 1] = np.cos((shift + tau) * scale) * t_mask
            last_border = cur_border
        return X


class gauss_func(func):
    def __init__(self, m, t_bnd, b, l, nu) -> None:
        super().__init__(b, l, nu)
        self.m = m
        self.sigma = (t_bnd[1] - t_bnd[0]) / (6 * m)

    def calc_x(self, tau):
        res = np.empty((tau.shape[0], self.m))
        for i in range(self.m):
            res[:, i] = 1 / (self.sigma * np.sqrt(2 * np.pi)) * \
                np.exp(- (tau - i) ** 2 / (2 * self.sigma ** 2))
        return res


class spiral_func(func):
    def __init__(self, m, b, l, nu):
        super().__init__(b, l, nu)
        self.m = m
        self.l = l
        self.nu = nu

    def calc_x(self, tau):
        X = np.empty((tau.shape[0], self.m), dtype=np.float32)
        for i in range(self.m):
            if i % 2 == 0:
                X[:, i] = tau * np.sin((i // 2 + 1) * tau)
            else:
                X[:, i] = tau * np.cos((i // 2 + 1) * tau)
        return X
