import numpy as np

def response_func(x, b, l, u, nu):
    return np.power(-np.log(u) / (l * np.exp(b * x)), 1 / nu)

class pow_func():
    def __init__(self, m, b, l, u, nu) -> None:
        self.m = m
        self.b = b
        self.l = l
        self.u = u
        self.nu = nu
    
    def calc_x(self, t):
        a = 1 / np.sqrt(self.m)
        pows = [a * (i + 1) for i in range(self.m)]
        X = np.empty((t.shape[0], self.m))
        for i, p in enumerate(pows):
            if 0.8 < p < 1.6:
                X[:, i] = np.random.normal(0, 1, t.shape[0])
            else:
                X[:, i] = np.power(t, a * (i + 1))
        return X

    def calc_y(self, t):
        return response_func(t, self.b, self.l, self.u, self.nu)


class log_func():
    def __init__(self, m, coeffs_neg, coeffs_pos, b, l, u, nu) -> None:
        self.log_coeffs = np.empty(m)
        self.m = m
        neg_num = m // 2
        self.log_coeffs[:neg_num] = np.random.uniform(coeffs_neg[0], coeffs_neg[1], neg_num)
        pos_num = m - neg_num
        self.log_coeffs[neg_num:neg_num + pos_num] = np.random.uniform(coeffs_pos[0], coeffs_pos[1], pos_num)
        rng = np.random.default_rng()
        rng.shuffle(self.log_coeffs)
        self.b = b
        self.l = l
        self.u = u
        self.nu = nu

    def calc_x(self, t):
        res = np.empty((t.shape[0], self.m))
        for i in range(self.m):
            res[:, i] = self.log_coeffs[i] * np.log(t)
        return res

    def calc_y(self, t):
        return response_func(t, self.b, self.l, self.u, self.nu)


class spiral_func():
    def __init__(self, m, b, l, u, nu):
        self.m = m
        self.b = b
        self.l = l
        self.u = u
        self.nu = nu
    
    def calc_x(self, t):
        X = np.empty((t.shape[0], self.m), dtype=np.float32)
        for i in range(self.m):
            if i % 2 == 0:
                X[:, i] = t * np.sin((i // 2 + 1) * t)
            else:
                X[:, i] = t * np.cos((i // 2 + 1) * t)
        return X

    def calc_y(self, t):
        return response_func(t, self.b, self.l, self.u, self.nu)