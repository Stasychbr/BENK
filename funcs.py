import numpy as np

class pow_func():
    def __init__(self, m) -> None:
        self.m = m
    
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


class log_func():
    def __init__(self, m, coeffs_neg, coeffs_pos) -> None:
        self.log_coeffs = np.empty(m)
        self.m = m
        neg_num = m // 2
        self.log_coeffs[:neg_num] = np.random.uniform(coeffs_neg[0], coeffs_neg[1], neg_num)
        pos_num = m - neg_num
        self.log_coeffs[neg_num:neg_num + pos_num] = np.random.uniform(coeffs_pos[0], coeffs_pos[1], pos_num)
        rng = np.random.default_rng()
        rng.shuffle(self.log_coeffs)

    def calc_x(self, t):
        res = np.empty((t.shape[0], self.m))
        for i in range(self.m):
            res[:, i] = self.log_coeffs[i] * np.log(t)
        return res


class spiral_func():
    def __init__(self, m):
        self.m = m
    
    def calc_x(self, t):
        X = np.empty((t.shape[0], self.m), dtype=np.float32)
        for i in range(self.m):
            if i % 2 == 0:
                X[:, i] = t * np.sin((i // 2 + 1) * t)
            else:
                X[:, i] = t * np.cos((i // 2 + 1) * t)
        return X