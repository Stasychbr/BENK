import torch
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import numba
from torch.utils.data import Dataset, DataLoader
from inspect import getargs
import time
import copy


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')


class Kernel(Module):
    def __init__(self, m) -> None:
        super().__init__()
        s = int(np.sqrt(m))
        self.sparse = torch.nn.Sequential(
            torch.nn.Linear(m, 2 * m),
            torch.nn.ReLU6()
        )
        self.sequent = torch.nn.Sequential(
            torch.nn.Linear(2 * m, m),
            torch.nn.ReLU6(),
            torch.nn.Linear(m, 2 * s),
            torch.nn.ReLU6(),
            torch.nn.Linear(2 * s, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1),
            torch.nn.Softplus()
        )

    def forward(self, x_1, x_2):
        sparse_1 = self.sparse(x_1)
        sparse_2 = self.sparse(x_2)
        total_in = torch.abs(sparse_1 - sparse_2)
        return self.sequent(total_in)


class BENK(Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.kernel = Kernel(m)

    def forward(self, x_in, T_in, delta_in, x_p):
        n = x_in.shape[1]
        x_p_repeat = x_p[:, None, :].repeat(1, n, 1)
        W = torch.reshape(self.kernel(x_in, x_p_repeat), (-1, n))
        W = W / torch.sum(W, dim=1, keepdim=True)
        sorted_T, sorted_args = torch.sort(T_in, 1)
        delta_sorted = torch.gather(delta_in, 1, sorted_args)
        W = torch.gather(W, 1, sorted_args)
        w_cumsum = torch.cumsum(W, dim=1)
        shifted_w_cumsum = w_cumsum - W
        ones = torch.ones_like(shifted_w_cumsum)
        bad_idx = torch.isclose(shifted_w_cumsum, ones) | torch.isclose(w_cumsum, ones)
        shifted_w_cumsum[bad_idx] = 0.0
        w_cumsum[bad_idx] = 0.0

        xi = torch.log(1.0 - shifted_w_cumsum)
        xi -= torch.log(1.0 - w_cumsum)

        filtered_xi = delta_sorted * xi
        hazards = torch.cumsum(filtered_xi, dim=1)
        surv = torch.exp(-hazards)
        output = torch.stack((surv, sorted_T), dim=2)
        return output

    def forward_in_points(self, x_in, T_in, delta_in, x_p, t_p):
        n = x_in.shape[1]
        k = t_p.shape[1]

        x_p_repeat = x_p[:, None, :].repeat(1, n, 1)
        W = torch.reshape(self.kernel(x_in, x_p_repeat), (-1, n))
        W = W / torch.sum(W, dim=1, keepdim=True)
        S = 1
        # eps = 10 ** -6
        for i in range(n):
            weight_sum = torch.sum(W * (T_in < T_in[:, i, None]), dim=1)
            bad_idx = weight_sum >= 1
            weight_sum[bad_idx] = 0
            cur_S = 1 - W[:, i] / (1 - weight_sum)
            cur_S[bad_idx] = 0
            cur_S = cur_S[:, None].repeat(1, k)
            cur_S[(T_in[:, None, i] > t_p) | (delta_in[:, None, i] == 0)] = 1
            S *= cur_S
        if torch.any(torch.logical_not(torch.isfinite(S))):
            raise ValueError('nan or inf in S')
        return S

    def predict_in_points(self, *args):
        self.eval()
        with torch.no_grad():
            return self.forward_in_points(*args).cpu().numpy()

    def predict(self, *args):
        self.eval()
        with torch.no_grad():
            return self(*args).cpu().numpy()


class BENKDataset(Dataset):
    def __init__(self, x_in, T_in, delta_in, x_p, t_p, d_p) -> None:
        super().__init__()
        args, _, _ = getargs(self.__init__.__code__)
        for arg in args[1:]:
            setattr(self, arg, torch.from_numpy(locals()[arg]).to(device))

    def __getitem__(self, index):
        return ((self.x_in[index], self.T_in[index], self.delta_in[index], self.x_p[index]), self.t_p[index], self.d_p[index])

    def __len__(self):
        return self.x_in.shape[0]


@numba.njit
def make_spec_set(x_in, x_p, T_in, T_p, delta_in, delta_p, n, m, mlp_coef):
    idx = np.arange(x_in.shape[0])
    x_in_out = np.zeros((x_p.shape[0], mlp_coef, n, m), dtype=np.float32)
    T_in_out = np.zeros((x_p.shape[0], mlp_coef, n), dtype=np.float32)
    x_p_out = np.zeros((x_p.shape[0], mlp_coef, m), dtype=np.float32)
    t_p_out = np.zeros((x_p.shape[0], mlp_coef), dtype=np.float32)
    labels = np.zeros((x_p.shape[0], mlp_coef), dtype=np.float32)
    delta_in_out = np.zeros((x_p.shape[0], mlp_coef, n), dtype=np.float32)
    for i in range(x_p.shape[0]):
        for j in range(mlp_coef):
            x_p_out[i, j, :] = x_p[i]
        t_p_out[i, :] = T_p[i]
        labels[i, :] = delta_p[i]
        if n == x_in.shape[0]:
            for j in range(mlp_coef):
                x_in_out[i, j, ...] = x_in
                T_in_out[i, j, :] = T_in
                delta_in_out[i, j, :] = delta_in
        else:
            for j in range(mlp_coef):
                cur_idx = np.random.choice(idx, n, False)
                x_in_out[i, j, ...] = x_in[cur_idx]
                T_in_out[i, j, :] = T_in[cur_idx]
                delta_in_out[i, j, :] = delta_in[cur_idx]
    x_in_out = np.reshape(x_in_out, (-1, n, m))
    T_in_out = np.reshape(T_in_out, (-1, n))
    x_p_out = np.reshape(x_p_out, (-1, m))
    t_p_out = np.reshape(t_p_out, (-1, 1))
    delta_in_out = np.reshape(delta_in_out, (-1, n))
    labels = np.reshape(labels, (-1, 1))
    return x_in_out, T_in_out, delta_in_out, x_p_out, t_p_out, labels


@numba.njit
def make_train_set(x, T, delta, n, m, mlp_coef):
    idx = np.asarray(list(range(1, x.shape[0])))
    uncens_num = np.count_nonzero(delta == 1)
    x_in = np.zeros((mlp_coef, uncens_num, n, m), dtype=np.float32)
    T_in = np.zeros((mlp_coef, uncens_num, n), dtype=np.float32)
    delta_in = np.zeros((mlp_coef, uncens_num, n), dtype=np.float32)
    x_p = np.zeros((mlp_coef, uncens_num, m), dtype=np.float32)
    t_p = np.zeros((mlp_coef, uncens_num), dtype=np.float32)
    delta_labels = np.zeros((mlp_coef, uncens_num), dtype=np.float32)
    i = 0
    for q in range(x.shape[0]):
        if delta[q] == 1:
            for j in range(mlp_coef):
                x_p[j, i, :] = x[q]
            t_p[:, i] = T[q]
            delta_labels[:, i] = delta[q]
            for j in range(mlp_coef):
                cur_idx = np.random.choice(idx, n, False)
                x_in[j, i, ...] = x[cur_idx]
                T_in[j, i, :] = T[cur_idx]
                delta_in[j, i, :] = delta[cur_idx]
            i += 1
        if q < len(idx):
            idx[q] = q
    x_in = np.reshape(x_in, (-1, n, m))
    T_in = np.reshape(T_in, (-1, n))
    delta_in = np.reshape(delta_in, (-1, n))
    x_p = np.reshape(x_p, (-1, m))
    t_p = np.reshape(t_p, (-1, 1))
    delta_labels = np.reshape(delta_labels, (-1, 1))
    return x_in, T_in, delta_in, x_p, t_p, delta_labels


def train_model(data_generator, model, loss_fn, optimizer, epochs, val_data=None, patience=0):
    start_time = time.time()
    if val_data is not None:
        weights = copy.deepcopy(model.state_dict())
        cur_patience = 0

        def get_val_loss():
            with torch.no_grad():
                val_pred = model(*val_data[0])
                val_loss = loss_fn(val_pred, val_data[1]).item()
            return val_loss

        best_val_loss = get_val_loss()

    model.train()
    for e in range(1, epochs + 1):
        cur_loss = 0
        i = 0
        print(f'----- Epoch {e} -----')
        time_stamp = time.time()
        dataloader = data_generator.get_data_loader()
        steps = len(dataloader)
        for data, t_labels, d_labels in dataloader:
            optimizer.zero_grad()

            pred = model(*data)
            d_mask = (d_labels == 1).ravel()
            if d_mask.max() == False:
                continue
            loss = loss_fn(pred[d_mask], t_labels[d_mask])

            loss.backward()

            for p in model.parameters():
                p.grad[torch.logical_not(torch.isfinite(p.grad))] = 0
            optimizer.step()

            cur_loss += loss.item()
            i += 1

            print(f'Loss: {round(cur_loss / i, 5)}, step {i}/{steps}', end='        \r')

        print()
        if val_data is not None:
            cur_patience += 1
            val_loss = get_val_loss()
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                weights = copy.deepcopy(model.state_dict())
                cur_patience = 0
            print(f'val loss: {round(val_loss, 5)}, patience: {cur_patience}')
            if cur_patience >= patience:
                print('Early stopping!')
                model.load_state_dict(weights)
                break
        print('time elapsed: ', round(time.time() - time_stamp, 4), ' sec.')
    print(f'Train is finished, {round(time.time() - start_time, 0)} sec. taken')


def tau_loss(S, T_labels):
    SF = S[..., 0]
    T = S[..., 1]
    T_diff = T[:, 1:] - T[:, :-1]
    integral = T[:, 0, None] + torch.sum(SF[:, :-1] * T_diff, dim=-1, keepdim=True)
    return F.mse_loss(integral, T_labels)


class BENKDataGenerator():
    def __init__(self, x, T, delta, batch, n, mlp_coef, f_shuffle=False) -> None:
        self.x, self.T, self.delta = x, T, delta
        self.n = n
        self.mlp_coef = mlp_coef
        self.batch = batch
        self.f_shuffle = f_shuffle

    def get_data_loader(self):
        *data, labels, delta_labels = make_train_set(self.x, self.T,
                                                     self.delta, self.n, self.x.shape[1], self.mlp_coef)
        return DataLoader(BENKDataset(*data, labels, delta_labels), self.batch, shuffle=self.f_shuffle)
