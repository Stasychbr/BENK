import torch
import numpy as np
import time
from collections import defaultdict
import pickle
from BENK import TNWDataGenerator, TNW, train_model, device, make_spec_set, make_train_set, tau_loss
from learners import TValLearner, SValLearner, XValLearner
from models import NWSurv, MyCox, MySurvForest
from funcs import log_func, pow_func, spiral_func

np.seterr(all='ignore')

def tau_metric(S, T, T_labels):
    assert S.ndim == 2
    if T.ndim == 1:
        T_diff = T[None, 1:] - T[None, :-1]
        T_0 = T[0]
    else:
        T_diff = T[:, 1:] - T[:, :-1]
        T_0 = T[:, 0]
    integral = T_0 + np.sum(S[:, :-1] * T_diff, axis=-1)
    return np.sqrt(np.mean((integral - T_labels.ravel()) ** 2))

def tau_cate_diff(S_diff, T, T_1, T_0):
    # assert S_diff.ndim  == T_1.ndim == T_0.ndim == 2
    if T.ndim == 1:
        T_diff = T[None, 1:] - T[None, :-1]
        T_start = T[0]
    else:
        T_diff = T[:, 1:] - T[:, :-1]
        T_start = T[:, 0]
    integral = T_start + np.sum(S_diff[:, :-1] * T_diff, axis=-1)
    return np.sqrt(np.mean((integral - (T_1.ravel() - T_0.ravel())) ** 2))
    
def weibull_distr(x, b, l, u, nu):
    return np.power(-np.log(u) / (l * np.exp(b * x)), 1 / nu)

def get_data_from_param_func(func, t, noise_percent, censored_part=0.6, treat = True):
    x = func.calc_x(t)
    if treat:
        T = np.float32(weibull_distr(t, b_trt, l_trt, u_trt, nu_trt))
    else:
        T = np.float32(weibull_distr(t, b_cnt, l_cnt, u_cnt, nu_cnt))
    sigma = np.mean(T) * noise_percent / 3
    T += np.random.normal(0, sigma, t.shape[0])
    T[T > 2000] = 2000
    delta = np.float32(np.random.binomial(1, 1 - censored_part, size=t.shape[0]))
    x = (x - x.mean(axis=0)[None, :]) / x.std(axis=0)[None, :]
    return x, T, delta

def make_treat_set(cnt_func, trt_func, noise_percent, t_bounds, censored_part, cnt_size, trt_size, test_size, val_size):
    t_cnt = np.random.uniform(t_bounds[0], t_bounds[1], cnt_size)
    x_cnt, T_cnt, delta_cnt = get_data_from_param_func(cnt_func, t_cnt, noise_percent, censored_part, False)
    t_trt = np.random.uniform(t_bounds[0], t_bounds[1], trt_size)
    x_trt, T_trt, delta_trt = get_data_from_param_func(trt_func, t_trt, noise_percent, censored_part, True)
    t_test = np.random.uniform(t_bounds[0], t_bounds[1], test_size)
    x_test, T_test_trt, delta_test = get_data_from_param_func(trt_func, t_test, noise_percent, censored_part, True)
    _, T_test_cnt, _ = get_data_from_param_func(cnt_func, t_test, noise_percent, censored_part, False)
    t_val = np.random.uniform(t_bounds[0], t_bounds[1], val_size)
    x_val, T_val, delta_val = get_data_from_param_func(cnt_func, t_val, noise_percent, censored_part, False)
    res_dict = dict()
    res_dict['x_cnt'], res_dict['T_cnt'], res_dict['delta_cnt'] = x_cnt, T_cnt, delta_cnt
    res_dict['x_trt'], res_dict['T_trt'], res_dict['delta_trt'] = x_trt, T_trt, delta_trt
    res_dict['x_test'], res_dict['T_test_cnt'], res_dict['T_test_trt'], res_dict['delta_test'] = x_test, T_test_cnt, T_test_trt, delta_test
    res_dict['x_val'], res_dict['T_val'], res_dict['delta_val'] = x_val, T_val, delta_val
    return res_dict

def arrs_to_torch_dev(*args):
    return [torch.from_numpy(a).to(device) for a in args]


def rmse(x, y):
    return np.sqrt(np.mean((x.ravel() - y.ravel()) ** 2))

def exp_iter(data_dict):
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_cnt, T_cnt, delta_cnt = data_dict['x_cnt'], data_dict['T_cnt'], data_dict['delta_cnt']
    x_trt, T_trt, delta_trt = data_dict['x_trt'], data_dict['T_trt'], data_dict['delta_trt']
    x_test, T_test_cnt, T_test_trt, delta_test = data_dict['x_test'], data_dict['T_test_cnt'], data_dict['T_test_trt'], data_dict['delta_test']
    x_val, T_val, delta_val = data_dict['x_val'], data_dict['T_val'], data_dict['delta_val']
    model = TNW(m).to(device)
    data_gen = TNWDataGenerator(x_cnt, T_cnt, delta_cnt, batch_size, n, mlp_coef)
    x_in_t_c, cnt_times, delta_in_t_c, x_p_t_c, cnt_labels, cnt_d = make_spec_set(x_cnt, x_test, T_cnt, T_test_cnt, delta_cnt, delta_test, x_cnt.shape[0], m, 1)
    x_in_t_c, T_in_t_c, delta_in_t_c, x_p_t_c = arrs_to_torch_dev(x_in_t_c, cnt_times, delta_in_t_c, x_p_t_c)
    x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t, trt_labels, trt_d = make_spec_set(x_trt, x_test, T_trt, T_test_trt, delta_trt, delta_test, x_trt.shape[0], m, 1)
    x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t = arrs_to_torch_dev(x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t)

    predict_T = np.sort(np.concatenate((T_cnt, T_trt)))
    predict_T_torch = arrs_to_torch_dev(predict_T)[0][None, :].repeat(x_in_t_c.shape[0], 1)
    def calc_my_model_metrics(keys):
        S_0 = model.predict_in_points(x_in_t_c, T_in_t_c, delta_in_t_c, x_p_t_c, predict_T_torch)
        res_dict[keys[0]].append(tau_metric(S_0, predict_T, cnt_labels))
        S_1 = model.predict_in_points(x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t, predict_T_torch)
        res_dict[keys[1]].append(tau_metric(S_1, predict_T, trt_labels))
        res_dict[keys[2]].append(tau_cate_diff(S_1 - S_0, predict_T, trt_labels, cnt_labels))

    *val_data, val_labels, val_d = make_spec_set(x_cnt, x_val, T_cnt, T_val, delta_cnt, delta_val, x_cnt.shape[0], m, 1)
    *val_data, val_labels, val_d = arrs_to_torch_dev(*val_data, val_labels, val_d)
    # optimizer = torch.optim.Adam(model.parameters(), 0.005)
    optimizer = torch.optim.Adagrad(model.parameters(), 0.01)
    train_model(data_gen, model, tau_loss, optimizer, epochs, (val_data, val_labels, val_d), patience)
    calc_my_model_metrics(('Kernel control', 'Kernel treat', 'Kernel CATE'))
    trees = [50, 100, 200]
    depth = [2, 4, 6]
    leaf_samples = [1, 0.1, 0.2]
    alpha_cox = [0.1, 0.5, 1, 2, 5]
    nw_gamma = [10 ** i for i in range(-3, 4)] + [0.5, 5, 50, 200, 500, 700]
    X = np.concatenate((x_cnt, x_trt), axis=0)
    T = np.concatenate((T_cnt, T_trt), axis=0)
    D = np.concatenate((delta_cnt, delta_trt), axis=0)
    W = np.concatenate((np.zeros(x_cnt.shape[0]), np.ones(x_trt.shape[0])), axis=0)
    other_models = {
        'T-SF': TValLearner(MySurvForest(trees, depth, leaf_samples), MySurvForest(trees, depth, leaf_samples), (x_val, T_val, delta_val), None),
        'S-SF': SValLearner(MySurvForest(trees, depth, leaf_samples), (x_val, T_val, delta_val), None),
        'X-SF': XValLearner(MySurvForest(trees, depth, leaf_samples), MySurvForest(trees, depth, leaf_samples), MySurvForest(trees, depth, leaf_samples), MySurvForest(trees, depth, leaf_samples), (x_val, T_val, delta_val), None),
        'T-NW': TValLearner(NWSurv(nw_gamma), NWSurv(nw_gamma), (x_val, T_val, delta_val), None),
        'S-NW': SValLearner(NWSurv(nw_gamma), (x_val, T_val, delta_val), None),
        'X-NW': XValLearner(NWSurv(nw_gamma), NWSurv(nw_gamma), NWSurv(nw_gamma), NWSurv(nw_gamma), (x_val, T_val, delta_val), None),
        'T-Cox': TValLearner(MyCox(alpha_cox), MyCox(alpha_cox), (x_val, T_val, delta_val), None),
        'S-Cox': SValLearner(MyCox(alpha_cox), (x_val, T_val, delta_val), None),
        'X-Cox': XValLearner(MyCox(alpha_cox), MyCox(alpha_cox), MyCox(alpha_cox), MyCox(alpha_cox), (x_val, T_val, delta_val), None),
    }
    T_test_cate = T_test_trt - T_test_cnt
    for key, instance in other_models.items():
        instance.fit(X, T, D, W)
        if (hasattr(instance, 'predict_control')):
            control_pred = instance.predict_control(x_test, predict_T)
            res_dict[f'{key} control metric'].append(rmse(control_pred, T_test_cnt))
        if (hasattr(instance, 'predict_treat')):
            treat_pred = instance.predict_treat(x_test, predict_T)
            res_dict[f'{key} treat metric'].append(rmse(treat_pred, T_test_trt))
        cate_pred = instance.predict(x_test, predict_T)
        res_dict[f'{key} CATE metric'].append(rmse(cate_pred, T_test_cate))
    

def size_exp(cnt_func, trt_func, noise_perc, t_bnds, censored_part, trt_part, test_size, val_part, size_list):
    global n 
    for s in size_list:
        n = trt_size = int(trt_part * s)
        val_size = int(val_part * s)
        data_dict = make_treat_set(cnt_func, trt_func, noise_perc, t_bnds, censored_part, s, trt_size, test_size, val_size)
        exp_iter(data_dict)


def part_exp(cnt_func, trt_func, noise_perc, t_bnds, censored_part, cnt_size, trt_parts, test_size, val_part):
    global n
    n = int(cnt_size * trt_parts[0])
    val_size = int(val_part * cnt_size)
    for trt_part in trt_parts:
        trt_size = int(trt_part * cnt_size)
        data_dict = make_treat_set(cnt_func, trt_func, noise_perc, t_bnds, censored_part, cnt_size, trt_size, test_size, val_size)
        exp_iter(data_dict)

def cens_exp(cnt_func, trt_func, noise_perc, t_bnds, censored_list, cnt_size, trt_part, test_size, val_part):
    for c in censored_list:
        data_dict = make_treat_set(cnt_func, trt_func, noise_perc, t_bnds, c, cnt_size, trt_size, test_size, val_size)
        exp_iter(data_dict)

def noise_exp(cnt_func, trt_func, noise_list, t_bnds, censored_part, cnt_size, trt_part, test_size, val_part):
    for n_p in noise_list:
        data_dict = make_treat_set(cnt_func, trt_func, n_p, t_bnds, censored_part, cnt_size, trt_size, test_size, val_size)
        exp_iter(data_dict)

def table_exp(cnt_func, trt_func, noise_perc, t_bnds, censored_part, cnt_size, trt_part, test_size, val_part):
    data_dict = make_treat_set(cnt_func, trt_func, noise_perc, t_bnds, censored_part, cnt_size, trt_size, test_size, val_size)
    exp_iter(data_dict)

if __name__ == '__main__':
    time_stamp = time.time()
    seed = 123
    np.random.seed(seed)
    try:
        ser_name = 'table0'
        res_dict = defaultdict(list)
        test_size = 1000
        cnt_size = 200
        val_part = 0.5
        val_size = int(val_part * cnt_size)
        censored_part = 0.25
        trt_part = 0.2
        trt_size = int(trt_part * cnt_size)
        batch_size = 128
        epochs = 1000
        patience = 10
        m = 10
        b_cnt = 0.5
        b_trt = 0.15
        nu_cnt = 1
        nu_trt = 1
        u_cnt = 0.02
        u_trt = 0.3
        l_cnt = 0.1
        l_trt = 0.1
        n = trt_size
        mlp_coef = 3
        t_bnds = (0, 10)
        noise_perc = 0.0
        trt_func = spiral_func(m)
        cnt_func = spiral_func(m)
        # cnt_func = pow_func(m)
        # trt_func = pow_func(m)
        # cnt_func = log_func(m, (-4, -1), (1, 4))
        # trt_func = log_func(m, (-4, -1), (1, 4))
        size_list = [100, 200, 300, 500, 1000]
        trt_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        cens_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        noise_list = [0, 0.05, 0.1, 0.15]
        # size_exp(cnt_func, trt_func, noise_perc, t_bnds, censored_part, trt_part, test_size, val_part, size_list)
        # part_exp(cnt_func, trt_func, noise_perc, t_bnds, censored_part, cnt_size, trt_list, test_size, val_part)
        # cens_exp(cnt_func, trt_func, noise_perc, t_bnds, cens_list, cnt_size, trt_part, test_size, val_part)
        noise_exp(cnt_func, trt_func, noise_list, t_bnds, censored_part, cnt_size, trt_part, test_size, val_part)
        # table_exp(cnt_func, trt_func, noise_perc, t_bnds, censored_part, cnt_size, trt_part, test_size, val_part)
        
        
    except KeyboardInterrupt:
        pass

    for key, val in res_dict.items():
        print(f'{key}: {np.mean(val)}')
    
    print('\ntime elapsed: ', round(time.time() - time_stamp, 0), ' s.')
    
    params = {
        'batch_size': batch_size,
        'epochs_num': epochs,
        'mlp_coef': mlp_coef,
        'patience': patience,
        'm': m,
        'n': n,
        'control_sizes': size_list,
        'control_size': cnt_size,
        'treat_part': trt_part,
        'treat_parts': trt_list,
        'cens_parts': cens_list,
        'test_size': test_size, 
        'val_part': val_part,
        'noise_perc': noise_perc,
        'noise_list': noise_list,
        't_bounds': t_bnds,
    }

    res_dict['params'] = params
    with open(f'surv_dicts/{ser_name}.pk', 'wb') as file:
        pickle.dump(res_dict, file)