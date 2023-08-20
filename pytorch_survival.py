import torch
import numpy as np
import time
from collections import defaultdict
import json
from BENK import BENKDataGenerator, BENK, train_model, device, make_spec_set, tau_loss
from learners import make_t_learner, make_s_learner, make_x_learner
from models import NWSurv, MyCox, MySurvForest
from funcs import gauss_func, circle_func, spiral_func
from sklearn.ensemble import RandomForestRegressor
from sksurv.linear_model import CoxnetSurvivalAnalysis

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
    if T.ndim == 1:
        T_diff = T[None, 1:] - T[None, :-1]
        T_start = T[0]
    else:
        T_diff = T[:, 1:] - T[:, :-1]
        T_start = T[:, 0]
    integral = T_start + np.sum(S_diff[:, :-1] * T_diff, axis=-1)
    return np.sqrt(np.mean((integral - (T_1.ravel() - T_0.ravel())) ** 2))


def get_data_from_param_func(func, t, censored_part=0.6):
    x = func.calc_x(t)
    x = (x - x.mean(axis=0)[None, :]) / x.std(axis=0)[None, :]
    T = np.float32(func.calc_T(t))
    # print('T mean:', np.mean(T))
    # print('T var:', np.var(T))
    T[T > 2000] = 2000
    delta = np.float32(np.random.binomial(1, 1 - censored_part, size=t.shape[0]))
    return x, T, delta


def make_treat_set(cnt_func, trt_func, t_bounds, censored_part, cnt_size, trt_size, test_size, val_size):
    x_cnt = x_trt = x_test = x_val = float('nan')
    while np.any(np.logical_not(np.isfinite(x_cnt))):
        t_cnt = np.random.uniform(t_bounds[0], t_bounds[1], cnt_size)
        x_cnt, T_cnt, delta_cnt = get_data_from_param_func(cnt_func, t_cnt, censored_part)
    while np.any(np.logical_not(np.isfinite(x_trt))):
        t_trt = np.random.uniform(t_bounds[0], t_bounds[1], trt_size)
        x_trt, T_trt, delta_trt = get_data_from_param_func(trt_func, t_trt, censored_part)
    while np.any(np.logical_not(np.isfinite(x_test))):
        t_test = np.random.uniform(t_bounds[0], t_bounds[1], test_size)
        x_test, T_test_trt, delta_test = get_data_from_param_func(trt_func, t_test, censored_part)
    _, T_test_cnt, _ = get_data_from_param_func(cnt_func, t_test, censored_part)
    T_test_trt_exp = trt_func.calc_expected(t_test)
    T_test_cnt_exp = cnt_func.calc_expected(t_test)
    while np.any(np.logical_not(np.isfinite(x_val))):
        t_val = np.random.uniform(t_bounds[0], t_bounds[1], val_size)
        x_val, T_val, delta_val = get_data_from_param_func(cnt_func, t_val, censored_part)
    res_dict = dict()
    res_dict['x_cnt'], res_dict['T_cnt'], res_dict['delta_cnt'] = x_cnt, T_cnt, delta_cnt
    res_dict['x_trt'], res_dict['T_trt'], res_dict['delta_trt'] = x_trt, T_trt, delta_trt
    res_dict['x_test'], res_dict['T_test_cnt'], res_dict['T_test_trt'], res_dict['delta_test'] = x_test, T_test_cnt, T_test_trt, delta_test
    res_dict['x_val'], res_dict['T_val'], res_dict['delta_val'] = x_val, T_val, delta_val
    res_dict['t_cnt'], res_dict['t_trt'], res_dict['t_val'], res_dict['t_test'] = t_cnt, t_trt, t_val, t_test
    res_dict['T_test_trt_exp'], res_dict['T_test_cnt_exp'] = T_test_trt_exp, T_test_cnt_exp
    return res_dict


def arrs_to_torch_dev(*args):
    return [torch.from_numpy(a).to(device) for a in args]


def rmse(x, y):
    return np.sqrt(np.mean((x.ravel() - y.ravel()) ** 2))


def exp_iter(data_dict, do_cv=True, models_params=None):
    x_cnt, T_cnt, delta_cnt = data_dict['x_cnt'], data_dict['T_cnt'], data_dict['delta_cnt']
    x_trt, T_trt, delta_trt = data_dict['x_trt'], data_dict['T_trt'], data_dict['delta_trt']
    x_test, T_test_cnt, T_test_trt, delta_test = data_dict['x_test'], data_dict[
        'T_test_cnt'], data_dict['T_test_trt'], data_dict['delta_test']
    x_val, T_val, delta_val = data_dict['x_val'], data_dict['T_val'], data_dict['delta_val']
    T_test_trt_exp, T_test_cnt_exp = data_dict['T_test_trt_exp'], data_dict['T_test_cnt_exp']
    model = BENK(m).to(device)
    data_gen = BENKDataGenerator(x_cnt, T_cnt, delta_cnt, batch_size, n, mlp_coef)
    x_in_t_c, cnt_times, delta_in_t_c, x_p_t_c, cnt_labels, cnt_d = make_spec_set(
        x_cnt, x_test, T_cnt, T_test_cnt, delta_cnt, delta_test, x_cnt.shape[0], m, 1)
    x_in_t_c, T_in_t_c, delta_in_t_c, x_p_t_c = arrs_to_torch_dev(
        x_in_t_c, cnt_times, delta_in_t_c, x_p_t_c)
    x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t, trt_labels, trt_d = make_spec_set(
        x_trt, x_test, T_trt, T_test_trt, delta_trt, delta_test, x_trt.shape[0], m, 1)
    x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t = arrs_to_torch_dev(
        x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t)

    predict_T = np.sort(np.concatenate((T_cnt, T_trt)))
    predict_T_torch = arrs_to_torch_dev(predict_T)[0][None, :].repeat(x_in_t_c.shape[0], 1)

    def calc_my_model_metrics(*keys):
        S_0 = model.predict_in_points(x_in_t_c, T_in_t_c, delta_in_t_c, x_p_t_c, predict_T_torch)
        res_dict[keys[0]].append(tau_metric(S_0, predict_T, cnt_labels))
        S_1 = model.predict_in_points(x_in_t_t, T_in_t_t, delta_in_t_t, x_p_t_t, predict_T_torch)
        res_dict[keys[1]].append(tau_metric(S_1, predict_T, trt_labels))
        res_dict[keys[2]].append(tau_cate_diff(S_1 - S_0, predict_T, trt_labels, cnt_labels))

        res_dict[keys[3]].append(tau_metric(S_0, predict_T, T_test_cnt_exp))
        res_dict[keys[4]].append(tau_metric(S_1, predict_T, T_test_trt_exp))
        res_dict[keys[5]].append(tau_metric(S_1 - S_0, predict_T, T_test_trt_exp - T_test_cnt_exp))

    *val_data, val_labels, val_d = make_spec_set(x_cnt, x_val,
                                                 T_cnt, T_val, delta_cnt, delta_val, x_cnt.shape[0], m, 1)
    *val_data, val_labels, val_d = arrs_to_torch_dev(*val_data, val_labels, val_d)
    optimizer = torch.optim.AdamW(model.parameters(), 0.001)
    train_model(data_gen, model, tau_loss, optimizer, epochs,
                (val_data, val_labels, val_d), patience)
    calc_my_model_metrics('Kernel control', 'Kernel treat', 'Kernel CATE',
                          'Kernel control PEHE', 'Kernel treat PEHE', 'Kernel CATE PEHE')

    X = np.concatenate((x_cnt, x_trt), axis=0)
    T = np.concatenate((T_cnt, T_trt), axis=0)
    D = np.concatenate((delta_cnt, delta_trt), axis=0)
    W = np.concatenate((np.zeros(x_cnt.shape[0]), np.ones(x_trt.shape[0])), axis=0)
    Y = np.ndarray(shape=(T.shape[0]), dtype=[('censor', '?'), ('time', 'f4')])
    Y['censor'] = D.astype(int)
    Y['time'] = T
    Y_val = np.ndarray(shape=(T_val.shape[0]), dtype=[('censor', '?'), ('time', 'f4')])
    Y_val['censor'] = delta_val.astype(int)
    Y_val['time'] = T_val
    t_train = np.concatenate((data_dict['t_cnt'], data_dict['t_trt']), axis=0)[:, None]
    t_test, t_val = data_dict['t_test'][:, None], data_dict['t_val'][:, None]

    n_splits = 3
    if do_cv:
        t_sf_cv = s_sf_cv = x_sf_cv = reg_sf_cv = reg_nw_cv = reg_cox_cv = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [3, 4, 5, 6],
            'min_samples_leaf': [1, 0.01, 0.05, 0.1],
        }
        t_nw_cv = s_nw_cv = x_nw_cv = {
            'gamma': [10 ** i for i in range(-4, 4)] + [0.5, 5, 50, 200, 500, 700]
        }
        n_alphas = 20
        cox = CoxnetSurvivalAnalysis(n_alphas=n_alphas).fit(X, Y)
        t_cox_cv = s_cox_cv = x_cox_cv = {
            'alphas': [[a] for a in cox.alphas_],
        }
        cox_sp = CoxnetSurvivalAnalysis(n_alphas=n_alphas).fit(t_train, Y)
        t_sp_cv = s_sp_cv = {
            'alphas': [[a] for a in cox_sp.alphas_],
        }
    else:
        reg_sf_cv, reg_nw_cv, reg_cox_cv = models_params['RF-SF'], models_params['RF-NW'], models_params['RF-Cox']
        t_sf_cv, s_sf_cv, x_sf_cv = models_params['T-SF'], models_params['S-SF'], models_params['X-SF']
        t_nw_cv, s_nw_cv, x_nw_cv = models_params['T-NW'], models_params['S-NW'], models_params['X-NW']
        t_cox_cv, s_cox_cv, x_cox_cv = models_params['T-Cox'], models_params['S-Cox'], models_params['X-Cox']
        t_sp_cv, s_sp_cv = models_params['T-Spoiler'], models_params['S-Spoiler']
    other_models = {
        'T-SF': (make_t_learner, (x_val, Y_val, MySurvForest, t_sf_cv), (X, Y, W)),
        'S-SF': (make_s_learner, (x_val, Y_val, MySurvForest, s_sf_cv), (X, Y, W)),
        'X-SF': (make_x_learner, (x_val, Y_val, MySurvForest, RandomForestRegressor, x_sf_cv, reg_sf_cv), (X, Y, W)),
        'T-NW': (make_t_learner, (x_val, Y_val, NWSurv, t_nw_cv), (X, Y, W)),
        'S-NW': (make_s_learner, (x_val, Y_val, NWSurv, s_nw_cv), (X, Y, W)),
        'X-NW': (make_x_learner, (x_val, Y_val, NWSurv, RandomForestRegressor, x_nw_cv, reg_nw_cv), (X, Y, W)),
        'T-Cox': (make_t_learner, (x_val, Y_val, MyCox, t_cox_cv), (X, Y, W)),
        'S-Cox': (make_s_learner, (x_val, Y_val, MyCox, s_cox_cv), (X, Y, W)),
        'X-Cox': (make_x_learner, (x_val, Y_val, MyCox, RandomForestRegressor, x_cox_cv, reg_cox_cv), (X, Y, W)),
        'T-Spoiler': (make_t_learner, (t_val, Y_val, MyCox, t_sp_cv), (t_train, Y, W)),
        'S-Spoiler': (make_t_learner, (t_val, Y_val, MyCox, s_sp_cv), (t_train, Y, W)),
    }
    T_test_cate = T_test_trt - T_test_cnt
    T_test_cate_exp = T_test_trt_exp - T_test_cnt_exp
    print(f'Kernel CATE:', res_dict['Kernel CATE'][-1])
    print(f'Kernel PEHE:', res_dict['Kernel CATE PEHE'][-1])
    if do_cv:
        params_dict = dict()
    for key, val in other_models.items():
        instance = val[0](*val[1], n_splits, random_state=np.random.randint(1, 4096), do_cv=do_cv)
        cv_params = instance.fit(*val[2])
        test_points = x_test if 'Spoiler' not in key else t_test
        if (hasattr(instance, 'predict_control')):
            control_pred = instance.predict_control(test_points)
            res_dict[f'{key} control metric'].append(rmse(control_pred, T_test_cnt))
            res_dict[f'{key} control PEHE'].append(rmse(control_pred, T_test_cnt_exp))
        if (hasattr(instance, 'predict_treat')):
            treat_pred = instance.predict_treat(test_points)
            res_dict[f'{key} treat metric'].append(rmse(treat_pred, T_test_trt))
            res_dict[f'{key} treat PEHE'].append(rmse(treat_pred, T_test_trt_exp))
        cate_pred = instance.predict(test_points)
        if do_cv:
            if key.startswith('X'):
                params_dict[f'RF-{key[2:]}'] = cv_params[1]
                cv_params = cv_params[0]
            params_dict[key] = cv_params
        res_dict[f'{key} CATE metric'].append(rmse(cate_pred, T_test_cate))
        res_dict[f'{key} CATE PEHE'].append(rmse(cate_pred, T_test_cate_exp))
        print(f'{key} CATE:', res_dict[f'{key} CATE metric'][-1])
        print(f'{key} PEHE:', res_dict[f'{key} CATE PEHE'][-1])
    if do_cv:
        return params_dict


def size_exp(cnt_func, trt_func, t_bnds, censored_part, trt_part, test_size, val_part, size_list):
    global n
    global models_params_list
    for i, s in enumerate(size_list):
        trt_size = int(trt_part * s)
        n = min(100, trt_size)
        val_size = int(val_part * s)
        data_dict = make_treat_set(cnt_func, trt_func, t_bnds,
                                   censored_part, s, trt_size, test_size, val_size)
        new_params = exp_iter(data_dict, do_cv, None if do_cv else models_params_list[i])
        if do_cv:
            models_params_list.append(new_params)


def part_exp(cnt_func, trt_func, t_bnds, censored_part, cnt_size, trt_parts, test_size, val_part):
    global n
    global models_params_list
    n = int(cnt_size * trt_parts[0])
    val_size = int(val_part * cnt_size)
    for i, trt_part in enumerate(trt_parts):
        trt_size = int(trt_part * cnt_size)
        data_dict = make_treat_set(cnt_func, trt_func, t_bnds, censored_part,
                                   cnt_size, trt_size, test_size, val_size)
        new_params = exp_iter(data_dict, do_cv, None if do_cv else models_params_list[i])
        if do_cv:
            models_params_list.append(new_params)


def cens_exp(cnt_func, trt_func, t_bnds, censored_list, cnt_size, trt_part, test_size, val_part):
    global models_params_list
    for i, c in enumerate(censored_list):
        data_dict = make_treat_set(cnt_func, trt_func, t_bnds, c,
                                   cnt_size, trt_size, test_size, val_size)
        new_params = exp_iter(data_dict, do_cv, None if do_cv else models_params_list[i])
        if do_cv:
            models_params_list.append(new_params)


random_seed = 12345
test_size = 1000
cnt_size = 300
val_part = 0.5
val_size = int(val_part * cnt_size)
censored_part = 0.33
trt_part = 0.2
trt_size = int(trt_part * cnt_size)
batch_size = 128
epochs = 500
patience = 10
m = 10
b_cnt = 1.6
b_trt = 0.8
nu_cnt = 2
nu_trt = 2
l_cnt = 0.0005
l_trt = 0.005
n = trt_size
mlp_coef = 3
t_bnds = (0, 10)
size_list = [100, 200, 300, 500, 1000]
trt_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
cens_list = [0.1, 0.2, 0.3, 0.4, 0.5]

iter_num = 100
cv_period = 5
max_attempts = 5
res_dict = None

if __name__ == '__main__':
    time_stamp = time.time()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    for i in range(iter_num):
        j = 0
        succeeded = False
        while j < max_attempts and not succeeded:
            res_dict = defaultdict(list)
            do_cv = i % cv_period == 0
            if do_cv:
                models_params_list = []
            try:
                ser_name = f'spiral_size_{i}'
                cnt_func = spiral_func(m, b_cnt, l_cnt, nu_cnt)
                trt_func = spiral_func(m, b_trt, l_trt, nu_trt)
                # cnt_func = circle_func(m, t_bnds, b_cnt, l_cnt, nu_cnt)
                # trt_func = circle_func(m, t_bnds, b_trt, l_trt, nu_trt)
                # cnt_func = gauss_func(m, t_bnds, b_cnt, l_cnt, nu_cnt)
                # trt_func = gauss_func(m, t_bnds, b_trt, l_trt, nu_trt)
                size_exp(cnt_func, trt_func, t_bnds, censored_part,
                         trt_part, test_size, val_part, size_list)
                # part_exp(cnt_func, trt_func, t_bnds, censored_part, cnt_size, trt_list, test_size, val_part)
                # cens_exp(cnt_func, trt_func, t_bnds, cens_list, cnt_size, trt_part, test_size, val_part)

            except ArithmeticError:
                j += 1
                continue

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
                't_bounds': t_bnds,
                'l_cnt': l_cnt,
                'l_trt': l_trt,
            }

            res_dict['params'] = params
            succeeded = True
            with open(f'surv_dicts/{ser_name}.json', 'w') as file:
                def f32_to_double(x): return x.item()
                json.dump(res_dict, file, indent=1, default=f32_to_double)


    print('\ntime elapsed: ', round(time.time() - time_stamp, 0), ' s.')
