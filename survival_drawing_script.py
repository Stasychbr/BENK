import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style='dark')

format = {
        'T-NW': {'marker': 'v', 'linestyle': (0, (1, 4)), 'markerfacecolor': 'white'},
        'S-NW': {'marker': '^', 'linestyle': (0, (1, 5)), 'markerfacecolor': 'white'},
        'X-NW': {'marker': 'o', 'linestyle': (0, (1, 5)), 'markerfacecolor': 'white'},
        'T-SF': {'marker': 'v', 'linestyle': (0, (5, 5)), 'markerfacecolor': 'white'},
        'S-SF': {'marker': '^', 'linestyle': (0, (5, 5)), 'markerfacecolor': 'white'},
        'X-SF': {'marker': 'o', 'linestyle': (0, (5, 5)), 'markerfacecolor': 'white'},
        'T-Cox': {'marker': 'v', 'linestyle': (0, (3, 5, 1, 5)), 'markerfacecolor': 'white'},
        'S-Cox': {'marker': '^', 'linestyle': (0, (3, 5, 1, 5)), 'markerfacecolor': 'white'},
        'X-Cox': {'marker': 'o', 'linestyle': (0, (3, 5, 1, 5)), 'markerfacecolor': 'white'},
        'Kernel': {'marker': 'X', 'markerfacecolor': 'white'},
    }

spoiler_models = ['T-Spoiler', 'S-Spoiler']
    
def determine_best_models(values):
    cate_keys = filter(lambda name: 'CATE' in name, values.keys())
    excluded_models = spoiler_models + ['Kernel']
    models = list(filter(lambda name: not any([ex_name in name for ex_name in excluded_models]), cate_keys))
    frameworks = ['T', 'S', 'X']
    selected_keys = []
    for framework in frameworks:
        cur_models = list(filter(lambda name: name.startswith(framework), models))
        results = np.empty((len(cur_models), len(values['Kernel CATE'])))
        for i, model in enumerate(cur_models):
            results[i] = values[model]
        best_args = np.argmin(results, axis=0, keepdims=True)
        all_args = np.arange(len(cur_models))[:, None]
        cmp_args = all_args == best_args
        arg_count = np.count_nonzero(cmp_args, axis=-1)
        selected_keys.append(cur_models[np.argmax(arg_count)])
    return selected_keys

def draw_parts(path, f_draw_all=True):
    with open(path, 'r') as file:
        full_dict = json.load(file)
    x = full_dict['params']['treat_parts']
    x_ticks = [f'{int(p  * 100)}%' for p in x]
    draw_figures(full_dict, x, x_ticks, 'Treated part', 'Parts', f_draw_all)

def draw_size(path, f_draw_all=True):
    with open(path, 'r') as file:
        full_dict = json.load(file)
    x = full_dict['params']['control_sizes']
    draw_figures(full_dict, x, x, 'Control size', 'Size', f_draw_all)

def draw_cens(path, f_draw_all=True):
    with open(path, 'r') as file:
        full_dict = json.load(file)
    x = full_dict['params']['cens_parts']
    x_ticks = [f'{int(p  * 100)}%' for p in x]
    draw_figures(full_dict, x, x_ticks, 'Censored part', 'Cens', f_draw_all)


def draw_figures(full_dict, x_values, x_ticks, axis_name, prefix_name, f_draw_all=True):
    d_mean = dict()
    d_med = dict()
    for key in full_dict:
        if 'Kernel' in key or 'metric' in key:
            d_mean[key] = np.nanmean(np.clip(full_dict[key], a_min=None, a_max=30), axis=0)
            d_med[key] = np.nanmedian(np.clip(full_dict[key], a_min=None, a_max=30), axis=0)
    for file_name, d in zip(('mean', 'median'), (d_mean, d_med)):
        best_models = determine_best_models(d) + ['Kernel CATE']
        models_list = [best_models]
        file_postfixes = ['best']
        if f_draw_all:
            cate_keys = filter(lambda name: 'CATE' in name, full_dict.keys())
            all_models = list(filter(lambda name: not any([sp_name in name for sp_name in spoiler_models]), cate_keys))
            models_list.append(all_models)
            file_postfixes.append('all')
        
        for file_postfix, m_names in zip(file_postfixes, models_list):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.set_xlabel(axis_name)
            ax.set_ylabel('RMSE')
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_ticks)
            ax.set_title('CATE')
                    
            for m in m_names:
                name = m.split(' ', 1)[0]
                if 'NW' in name:
                    label = f'{name[:2]}Beran'
                elif 'Kernel' in name:
                    label = 'BENK'
                else:
                    label = name
                ax.plot(x_values, d[m], **format[name], label=label)
                
            ax.legend()
            ax.grid()
            fig.savefig(f'{prefix_name}_CATE_{file_name}_{file_postfix}_{func}.pdf')
            

def draw_best_boxplot_parts(path):
    with open(path, 'r') as file:
        d = json.load(file)
    d_mean = dict()
    for key in d:
        if 'Kernel' in key or 'metric' in key:
            d_mean[key] = np.mean(d[key], axis=0)
    d_mean['params'] = d['params']
    parts = d['params']['treat_parts']
    positions = [p * 10 for p in parts]
    best_models = determine_best_models(d_mean)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.boxplot(np.asarray(d['Kernel CATE']), positions=positions, showfliers=False)
    ax.set_xlabel('treatment part')
    ax.set_ylabel('RMSE')
    ax.set_xticks(parts)
    ax.set_xticklabels([f'{int(p  * 100)}%' for p in parts])
    ax.set_title('CATE')
    
    def get_short_name(res_name):
        for f_name in format:
            if res_name.startswith(f_name):
                return f_name
            
    for m in best_models:
        name = get_short_name(m)
        label = f'{name[:2]}Beran' if 'NW' in name else name
        ax.boxplot(np.asarray(d[m]), positions=positions, showfliers=False)
    # ax.legend()
    ax.grid()
    fig.savefig(f'Parts_CATE_boxplot_{m}_{func}.pdf')


func = 'circle' # only for filename
# draw_best_boxplot_parts('surv_dicts/spiral_part_merge.json')
draw_size(f'surv_dicts/{func}_size_merge.json')
draw_parts(f'surv_dicts/{func}_part_merge.json')
draw_cens(f'surv_dicts/{func}_cens_merge.json')
