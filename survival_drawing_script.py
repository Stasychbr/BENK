import pickle
import matplotlib.pyplot as plt

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

def draw_size(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    sizes = d['params']['control_sizes']
    models = ['SF', 'NW', 'Cox']
    frameworks = ['T', 'S', 'X']
    for m in models:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel('control size')
        ax.set_ylabel('RMSE')
        ax.set_xticks(sizes)
        ax.set_title('CATE')
        ax.semilogy(sizes, d['Kernel CATE'], **format['Kernel'], label='BENK')
        for fw in frameworks:
            name = f'{fw}-{m}'
            ax.semilogy(sizes, d[f'{name} CATE metric'], **format[name], label=name)
        ax.legend()
        ax.grid()
        fig.savefig(f'Size_CATE_{m}_{func}.pdf')

def draw_part(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    parts = d['params']['treat_parts']
    models = ['SF', 'NW', 'Cox']
    frameworks = ['T', 'S', 'X']
    for m in models:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel('treatment part')
        ax.set_ylabel('RMSE')
        ax.set_xticks(parts)
        ax.set_xticklabels([f'{int(p  * 100)}%' for p in parts])
        ax.set_title('CATE')
        ax.semilogy(parts, d['Kernel CATE'], **format['Kernel'], label='BENK')
        for fw in frameworks:
            name = f'{fw}-{m}'
            ax.semilogy(parts, d[f'{name} CATE metric'], **format[name], label=name)
        ax.legend()
        ax.grid()
        fig.savefig(f'Parts_CATE_{m}_{func}.pdf')

def draw_cens(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    parts = d['params']['cens_parts']
    models = ['SF', 'NW', 'Cox']
    frameworks = ['T', 'S', 'X']
    for m in models:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel('censored part')
        ax.set_ylabel('RMSE')
        ax.set_xticks(parts)
        ax.set_xticklabels([f'{int(p  * 100)}%' for p in parts])
        ax.set_title('CATE')
        ax.semilogy(parts, d['Kernel CATE'], **format['Kernel'], label='BENK')
        for fw in frameworks:
            name = f'{fw}-{m}'
            ax.semilogy(parts, d[f'{name} CATE metric'], **format[name], label=name)
        ax.legend()
        ax.grid()
        fig.savefig(f'Cens_CATE_{m}_{func}.pdf')

def draw_noise(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    parts = d['params']['noise_list']
    models = ['SF', 'NW', 'Cox']
    frameworks = ['T', 'S', 'X']
    for m in models:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel('noise level')
        ax.set_ylabel('RMSE')
        ax.set_xticks(parts)
        ax.set_xticklabels([f'{int(p  * 100)}%' for p in parts])
        ax.set_title('CATE')
        ax.semilogy(parts, d['Kernel CATE'], **format['Kernel'], label='BENK')
        for fw in frameworks:
            name = f'{fw}-{m}'
            ax.semilogy(parts, d[f'{name} CATE metric'], **format[name], label=name)
        ax.legend()
        ax.grid()
        fig.savefig(f'Noise_CATE_{m}_{func}.pdf')

func = 'table0' # only for filename
# draw_size('surv_dicts/table0.pk')
# draw_part('surv_dicts/srl_part_fix.pk')
# draw_cens('surv_dicts/srl_cens_fix.pk')
draw_noise('surv_dicts/table0.pk')
