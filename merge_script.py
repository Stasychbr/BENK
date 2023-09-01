from pathlib import Path
import re
import json
from collections import defaultdict

NAME_PATTERN = 'spiral_size'
NAME_REG_EXPR = NAME_PATTERN + r'_\d+\.json'
METRIC_REG_EXPR = r'Kernel|metric|PEHE'
PATH = 'surv_dicts'

if __name__ == '__main__':
    name_reg = re.compile(NAME_REG_EXPR)
    metric_reg = re.compile(METRIC_REG_EXPR)
    dir = Path(PATH)
    d_merge = defaultdict(list)
    files_to_proc = filter(lambda d: name_reg.fullmatch(d.name), dir.iterdir())
    empty_flag = True
    for p_file in files_to_proc:
        empty_flag = False
        with open(str(p_file.absolute()), 'r') as file:
            d = json.load(file)
            for key in d:
                if metric_reg.search(key):
                    d_merge[key].append(d[key])
        print('Processed', p_file.name)
    if empty_flag:
        raise RuntimeError('No files to process!')
    d_merge['params'] = d['params']
    with open(f'surv_dicts/{NAME_PATTERN}_merge.json', 'w') as file:
        json.dump(d_merge, file, indent=1)
