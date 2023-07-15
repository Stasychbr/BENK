from pathlib import Path
import re
import json
from collections import defaultdict

NAME_PATTERN = 'spiral_size'
REG_PATTERN = NAME_PATTERN + r'_\d+\.json'
PATH = 'surv_dicts'

if __name__ == '__main__':
    reg = re.compile(REG_PATTERN)
    dir = Path(PATH)
    d_merge = defaultdict(list)
    files_to_proc = filter(lambda d: reg.fullmatch(d.name) is not None, dir.iterdir())
    empty_flag = True
    for p_file in files_to_proc:
        print('Processed', p_file.name)
        empty_flag = False
        with open(str(p_file.absolute()), 'r') as file:
            d = json.load(file)
            for key in d:
                if 'Kernel' in key or 'metric' in key:
                    d_merge[key].append(d[key])
    if empty_flag:
        raise RuntimeError('No files to process!')
    d_merge['params'] = d['params']
    with open(f'surv_dicts/{NAME_PATTERN}_merge.json', 'w') as file:
        json.dump(d_merge, file, indent=1)
