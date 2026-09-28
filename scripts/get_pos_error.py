import json
import sys

import numpy as np


for target in sys.argv[1:]:
    data = json.load(open(target))

    obs_errors = []
    pos_errors = []
    for entry in data['runs']:
        obs_errors.append(entry['obs_error'])
        pos_errors.append(entry['pos_error'])
    print(f"obs {np.mean(obs_errors):.3f} pos {np.mean(pos_errors):.3f} | {target}")
