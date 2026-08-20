import json
import os
SCRIPT_DIR = os.path.dirname(__file__)

import numpy as np
import tqdm

from env_2d_dataset import SmallPackedDataset
from env_2d import CLOSE_DROP_ID, CLOSE_PICKUP_ID

dataset = SmallPackedDataset(root=os.path.join(SCRIPT_DIR, "world2d_reorder"))

actions = dataset.data_map['action']
entries = dataset.data_map['observation.tokens']
categories = dataset.data_map['observation.token_categories']
episode_idxs = dataset.data_map['episode_index']

print(len(actions))
exit(0)

has_bad_pickup = 0
has_bad_drop = 0
has_re_pickup = 0

for i in tqdm.trange(0, np.max(episode_idxs)):
    mask = episode_idxs == i
    episode = entries[mask]
    episode_cat = categories[mask]
    episode_actions = actions[mask]

    can_pickup_container = False
    can_pickup = False
    can_drop = False
    for action, obs, cat in zip(episode_actions, episode, episode_cat):

        if not can_pickup and action[2] == 1:
            print("bad pickup", i)
            has_bad_pickup += 1
        if not can_drop and action[2] == -1:
            print("bad drop", i)
            has_bad_drop += 1

        if can_pickup_container and action[2] == 1:
            print("re pickup", i)
            has_re_pickup += 1

        can_pickup_container = False
        close_pickup = cat[:, CLOSE_PICKUP_ID] == 1
        if np.any(close_pickup):
            idx = np.argwhere(close_pickup)[0, 0]
            if obs[idx, -1] > 0 and not can_pickup:
                # Just grab the rising edge
                can_pickup_container = True
            can_pickup = True
        else:
            can_pickup = False
                
        can_drop = obs[0, 2] > 0

    #if has_bad_pickup and has_bad_drop and has_re_pickup:
    #    break
print(has_bad_pickup, has_bad_drop, has_re_pickup)
