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

positions = []

for i in tqdm.trange(0, np.max(episode_idxs)):
    mask = episode_idxs == i
    episode = entries[mask]
    episode_cat = categories[mask]
    episode_actions = actions[mask]

    for action, obs, cat in zip(episode_actions, episode, episode_cat):
        robot_pos = obs[0, :2]
        positions.append(robot_pos)
    break
#print(np.array(positions))

import py_terminal_plotter as ptp

plotter = ptp.TerminalPlot(x_range=[0, 1], y_range=[0, 1])
plotter.create_axes(title="First Trajectory")
plotter.setup_image(100, 100, z_range=[0, 2])

display = np.zeros((100, 100))
for x, y in positions:
    px = int(x * 99)
    py = int((1 - y) * 99)
    print(px, py)
    display[py, px] = 1.0

plotter.plot_image_section(display, start_row=0)
plotter.draw()
