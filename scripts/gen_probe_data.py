import json
import os
SCRIPT_DIR = os.path.dirname(__file__)

import numpy as np
import torch
import tqdm

from memory_wm.module import Predictor

from env_2d import N_TYPES, OBS_SIZE, World2d, MAX_TOKENS
from robot_utils import model_update, control_robot_to, gen_sample

def load_model(model_config):
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data = torch.load(os.path.join(out_dir, "199.pth"), weights_only=True)

    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    model.eval()
    return model, data['latent_cache']

if __name__ == "__main__":
    with open(os.path.join(SCRIPT_DIR, "world.json"), "r") as jf:
        data = json.load(jf)
    world = World2d(data)

    with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
        config = json.load(jf)
    model, latents = load_model(config)

    positions = []
    latents = []

    nsample = 5000
    # for i in tqdm.trange(nsample):
    #     obs1, real1, latent1 = gen_sample(world, model, seed=i, container=0)
    #     obs2, real2, latent2 = gen_sample(world, model, seed=i, container=1)
    #     pos1 = world.containers[0].pos
    #     pos2 = world.containers[1].pos

    #     positions.append(pos1)
    #     latents.append(latent1[-1].numpy())

    #     positions.append(pos2)
    #     latents.append(latent2[-1].numpy())

    # np.save("probe_positions.npy", np.stack(positions))
    # np.save("probe_latents.npy", np.stack(latents))

    positions = []
    latents = []
    for i in tqdm.trange(nsample, nsample+500):
        obs1, real1, latent1 = gen_sample(world, model, seed=i, container=0)
        obs2, real2, latent2 = gen_sample(world, model, seed=i, container=1)
        pos1 = world.containers[0].pos
        pos2 = world.containers[1].pos

        positions.append(pos1)
        latents.append(latent1[-1].numpy())

        positions.append(pos2)
        latents.append(latent2[-1].numpy())

    np.save("probe_positions_val.npy", np.stack(positions))
    np.save("probe_latents_val.npy", np.stack(latents))
