import json
import os
SCRIPT_DIR = os.path.dirname(__file__)

from einops import rearrange
import numpy as np
import mediapy
import torch
import torch.optim as optim
import tqdm

from memory_wm.module import Predictor

from probe_network import MLPProbe

from env_2d import N_TYPES, OBS_SIZE, MAX_TOKENS, World2d, Item
from robot_utils import model_update, control_robot_to, gen_sample

hidden_size = 192

def load_model(model_config):
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data = torch.load(os.path.join(out_dir, "399.pth"), weights_only=True)

    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    model.eval()
    return model

if __name__ == "__main__":
    probe = MLPProbe()
    probe.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "probe.pth")))
    probe.cuda()

    with open("world.json", "r") as datafile:
        data = json.load(datafile)
    world = World2d(data)

    with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
        config = json.load(jf)
    model = load_model(config)

    #min_bounds = np.array(world.data['bounds']['min'])
    #max_bounds = np.array(world.data['bounds']['max'])
    min_bounds = np.array([-1, -1])
    max_bounds = np.array([2, 2])
    fake_item = Item(-1, None, -1)
    fake_item.color = (0, 255, 0)
    image_out_dir = os.path.join(SCRIPT_DIR, "renders")
    os.makedirs(image_out_dir, exist_ok=True)
    for i in tqdm.trange(5000, 5020):
        print(i)
        obs1, real1, latent1 = gen_sample(world, model, seed=i, container=0)
        canvas = world.render(resolution=512, draw_items=True, bounds=[min_bounds, max_bounds])
        with torch.no_grad():
            p0 = probe(latent1[-1].cuda()).squeeze().cpu().numpy()
            fake_item.pos = p0
        fake_item.render(canvas, min_bounds, max_bounds)
        mediapy.write_image(os.path.join(image_out_dir, f"{i}_0.png"), canvas)

        obs2, real2, latent2 = gen_sample(world, model, seed=i, container=1)
        canvas = world.render(resolution=512, draw_items=True, bounds=[min_bounds, max_bounds])
        with torch.no_grad():
            p1 = probe(latent2[-1].cuda()).squeeze().cpu().numpy()
            fake_item.pos = p1
        fake_item.render(canvas, min_bounds, max_bounds)
        mediapy.write_image(os.path.join(image_out_dir, f"{i}_1.png"), canvas)
        
        target_vec = world.containers[1].pos - world.containers[0].pos
        real_vec = p1 - p0
        if (real_vec @ target_vec) > 0:
            print("success")
        else:
            print("failure")
