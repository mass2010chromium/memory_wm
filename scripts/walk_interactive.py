import json
import os
SCRIPT_DIR = os.path.dirname(__file__)

import einops
import numpy as np
import torch

import py_terminal_plotter as ptp

plotter = ptp.TerminalPlot(x_range=[0, 1], y_range=[0, 1])
plotter.create_axes(title="Interactive world")
plotter.setup_image(256, 256, z_range=[0, 255])

from memory_wm.module import Predictor

from env_2d import tokenize_obs, World2d, MAX_TOKENS

def load_model(model_config):
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data = torch.load(os.path.join(out_dir, "99.pth"), weights_only=True)

    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    model.eval()
    return model, data['latent_cache']

with open(os.path.join(SCRIPT_DIR, "world.json"), "r") as jf:
    data = json.load(jf)
world = World2d(data)
world.reset()

with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
    config = json.load(jf)
model, latents = load_model(config)

def model_update(latent, obs, action):
    obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
    with torch.no_grad():
        obs_emb, latents, obs_reconstruct = model(
            latent.unsqueeze(0).cuda(),
            torch.tensor(obs_tokens).float().unsqueeze(0).cuda(),
            torch.tensor(token_mask).unsqueeze(0).cuda(),
            torch.tensor(obs_categories).unsqueeze(0).cuda(),
            action.unsqueeze(0).cuda()
        )
        return obs_emb[0].cpu(), latents[0].cpu(), obs_reconstruct[0].cpu()

init_obs = world.update([0.0, 0.0, 0.0])
obs_tokens, obs_categories, token_mask = tokenize_obs(init_obs, pad_to_size=MAX_TOKENS)
with torch.no_grad():
    init_obs_embed = model.embed_obs(
        torch.tensor(obs_tokens).float().unsqueeze(0).cuda(),
        torch.tensor(token_mask).unsqueeze(0).cuda(),
        torch.tensor(obs_categories).unsqueeze(0).cuda()
    )
    # Unbatch
    prior_latent = model.init_state(init_obs_embed[0])


import sys, select, termios, time, tty
def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
    return key
_settings = termios.tcgetattr(sys.stdin)

def simplify_obs(obs):
    res = "r" + 'c'*len(obs['containers']) + 'i'*len(obs['items'])
    if obs['pickup'] is not None:
        res += 'p'
    if obs['drop'] is not None:
        res += 'd'
    return res

def render(action):
    global prior_latent
    action = torch.tensor(action)
    obs_new = world.update(action)
    obs_emb, latents, obs_reconstruct = model_update(prior_latent, obs_new, action)

    obs_err = (obs_emb - obs_reconstruct).pow(2).mean()
    prior_latent = latents[-1]
    obs_simplify = simplify_obs(obs_new)
    a = action.tolist()
    plotter.set_title(f"Interactive world (obs: {obs_simplify}, action: [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}], obs_err: {obs_err:.3f})")

    display = world.render()
    display = 255 - np.mean(display, axis=-1)
    plotter.plot_image_section(display, start_row=0)
    plotter.draw()

try:
    while True:
        key = getKey()
        if key == 'w':
            render([0.0, 0.05, 0.0])
        if key == 's':
            render([0.0, -0.05, 0.0])
        if key == 'd':
            render([0.05, 0.0, 0.0])
        if key == 'a':
            render([-0.05, 0.0, 0.0])
        if key == 'x':
            render([0.0, 0.0, 1.0])
        if key == 'c':
            render([0.0, 0.0, -1.0])
        if key == 'q':
            break
        time.sleep(0.05)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
