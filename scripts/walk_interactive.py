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
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints_2")
    data = torch.load(os.path.join(out_dir, "9.pth"), weights_only=True)

    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    model.eval()
    return model, data['latent_cache']

with open(os.path.join(SCRIPT_DIR, "world.json"), "r") as jf:
    data = json.load(jf)
seed = 42
np.random.seed(seed)
world = World2d(data)
world.reset()
precomputed_embeddings = np.load(f"embeddings/{seed}/embeddings.npy")

with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
    config = json.load(jf)
model, latents = load_model(config)

def model_update(latent, obs, action):
    obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
    print("tokens:", obs_tokens)
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
    obs_reconstruct = model.reconstruction(prior_latent)
    obs_err = (init_obs_embed - obs_reconstruct).pow(2).mean()
    prior_latent = prior_latent.cpu()
    print(obs_tokens)
    print(prior_latent)
    print(init_obs_embed[0])
past_obs = init_obs_embed[0].cpu()


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

from probe_network import MLPProbe
probe = MLPProbe()
probe.load_state_dict(torch.load("probe.pth"))
probe = probe.cuda()

def simplify_obs(obs):
    res = "r" + 'c'*len(obs['containers']) + 'i'*len(obs['items'])
    if obs['pickup'] is not None:
        res += 'p'
    if obs['drop'] is not None:
        res += 'd'
    return res

def render(action):
    global prior_latent
    action = [ 0.00183289, -0.00295556, 0.0 ]
    action = torch.tensor(action)
    obs_new = world.update(action)
    obs_emb, latents, obs_reconstruct = model_update(prior_latent, obs_new, action)

    obs_err = (obs_emb - obs_reconstruct).norm().mean()
    prev_latent = prior_latent
    #print(latents[0].norm(), latents[1].norm())
    #input()
    #prior_latent = model.init_state(obs_emb.cuda())
    prior_latent = latents[-1]
    #prior_latent = latents[-2]
    pred_err = (latents[-2] - latents[-1]).norm().mean()
    obs_simplify = simplify_obs(obs_new)
    a = action.tolist()
    obs_mag = torch.norm(obs_emb)
    latent_mag = torch.norm(latents[-1])
    #prior_latent /= latent_mag
    #prior_latent *= 8.0
    init_state = model.init_state(obs_emb.cuda())
    cheat_reconstruct = model.reconstruction(init_state).detach().cpu()
    
    obs_delta = obs_emb - past_obs
    pred_delta = obs_reconstruct - past_obs
    print("cheat | recons:", (cheat_reconstruct - obs_emb).norm(), (obs_reconstruct - obs_emb).norm())
    print(past_obs)
    print(obs_emb)
    print(obs_delta)
    #print("Obs delta:")
    print("norm(d) norm(d') align center", obs_delta.norm(), pred_delta.norm(), (obs_delta @ pred_delta) / (obs_delta.norm() * pred_delta.norm()), pred_delta @ obs_reconstruct)
    print("latent_v", (prior_latent - prev_latent).norm())
    print("emb rec pas", obs_emb.norm(), obs_reconstruct.norm(), past_obs.norm())
    print("raw   ", obs_emb)
    print("reset ", cheat_reconstruct)
    print("recons", obs_reconstruct)
    input()

    distances = np.linalg.norm(obs_reconstruct.numpy() - precomputed_embeddings, axis=-1)
    #distances = np.linalg.norm(cheat_reconstruct.numpy() - precomputed_embeddings, axis=-1)
    # Coordinates in distance grid are (x, y)
    max_position = np.array(np.unravel_index(np.argmin(distances), distances.shape)) / 100
    x, y = world.robot.pos
    px = int(np.round(x * 99))
    py = int(np.round(x * 99))
    #print(precomputed_embeddings[px, py])
    #print(obs_reconstruct)
    #print(obs_emb)
    #print(np.min(distances), obs_err, max_position)
    #input()

    with torch.no_grad():
        v = obs_reconstruct.unsqueeze(0).cuda()
        probe_res = probe(v).cpu()[0]
    probe_x, probe_y = probe_res
    title = f"Interactive world (obs: {obs_simplify}, action: [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}], obs_err: {obs_err:.3f} obs_mag: {obs_mag:.3f}"
    title += f" latent_mag: {latent_mag:.3f} pred_err: {pred_err:.3f} probe ({probe_x:.3f}, {probe_y:.3f})"
    title += f" closest ({max_position[0]:.3f}, {max_position[1]:.3f}) real ({world.robot.pos[0]:.3f}, {world.robot.pos[1]:.3f})"
    plotter.set_title(title)

    display = world.render()
    display = 255 - np.mean(display, axis=-1)
    plotter.plot_image_section(display, start_row=0)
    #plotter.draw()

try:
    while True:
        key = getKey()
        #key = 'w'
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
        if key == ' ':
            render([0.0, 0.0, 0.0])
        if key == 'q':
            break
        time.sleep(0.05)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
