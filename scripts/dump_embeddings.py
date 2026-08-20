import json
import os
SCRIPT_DIR = os.path.dirname(__file__)

import numpy as np
import torch
import tqdm

from memory_wm.module import Predictor

from env_2d import N_TYPES, OBS_SIZE, tokenize_obs, World2d, MAX_TOKENS
from robot_utils import model_update, control_robot_to, gen_sample

def load_model(model_config):
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data = torch.load(os.path.join(out_dir, "149.pth"), weights_only=True)

    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    model.eval()
    return model, data['latent_cache']

def embed_pos(model, world, pos):
    #world.robot.pos[:] = pos
    world.robot.pos[:] = [0.5, 0.5]

    with torch.no_grad():
        obs = world.get_obs()
        # obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
        # obs_embedding = model.embed_obs(
        #     torch.tensor(obs_tokens).unsqueeze(0).float().cuda(),
        #     torch.tensor(token_mask).unsqueeze(0).cuda(),
        #     torch.tensor(obs_categories).unsqueeze(0).float().cuda()
        # )
        # return obs_embedding[0].detach().cpu().numpy(), interaction_flag
        hidden_size = 32
        target_pos = np.array([0.5, 0.5])
        prev_latent = torch.zeros((1, hidden_size))
        obs, real_traj, latent_traj = control_robot_to(world, pos, model, obs, prev_latent)
        interaction_flag = 0
        if obs['pickup'] is not None:
            interaction_flag += 1
        if obs['drop'] is not None:
            interaction_flag += 2
        # Open-loop latent
        return latent_traj[-1][:, 0, :].numpy(), interaction_flag

if __name__ == "__main__":
    with open(os.path.join(SCRIPT_DIR, "world.json"), "r") as jf:
        data = json.load(jf)
    world = World2d(data)

    import sys
    seed = 42
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    np.random.seed(seed)
    world.reset()

    import mediapy
    img = world.render()
    mediapy.write_image('out_image.png', img)
    #exit(0)

    with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
        config = json.load(jf)
    model, latents = load_model(config)

    embeddings = np.empty((100, 100, config['hidden_dim']))
    interaction_flag = np.empty((100, 100), dtype=int)
    for i, xpos in enumerate(np.linspace(0, 1, 100)):
        for j, ypos in enumerate(np.linspace(0, 1, 100)):
            embeddings[i, j, :], interaction_flag[i, j] = embed_pos(model, world, [xpos, ypos])
    np.save(f"embeddings/{seed}/embeddings.npy", embeddings)
    np.save(f"embeddings/{seed}/interactions.npy", interaction_flag)
