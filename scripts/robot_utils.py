import numpy as np

import torch

from env_2d import tokenize_obs, World2d, MAX_TOKENS

hidden_size = 32

def model_update(model, latent, obs, action):
    obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
    with torch.no_grad():
        obs_emb, latents, obs_reconstruct, past_predictions = model(
            latent.cuda(),
            torch.tensor(obs_tokens).float().unsqueeze(0).cuda(),
            torch.tensor(token_mask).unsqueeze(0).cuda(),
            torch.tensor(obs_categories).unsqueeze(0).cuda(),
            action.unsqueeze(0).cuda()
        )
        return latents[:, 1, :]

def control_robot_to(world: World2d, target, model, prev_obs, prev_latent, speed_factor=1, interaction=0):
    chosen_speed = speed_factor * world.robot.max_speed
    latent_traj = []
    real_traj = []
    while np.linalg.norm(world.robot.pos - target) > 1e-5:
        motion = target - world.robot.pos
        distance = np.linalg.norm(motion)
        motion = motion * (min(chosen_speed, distance) / distance)

        action = torch.zeros(3)
        action[0:2] = torch.tensor(motion)
        action[2] = interaction
        interaction = 0

        prev_latent = model_update(model, prev_latent, prev_obs, action)
        latent_traj.append(prev_latent.detach().cpu())

        prev_obs = world.update(action)
        real_traj.append(prev_obs)
    # Current obs, real trajectory, latent trajectory
    return prev_obs, real_traj, latent_traj

def gen_sample(world: World2d, model, seed=0, container=0):
    np.random.seed(seed)

    obs = world.reset()

    init_pos = np.copy(world.robot.pos)
    target_pos1 = world.items[0].pos
    target_pos2 = world.containers[container].pos
    target_pos3 = (world.containers[0].pos + world.containers[1].pos) / 2

    prev_latent = torch.zeros((1, hidden_size))
    obs, real_traj1, latent_traj1 = control_robot_to(world, target_pos1, model, obs, prev_latent, interaction=0)
    obs, real_traj2, latent_traj2 = control_robot_to(world, target_pos2, model, obs, prev_latent, interaction=1)
    obs, real_traj3, latent_traj3 = control_robot_to(world, target_pos3, model, obs, prev_latent, interaction=-1)
    obs, real_traj4, latent_traj4 = control_robot_to(world, init_pos, model, obs, prev_latent, interaction=0)
    return (
        obs,
        real_traj1 + real_traj2 + real_traj3 + real_traj4,
        latent_traj1 + latent_traj2 + latent_traj3 + latent_traj4
    )
