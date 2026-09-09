import json
import os
SCRIPT_DIR = os.path.dirname(__file__)

from einops import rearrange, einsum
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import tqdm

import wandb

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from memory_wm.module import SIGReg, Predictor

from env_2d import N_TYPES, OBS_SIZE, SINGLE_ID, ACTION_SIZE
from env_2d_dataset import World2dDataset, SmallPackedDataset

from math_utils import slerp

# Reproducibility
torch.manual_seed(42)

out_dir = os.path.join(SCRIPT_DIR, "checkpoints_2")
os.makedirs(out_dir, exist_ok=True)
#dataset = World2dDataset(LeRobotDataset("local/world2d", root=os.path.join(SCRIPT_DIR, "world2d")))
dataset = SmallPackedDataset(root=os.path.join(SCRIPT_DIR, "world2d_reorder"))
batch_size = 1024
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
    config = json.load(jf)

hidden_size = config['hidden_dim']
obs_dim = config['obs_dim']
def init_model(model_config):
    model = Predictor(**model_config).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    latent_cache = torch.zeros((len(dataset), hidden_size)).to(device)
    observation_cache = torch.zeros((len(dataset), obs_dim)).to(device)
    return model, optimizer, latent_cache, observation_cache

def load_model(model_config, epoch):
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data = torch.load(os.path.join(out_dir, f"{epoch}.pth"), weights_only=True)

    model = Predictor(**model_config).to(device)
    model.load_state_dict(data['model_state'])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(data['optimizer_state'])
    model.train()
    return model, optimizer, data['latent_cache'], data['obs_cache']

sigreg = SIGReg().to(device)
#start_epoch = 25
#model, optimizer, latent_cache, observation_cache = load_model(config, start_epoch-1)
start_epoch = 0
model, optimizer, latent_cache, observation_cache = init_model(config)
obs_freeze_epoch = -1

all_actions = torch.tensor(dataset.data_map['action']).to(device)

num_epochs = 500
scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=num_epochs)
scheduler.step(start_epoch)
save_interval = 5

use_temporal_straightening = True
predict_past = False
if use_temporal_straightening:
    straightness_measure = torch.nn.CosineSimilarity()

def get_init_latent(data_batch):
    frame_index = data_batch['frame_index']
    active_frames = data_batch['index']
    first_mask = frame_index <= 0
    actions = all_actions[active_frames]

    prior_latents = latent_cache[active_frames - 1]

    obs_embed = model.embed_obs(
        data_batch['observation.tokens'].to(device),   # x
        data_batch['observation.token_mask'].to(device),
        data_batch['observation.token_categories'].to(device)
    )
    _latents = model.predict_latent(prior_latents, obs_embed, actions)

    latents = torch.zeros_like(_latents)    # Need new tensor for gradient tracking, instead of reassigning _latents components

    init_latents = model.init_state(observation_cache[active_frames[first_mask]])
    latents[first_mask, :, :] = init_latents.unsqueeze(1)   # Fake OL + CL latents so they get written into the latent cache.
    latents[torch.logical_not(first_mask), :, :] = _latents[torch.logical_not(first_mask), :, :]
    return first_mask, latents, obs_embed

def get_losses(latents, future_frames, match_obs, horizon_factor):
    future_factor = (future_frames > 0).unsqueeze(-1) * horizon_factor
    obs_reconstruct = model.reconstruction(latents[:, 0, :])
    pred_err = (match_obs - obs_reconstruct) * future_factor
    pred_loss = pred_err.pow(2).sum()

    cl_latents = latents[:, -1, :]
    ol_latents = latents[:, -2, :]
    latent_err = (ol_latents - cl_latents) * future_factor
    latent_pred_loss = latent_err.pow(2).sum()

    return pred_loss, latent_pred_loss, future_factor.sum()

def rollout_latents(latents, active_frames, closed_loop=0):
    obs_emb = observation_cache[active_frames]
    actions = all_actions[active_frames]
    # 0 for openloop rollout, 1 for closed loop rollout
    next_latents = model.predict_latent(latents[:, closed_loop, :], obs_emb, actions)
    return next_latents, obs_emb


#run = None
with wandb.init(name="mini-wm-multistep-openloop") as run:
#if True:
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_dynamics_loss = 0.0
        running_sigreg_loss = torch.zeros(4)
        running_curvature_loss = 0.0
        running_drift_mag = torch.zeros(2)

        next_latent_cache = torch.zeros_like(latent_cache)
        next_observation_cache = torch.zeros_like(observation_cache)
        horizon_weight = torch.tensor(0.0, device=device)

        for batch_idx, data_batch in enumerate(tqdm.tqdm(dataloader)):
            optimizer.zero_grad()          # clear gradients
            frame_index = data_batch['frame_index']
            active_frames = data_batch['index']
            B = len(frame_index)

            # State based initialization
            prior_latents = latent_cache[active_frames - 1]
            prior_latents[frame_index <= 0] = 0

            prior_latents_2 = latent_cache[active_frames - 2]
            prior_latents_2[frame_index <= 1] = 0

            first_mask, latents, obs_emb = get_init_latent(data_batch)

            cl_latents = latents[:, -1, :]
            ol_latents = latents[:, -2, :]

            horizon_factor = torch.minimum(0.9 ** (frame_index - epoch/10), torch.tensor(1.0)).unsqueeze(-1).to(device)

            future_frames = data_batch['future_frames'].to(device)
            # Special: This on uses live computed observations. Others use obs_emb cache... is this a problem?
            pred_loss, latent_pred_loss, weight = get_losses(latents, future_frames, obs_emb, horizon_factor)
            horizon_weight += weight

            predict_horizon = 5
            future_obs = obs_emb
            future_latents = latents
            for i in range(1, predict_horizon):
                future_latents, future_obs = rollout_latents(future_latents, (active_frames + i) % len(all_actions), closed_loop=0)
                future_pred_loss, future_latent_loss, weight = get_losses(future_latents, future_frames - i, future_obs, horizon_factor)
                horizon_weight += weight
                pred_loss += future_pred_loss
                latent_pred_loss += future_latent_loss

            future_obs = obs_emb
            future_latents = latents
            for i in range(1, predict_horizon):
                future_latents, future_obs = rollout_latents(future_latents, (active_frames + i) % len(all_actions), closed_loop=1)
                future_pred_loss, future_latent_loss, weight = get_losses(future_latents, future_frames - i, future_obs, horizon_factor)
                horizon_weight += weight
                pred_loss += future_pred_loss
                latent_pred_loss += future_latent_loss

            # SIGReg is needed on velocities to ensure the distribution doesn't collapse to uniform+discrete
            # This might be huge... ask Devesh
            velocity = cl_latents - prior_latents

            prior_obs = observation_cache[active_frames - 1]
            prior_obs[first_mask] = 0
            obs_velocity = obs_emb - prior_obs

            sigreg_losses = [0.1*sigreg(cl_latents), sigreg(obs_emb), 0.1*sigreg(10*velocity), sigreg(10*obs_velocity)]
            sigreg_loss = sum(sigreg_losses)

            # Full loss (reconstruction and dynamics)
            # Copied from jepawm (lambda=0.09)
            loss = (5/B)*pred_loss + (0.5/B)*latent_pred_loss + 0.09 * sigreg_loss
            outputs = cl_latents

            if use_temporal_straightening:
                prev_velocity = prior_latents - prior_latents_2
                # Negative: We want it to be high (straight)
                straightness_loss = straightness_measure(velocity, prev_velocity).mean()
                loss -= straightness_loss
                running_curvature_loss += straightness_loss.item() * B

            # Constrain the outputs of obs_embedding and latent embedding modules to match the existing ones.
            drift_factor = max(0.01, 0.99**(epoch))
            consistency_losses = [
                (outputs - latent_cache[active_frames]).pow(2).mean(),
                (obs_emb - observation_cache[active_frames]).pow(2).mean()
            ]

            loss = loss * drift_factor + sum(consistency_losses) * (1 - drift_factor)

            outputs = outputs.detach()
            obs_emb = obs_emb.detach()

            if epoch == 0:
                next_latent_cache[active_frames] = outputs
                next_observation_cache[active_frames] = obs_emb
            else:
                next_latent_cache[active_frames] = slerp(latent_cache[active_frames], outputs, drift_factor)
                next_observation_cache[active_frames] = obs_emb

            loss.backward()                # backprop
            optimizer.step()               # update weights

            running_loss += loss.item() * B
            running_reconstruction_loss += pred_loss.item()
            running_dynamics_loss += latent_pred_loss.item()
            running_sigreg_loss += torch.tensor(sigreg_losses).detach().cpu() * B
            running_drift_mag += torch.tensor(consistency_losses).detach().cpu() * B

        scheduler.step(epoch+1)
        latent_cache = next_latent_cache
        observation_cache = next_observation_cache

        epoch_loss = running_loss / len(dataset)
        epoch_reconstruction_loss = running_reconstruction_loss / horizon_weight.item()
        epoch_dynamics_loss = running_dynamics_loss / horizon_weight.item()
        epoch_sigreg_loss = running_sigreg_loss / len(dataset)
        epoch_curvature_loss = running_curvature_loss / len(dataset)
        epoch_drift_mag = running_drift_mag / len(dataset)
        latents_norm = torch.norm(latent_cache, dim=1).mean()
        obs_norm = torch.norm(observation_cache, dim=1).mean()
        if run:
            run.log({
                "loss": epoch_loss,
                "obs_loss": epoch_reconstruction_loss,
                "pred_loss": epoch_dynamics_loss,
                "sigreg_loss_state": epoch_sigreg_loss[0],
                "sigreg_loss_obs": epoch_sigreg_loss[1],
                "sigreg_loss_state_vel": epoch_sigreg_loss[2],
                "sigreg_loss_obs_vel": epoch_sigreg_loss[3],
                "curvature_loss": epoch_curvature_loss,
                "latent_drift": epoch_drift_mag[0],
                "obs_drift": epoch_drift_mag[1],
                "latent_norm": latents_norm,
                "obs_norm": obs_norm,
            })
        print(f"Epoch {epoch+1}/{num_epochs} — loss: {epoch_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "latent_cache": latent_cache,
                "obs_cache": observation_cache
            }, os.path.join(out_dir, f"{epoch}.pth"))
