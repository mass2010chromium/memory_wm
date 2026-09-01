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

# Reproducibility
torch.manual_seed(42)

out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(out_dir, exist_ok=True)
#dataset = World2dDataset(LeRobotDataset("local/world2d", root=os.path.join(SCRIPT_DIR, "world2d")))
dataset = SmallPackedDataset(root=os.path.join(SCRIPT_DIR, "world2d_reorder"))
batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
    config = json.load(jf)

hidden_size = config['hidden_dim']
obs_dim = config['obs_dim']
def init_model(model_config):
    model = Predictor(**model_config).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    latent_cache = torch.empty((len(dataset), hidden_size))
    observation_cache = torch.empty((len(dataset), obs_dim))
    return model, optimizer, latent_cache, observation_cache

def load_model(model_config, epoch):
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data = torch.load(os.path.join(out_dir, f"{epoch}.pth"), weights_only=True)

    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(data['optimizer_state'])
    model.train()
    return model, optimizer, data['latent_cache'], data['obs_cache']

sigreg = SIGReg().cuda()
#start_epoch = 25
#model, optimizer, latent_cache, observation_cache = load_model(config, start_epoch-1)
start_epoch = 0
model, optimizer, latent_cache, observation_cache = init_model(config)
obs_freeze_epoch = -1

all_actions = torch.tensor(dataset.data_map['action'])

num_epochs = 500
scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=num_epochs)
scheduler.step(start_epoch)
save_interval = 5

use_temporal_straightening = True
predict_past = False
if use_temporal_straightening:
    straightness_measure = torch.nn.CosineSimilarity()

#run = None
with wandb.init(name="mini-wm-no-state") as run:
#if True:
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_past_loss = 0.0
        running_past_loss_2 = 0.0
        running_past_loss_8 = 0.0
        running_dynamics_loss = 0.0
        running_sigreg_loss = 0.0
        running_curvature_loss = 0.0

        for batch_idx, data_batch in enumerate(tqdm.tqdm(dataloader)):
            optimizer.zero_grad()          # clear gradients
            B = len(data_batch['frame_index'])
            frame_index = data_batch['frame_index']
            #prior_latents = torch.zeros((B, hidden_size), dtype=torch.float32)

            #if use_temporal_straightening:
            #    prior_latents_2 = torch.zeros((B, hidden_size), dtype=torch.float32)

            active_frames = data_batch['index']
            #prior_latents = latent_cache[active_frames - 1].cuda()
            #prior_latents[frame_index <= 0] = 0
            # Problem: first frame badness.
            # Solution: Mask out first frame
            prior_latents = model.init_state(observation_cache[active_frames-1].cuda())
            prior_latents[frame_index <= 0] = 0

            prior_latents_2 = latent_cache[active_frames - 2].cuda()
            prior_latents_2[frame_index <= 1] = 0

            actions = data_batch['action'].cuda()

            if obs_freeze_epoch < 0 or epoch < obs_freeze_epoch:
                obs_emb, latents, obs_reconstruct = model(
                    prior_latents,
                    data_batch['observation.tokens'].cuda(),   # x
                    data_batch['observation.token_mask'].cuda(),
                    data_batch['observation.token_categories'].cuda(),
                    actions
                )
            elif epoch == obs_freeze_epoch:
                # NOTE: Grad is stopped globally in this iteration.
                obs_emb = model.embed_obs(
                    data_batch['observation.tokens'].cuda(),   # x
                    data_batch['observation.token_mask'].cuda(),
                    data_batch['observation.token_categories'].cuda(),
                )
                latents = model.predict_latent(prior_latents, obs_emb, actions)
                # Action conditioned, but not next-observation conditioned.
                obs_reconstruct = model.reconstruction(latents[:, 0, :])
            else:
                # No need to recompute observations, even
                obs_emb = observation_cache[active_frames].cuda()
                latents = model.predict_latent(prior_latents, obs_emb, actions)
                # Action conditioned, but not next-observation conditioned.
                obs_reconstruct = model.reconstruction(latents[:, 0, :])

            if predict_past:
                prior_latents_3 = latent_cache[active_frames - 3].cuda()
                prior_obs_2 = observation_cache[active_frames - 2].cuda()
                prior_action_2 = all_actions[active_frames - 2].cuda()
                pred_prior_latents_2 = model.predict_latent(prior_latents_3, prior_obs_2, prior_action_2)[:, 1]

                prior_latents_9 = latent_cache[active_frames - 9].cuda()
                prior_obs_8 = observation_cache[active_frames - 8].cuda()
                prior_action_8 = all_actions[active_frames - 8].cuda()
                pred_prior_latents_8 = model.predict_latent(prior_latents_9, prior_obs_8, prior_action_8)[:, 1]

                past_error = past_predictions[:, 0] - pred_prior_latents_2
                past_error[frame_index <= 2] = 0
                past_error_8 = past_predictions[:, 1] - pred_prior_latents_8
                past_error_8[frame_index <= 8] = 0
                past_loss_2 = past_error.pow(2).mean()
                past_loss_8 = past_error_8.pow(2).mean()
                past_loss = past_loss_2 + past_loss_8
            else:
                past_loss = 0

            cl_latents = latents[:, -1, :]
            ol_latents = latents[:, -2, :]

            # Open-loop closed-loop latent formulation
            pred_err = obs_emb - obs_reconstruct
            pred_err[frame_index <= 0] = 0
            pred_loss = pred_err.abs().mean()#pow(2).mean()
            latent_err = ol_latents - cl_latents
            latent_err[frame_index <= 0] = 0
            latent_pred_loss = latent_err.abs().mean()#pow(2).mean()

            # SIGReg is needed on velocities to ensure the distribution doesn't collapse to uniform+discrete
            # This might be huge... ask Devesh
            velocity = cl_latents - prior_latents

            sigreg_loss = sigreg(obs_emb) + sigreg(cl_latents) + sigreg(10*velocity)

            # Full loss (reconstruction and dynamics)
            # Copied from jepawm (lambda=0.09)
            loss = 2*pred_loss + 0.5*latent_pred_loss + 0.2 * past_loss + 0.09 * sigreg_loss
            # Ablation: No past loss version, only sigreg and reconstruction
            # loss = pred_loss + latent_pred_loss + 0.09 * sigreg_loss
            # Ablation: No reconstruction loss version, only sigreg
            #loss = latent_pred_loss + past_loss + 0.09 * sigreg_loss
            outputs = cl_latents

            # Observation embedding formulation (JEPA), no closed-loop latent
            #pred_loss = (ol_latents - obs_emb).pow(2).mean()
            #sigreg_loss = sigreg(obs_emb)
            # Only reconstruction loss
            #loss = pred_loss + past_loss + 0.09 * sigreg_loss
            #outputs = obs_emb

            if use_temporal_straightening:
                prev_velocity = prior_latents - prior_latents_2
                # Negative: We want it to be high (straight)
                straightness_loss = straightness_measure(velocity, prev_velocity).mean()
                loss -= straightness_loss
                running_curvature_loss += straightness_loss.item() * B

            outputs = outputs.detach().cpu()
            obs_emb = obs_emb.detach().cpu()
            if epoch == 0:
                latent_cache[active_frames] = outputs
                observation_cache[active_frames] = obs_emb
            else:
                latent_cache[active_frames] = latent_cache[active_frames] * 0.9 + outputs * 0.1
                observation_cache[active_frames] = observation_cache[active_frames] * 0.9 + obs_emb * 0.1

            if epoch == obs_freeze_epoch:
                observation_cache[active_frames] = obs_emb
            else:
                loss.backward()                # backprop
                optimizer.step()               # update weights

            running_loss += loss.item() * B
            running_reconstruction_loss += pred_loss.item() * B
            running_dynamics_loss += latent_pred_loss.item() * B
            running_sigreg_loss += sigreg_loss.item() * B
            if predict_past:
                running_past_loss += past_loss.item() * B
                running_past_loss_2 += past_loss_2.item() * B
                running_past_loss_8 += past_loss_8.item() * B

        scheduler.step(epoch+1)

        epoch_loss = running_loss / len(dataset)
        epoch_reconstruction_loss = running_reconstruction_loss / len(dataset)
        epoch_past_loss = running_past_loss / len(dataset)
        epoch_past_loss_2 = running_past_loss_2 / len(dataset)
        epoch_past_loss_8 = running_past_loss_8 / len(dataset)
        epoch_dynamics_loss = running_dynamics_loss / len(dataset)
        epoch_sigreg_loss = running_sigreg_loss / len(dataset)
        epoch_curvature_loss = running_curvature_loss / len(dataset)
        latents_norm = torch.norm(latent_cache, dim=1).mean()
        if run:
            run.log({
                "loss": epoch_loss,
                "obs_loss": epoch_reconstruction_loss,
                "pred_loss": epoch_dynamics_loss,
                "past_loss": epoch_past_loss,
                "past_loss_2": epoch_past_loss_2,
                "past_loss_8": epoch_past_loss_8,
                "sigreg_loss": epoch_sigreg_loss,
                "curvature_loss": epoch_curvature_loss,
                "latent_norm": latents_norm,
            })
        else:
            pass
            #if epoch == obs_freeze_epoch:
            #    print("A", observation_cache[0])
            #if epoch == obs_freeze_epoch+1:
            #    print("B", observation_cache[0])
        print(f"Epoch {epoch+1}/{num_epochs} — loss: {epoch_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "latent_cache": latent_cache,
                "obs_cache": observation_cache
            }, os.path.join(out_dir, f"{epoch}.pth"))
