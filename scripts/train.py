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

from env_2d import N_TYPES, OBS_SIZE, SINGLE_ID
from env_2d_dataset import World2dDataset, SmallPackedDataset

# Reproducibility
torch.manual_seed(42)

out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(out_dir, exist_ok=True)
#dataset = World2dDataset(LeRobotDataset("local/world2d", root=os.path.join(SCRIPT_DIR, "world2d")))
dataset = SmallPackedDataset(root=os.path.join(SCRIPT_DIR, "world2d_reorder"))
batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
    config = json.load(jf)

hidden_size = config['hidden_dim']
def init_model(model_config):
    model = Predictor(**model_config).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    latent_cache = torch.empty((len(dataset), hidden_size))
    return model, optimizer, latent_cache

def load_model(model_config, epoch):
    out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data = torch.load(os.path.join(out_dir, f"{epoch}.pth"), weights_only=True)

    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(data['optimizer_state'])
    model.train()
    return model, optimizer, data['latent_cache']

sigreg = SIGReg().cuda()
start_epoch = 300
model, optimizer, latent_cache = load_model(config, start_epoch-1)
#start_epoch = 0
#model, optimizer, latent_cache = init_model(config)


num_epochs = 500
scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=num_epochs)
scheduler.step(start_epoch)
save_interval = 50
use_temporal_straightening = True
if use_temporal_straightening:
    straightness_measure = torch.nn.CosineSimilarity()

SINGLE_TOKEN = torch.zeros((1, 1, OBS_SIZE)).cuda()
SINGLE_CATEGORY = rearrange(
    torch.eye(N_TYPES, dtype=torch.bool)[SINGLE_ID].cuda(),
    "b -> 1 1 b"
)
TOKEN_MASK = torch.tensor([[1]], dtype=torch.bool).cuda()
with wandb.init(name="mini-wm") as run:
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_past_loss = 0.0
        running_past_loss_2 = 0.0
        running_past_loss_3 = 0.0
        running_dynamics_loss = 0.0
        running_sigreg_loss = 0.0
        running_curvature_loss = 0.0

        for data_batch in tqdm.tqdm(dataloader):
            optimizer.zero_grad()          # clear gradients
            B = len(data_batch['frame_index'])
            frame_index = data_batch['frame_index']
            prior_latents = torch.zeros((B, hidden_size), dtype=torch.float32)

            if use_temporal_straightening:
                prior_latents_2 = torch.zeros((B, hidden_size), dtype=torch.float32)

            active_frames = data_batch['index']
            prior_latents = latent_cache[active_frames - 1].cuda()
            prior_latents[frame_index <= 0] = 0

            prior_latents_2 = latent_cache[active_frames - 2].cuda()
            prior_latents_2[frame_index <= 1] = 0

            prior_latents_3 = latent_cache[active_frames - 3].cuda()
            prior_latents_3[frame_index <= 2] = 0

            actions = data_batch['action'].cuda()

            obs_emb, latents, obs_reconstruct, past_predictions = model(
                prior_latents,
                data_batch['observation.tokens'].cuda(),   # x
                data_batch['observation.token_mask'].cuda(),
                data_batch['observation.token_categories'].cuda(),
                actions
            )
            past_error = past_predictions[:, 0] - prior_latents_2
            past_error[frame_index <= 1] = 0
            past_error_3 = past_predictions[:, 1] - prior_latents_3
            past_error_3[frame_index <= 2] = 0
            past_loss_2 = past_error.pow(2).mean()
            past_loss_3 = past_error_3.pow(2).mean()
            past_loss = past_loss_2 + past_loss_3

            pred_loss = (obs_emb - obs_reconstruct).pow(2).mean()

            cl_latents = latents[:, 1, :]
            ol_latents = latents[:, 0, :]
            latent_pred_loss = (ol_latents - cl_latents).pow(2).mean()
            sigreg_loss = sigreg(obs_emb) + sigreg(cl_latents)
            # NOTE: should this be averaged or something?
            # Copied from jepawm (lambda=0.09), halved for double counting
            loss = pred_loss + latent_pred_loss + past_loss + 0.09 * sigreg_loss
            if use_temporal_straightening:
                velocity = cl_latents - prior_latents
                prev_velocity = prior_latents - prior_latents_2
                # Negative: We want it to be high (straight)
                straightness_loss = straightness_measure(velocity, prev_velocity).mean()
                loss -= straightness_loss * 5
                running_curvature_loss += straightness_loss.item() * B

            outputs = cl_latents.detach().cpu()
            if epoch == 0:
                latent_cache[active_frames] = outputs
            else:
                latent_cache[active_frames] = latent_cache[active_frames] * 0.9 + outputs * 0.1

            loss.backward()                # backprop
            optimizer.step()               # update weights

            running_loss += loss.item() * B
            running_reconstruction_loss += latent_pred_loss.item() * B
            running_past_loss += past_loss.item() * B
            running_past_loss_2 += past_loss_2.item() * B
            running_past_loss_3 += past_loss_3.item() * B
            running_dynamics_loss += pred_loss.item() * B
            running_sigreg_loss += sigreg_loss.item() * B

        scheduler.step(epoch+1)

        epoch_loss = running_loss / len(dataset)
        epoch_reconstruction_loss = running_reconstruction_loss / len(dataset)
        epoch_past_loss = running_past_loss / len(dataset)
        epoch_past_loss_2 = running_past_loss_2 / len(dataset)
        epoch_past_loss_3 = running_past_loss_3 / len(dataset)
        epoch_dynamics_loss = running_dynamics_loss / len(dataset)
        epoch_sigreg_loss = running_sigreg_loss / len(dataset)
        epoch_curvature_loss = running_curvature_loss / len(dataset)
        latents_norm = torch.norm(latent_cache, dim=1).mean()
        run.log({
            "loss": epoch_loss,
            "obs_loss": epoch_reconstruction_loss,
            "pred_loss": epoch_dynamics_loss,
            "past_loss": epoch_past_loss,
            "past_loss_2": epoch_past_loss_2,
            "past_loss_3": epoch_past_loss_3,
            "sigreg_loss": epoch_sigreg_loss,
            "curvature_loss": epoch_curvature_loss,
            "latent_norm": latents_norm,
        })
        print(f"Epoch {epoch+1}/{num_epochs} — loss: {epoch_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "latent_cache": latent_cache
            }, os.path.join(out_dir, f"{epoch}.pth"))
