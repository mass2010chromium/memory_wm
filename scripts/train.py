from einops import rearrange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from memory_wm.module import SIGReg, Predictor

from env_2d import N_TYPES, OBS_SIZE
from env_2d_dataset import World2dDataset

# Reproducibility
torch.manual_seed(42)

dataset = World2dDataset(LeRobotDataset("local/world2d", root="./world2d"))
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 192
model = Predictor(hidden_dim=hidden_size, action_dim=3, input_dim=OBS_SIZE, categories=N_TYPES, depth=6, heads=16, mlp_dim=hidden_size).cuda()
sigreg = SIGReg().cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

latent_cache = torch.empty((len(dataset), hidden_size))
latents_filled = False

num_epochs = 500
use_temporal_straightening = True
if use_temporal_straightening:
    straightness_measure = torch.nn.CosineSimilarity()

with wandb.init(name="mini-wm") as run:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
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
            for i, (f, x) in enumerate(zip(active_frames, frame_index)):
                if x > 0:
                    prior_latents[i] = latent_cache[f-1]
                if use_temporal_straightening and x > 1:
                    prior_latents_2[i] = latent_cache[f-2]

            outputs = model(
                prior_latents.cuda(),
                data_batch['observation.tokens'].cuda(),   # x
                data_batch['observation.token_mask'].cuda(),
                data_batch['observation.token_categories'].cuda(),
                data_batch['action'].cuda()
            )        # forward pass

            if latents_filled:
                latent_targets = latent_cache[active_frames].cuda()
                pred_loss = (outputs - latent_targets).pow(2).mean()
            else:
                pred_loss = 0

            sigreg_loss = sigreg(outputs)
            # Copied from jepawm (lambda=0.09)
            loss = pred_loss + 0.09 * sigreg_loss
            if latents_filled and use_temporal_straightening:
                prior_latents = prior_latents.cuda()
                velocity = outputs - prior_latents
                prev_velocity = prior_latents - prior_latents_2.cuda()
                # Negative: We want it to be high (straight)
                straightness_loss = straightness_measure(velocity, prev_velocity).mean()
                loss -= straightness_loss
                running_curvature_loss += straightness_loss.item() * B

            outputs = outputs.detach().cpu()
            latent_cache[active_frames] = outputs

            loss.backward()                # backprop
            optimizer.step()               # update weights

            running_loss += loss.item() * B
            running_dynamics_loss += pred_loss.item() * B
            running_sigreg_loss += sigreg_loss.item() * B

        epoch_loss = running_loss / len(dataset)
        epoch_dynamics_loss = running_dynamics_loss / len(dataset)
        epoch_sigreg_loss = running_sigreg_loss / len(dataset)
        epoch_curvature_loss = running_curvature_loss / len(dataset)
        run.log({
            "loss": epoch_loss,
            "pred_loss": epoch_dynamics_loss,
            "sigreg_loss": epoch_sigreg_loss,
            "curvature_loss": epoch_curvature_loss,
        })
        print(f"Epoch {epoch+1}/{num_epochs} — loss: {epoch_loss:.4f}")
        latents_filled = True

torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "latent_cache": latent_cache
}, "model.save")

