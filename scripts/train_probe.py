import os
SCRIPT_DIR = os.path.dirname(__file__)

from einops import rearrange
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

from probe_network import MLPProbe

torch.manual_seed(0)

positions = torch.tensor(np.load(os.path.join(SCRIPT_DIR, "probe_positions.npy"))).cuda()
latents = rearrange(torch.tensor(np.load(os.path.join(SCRIPT_DIR, "probe_latents.npy"))), "b 1 n -> b n").cuda()
positions_val = torch.tensor(np.load(os.path.join(SCRIPT_DIR, "probe_positions_val.npy"))).cuda()
latents_val = rearrange(torch.tensor(np.load(os.path.join(SCRIPT_DIR, "probe_latents_val.npy"))), "b 1 n -> b n").cuda()

model = MLPProbe().cuda()

def get_accuracy(pred_positions, actual_positions):
    pos_pairs = rearrange(actual_positions, "(b n) d -> n b d", n=2)
    pred_pairs = rearrange(pred_positions, "(b n) d -> n b d", n=2)
    target_vectors = pos_pairs[1] - pos_pairs[0]
    pred_vec = pred_pairs[1] - pred_pairs[0]
    dots = (pred_vec * target_vectors).sum(dim=1)
    return (dots > 0).sum() / len(dots)
    # distance_0_correct = torch.norm(pos_pairs[0] - pred_pairs[0], dim=1)
    # distance_0_wrong = torch.norm(pos_pairs[1] - pred_pairs[0], dim=1)
    # distance_1_correct = torch.norm(pos_pairs[1] - pred_pairs[1], dim=1)
    # distance_1_wrong = torch.norm(pos_pairs[0] - pred_pairs[0], dim=1)
    # score_0 = distance_0_wrong - distance_0_correct
    # score_1 = distance_1_wrong - distance_1_correct
    # correct_0 = torch.sum(distance_0_correct < distance_0_wrong)
    # correct_1 = torch.sum(distance_1_correct < distance_1_wrong)
    # return (correct_0 + correct_1) / len(pred_positions)

def get_loss_contrastive(pred_positions, actual_positions):
    pos_pairs = rearrange(actual_positions, "(b n) d -> n b d", n=2)
    pred_pairs = rearrange(pred_positions, "(b n) d -> n b d", n=2)
    target_vectors = pos_pairs[1] - pos_pairs[0]
    target_vectors /= torch.norm(target_vectors, dim=1, keepdim=True)
    pred_vec = pred_pairs[1] - pred_pairs[0]
    dots = (pred_vec * target_vectors).sum(dim=1)
    dots[dots > 0] = torch.log(dots[dots > 0] + 1)
    distance = torch.norm(pred_positions - actual_positions, dim=1)
    return -dots.mean() + distance.mean()
    

n_epochs = 10000
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=n_epochs)
 
for epoch in tqdm.trange(1, n_epochs + 1):
    model.train()
    running_loss = 0.0

    optimizer.zero_grad()       # clear gradients from previous step
    pred_positions = model(latents)          # forward pass
    loss = get_loss_contrastive(pred_positions, positions)
    #loss = (pred_positions - positions).pow(2).mean()
    #loss = (pred_positions - positions).abs().mean()
    loss.backward()             # backprop
    optimizer.step()            # update weights
    #scheduler.step()

    running_loss = loss.item()
 
    train_loss = running_loss

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            pred_positions_val = model(latents_val)
            val_error = get_loss_contrastive(pred_positions_val, positions_val)
            #val_error = (pred_positions_val - positions_val).pow(2).mean()
            #val_error = (pred_positions_val - positions_val).abs().mean()
            val_accuracy = get_accuracy(pred_positions_val, positions_val)
        print(f"Epoch {epoch:2d} | train err: {train_loss:.4f} | val err: {val_error} | val acc: {val_accuracy}")


torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "probe.pth"))
