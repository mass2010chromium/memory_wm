import os
import sys
SCRIPT_DIR = os.path.dirname(__file__)

if len(sys.argv) > 1:
    ROOT_DIR = os.path.expanduser(sys.argv[1])
else:
    ROOT_DIR = SCRIPT_DIR

from einops import einsum
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

embedding_seed = 42
if embedding_seed is None:
    embeddings = np.load("embeddings.npy")
    interactions = np.load("interactions.npy")
else:
    embeddings = np.load(f"embeddings/{embedding_seed}/embeddings.npy")
    interactions = np.load(f"embeddings/{embedding_seed}/interactions.npy")
pickup = (interactions & 1) > 0
drop = (interactions & 2) > 0

from probe_network import MLPProbe
def train_probe(embeddings, interactions):
    model = MLPProbe(out_dim=4).cuda()

    data = torch.tensor(embeddings.reshape((-1, embeddings.shape[-1])), dtype=torch.float32).cuda()
    pickup = (interactions & 1) > 0
    drop = (interactions & 2) > 0
    supervision = torch.tensor(np.stack([pickup, drop], axis=-1).reshape(-1, 2), dtype=torch.float32).cuda()
    neg_supervision = 1.0 - supervision

    n_epochs = 10000
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    #scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=n_epochs)

    best_val_err = np.inf
    best_val_acc = 0.0
    best_val_iter = 0
     
    for epoch in tqdm.trange(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()       # clear gradients from previous step
        logits = model(data)        # forward pass, output is log-likelihood
        loss = -torch.sum(supervision * pred_state + neg_supervision * neg_state)
        loss.backward()             # backprop
        optimizer.step()            # update weights
        #scheduler.step()

        running_loss = loss.item()

        model.eval()
        with torch.no_grad():
            bin_pred_state = (pred_state > 0.5).float() # Binarize
            acc = (bin_pred_state * supervision + (1 - bin_pred_state) * neg_supervision).sum() / (bin_pred_state.shape[0] * 2)
            if acc > best_val_acc:
                best_val_err = running_loss 
                best_val_acc = acc
                best_val_iter = epoch
                torch.save(model.state_dict(), os.path.join(ROOT_DIR, "best_interaction.pth"))

        if epoch % 100 == 0:
            print(f"Epoch {epoch:2d} | train err: {running_loss:.4f} val acc: {best_val_acc}")


    print(f"Best: epoch {best_val_iter} err {best_val_err} acc {best_val_acc}")
    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "probe_interaction.pth"))

train_probe(embeddings, interactions)

pickups = embeddings[pickup]
not_pickups = embeddings[np.logical_not(pickup)]
drops = embeddings[drop]
not_drops = embeddings[np.logical_not(drop)]

def linear_classifier(positives, negatives, direction=None):
    if direction is None:
        direction = positives.mean(axis=0) - negatives.mean(axis=0)
        direction /= np.linalg.norm(direction)

    positives = einsum(positives, direction, 'n x, x -> n')
    negatives = einsum(negatives, direction, 'n x, x -> n')
    true_positive = np.sum(positives > 0)
    true_negative = np.sum(negatives < 0)
    print(f"True positives: {true_positive}/{len(positives)} ({true_positive / len(positives):.5f})")
    print(f"True negatives: {true_negative}/{len(negatives)} ({true_negative / len(negatives):.5f})")
    return direction

load_seed = None
if load_seed is None:
    pickup_vec = None
    drop_vec = None
else:
    pickup_vec = np.load(f"embeddings/{load_seed}/pickup_vec.npy")
    drop_vec = np.load(f"embeddings/{load_seed}/drop_vec.npy")

print(f"Embedding seed={embedding_seed}, Load seed={load_seed}")
print("Pickup available classifier:")
pickup_vec = linear_classifier(pickups, not_pickups, direction=pickup_vec)
print("Drop available classifier:")
drop_vec = linear_classifier(drops, not_drops, direction=drop_vec)

if load_seed is None:
    np.save(f"embeddings/{embedding_seed}/pickup_vec.npy", pickup_vec)
    np.save(f"embeddings/{embedding_seed}/drop_vec.npy", drop_vec)
