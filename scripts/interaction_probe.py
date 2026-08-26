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
    #embeddings2 = np.load(f"embeddings/{45}/embeddings.npy")
    #interactions2 = np.load(f"embeddings/{45}/interactions.npy")
    #embeddings = np.stack((embeddings, embeddings2), axis=0)
    #interactions = np.stack((interactions, interactions2), axis=0)
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

    n_epochs = 1000
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
        pickup_pred = torch.nn.functional.softmax(logits[:, :2], dim=-1)
        drop_pred = torch.nn.functional.softmax(logits[:, 2:], dim=-1)
        pred_state = torch.stack([pickup_pred[:, 0], drop_pred[:, 0]], dim=1)
        neg_state = 1.0 - pred_state
        eps = 1e-6
        loss = -torch.sum(supervision * torch.log(pred_state + eps) + neg_supervision * torch.log(neg_state + eps))
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
                torch.save(model.state_dict(), os.path.join(ROOT_DIR, "embeddings", str(embedding_seed), "best_interaction.pth"))

        if epoch % 100 == 0:
            print(f"Epoch {epoch:2d} | train err: {running_loss:.4f} val acc: {best_val_acc}")


    print(f"Best: epoch {best_val_iter} err {best_val_err} acc {best_val_acc}")
    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "embeddings", str(embedding_seed), "probe_interaction.pth"))
    return model

model = train_probe(embeddings, interactions)

def test_probe(model, embeddings, interactions):
    pickup = (interactions & 1) > 0
    drop = (interactions & 2) > 0
    supervision = torch.tensor(np.stack([pickup, drop], axis=-1).reshape(-1, 2), dtype=torch.float32).cuda()

    data = torch.tensor(embeddings.reshape((-1, embeddings.shape[-1])), dtype=torch.float32).cuda()
    model.eval()
    with torch.no_grad():
        logits = model(data)        # forward pass, output is log-likelihood
        pickup_pred = torch.nn.functional.softmax(logits[:, :2], dim=-1)
        drop_pred = torch.nn.functional.softmax(logits[:, 2:], dim=-1)
        pred_state = torch.stack([pickup_pred[:, 0], drop_pred[:, 0]], dim=1)
        bin_pred_state = (pred_state > 0.5).float() # Binarize

        positives = bin_pred_state * supervision
        negatives = (1 - bin_pred_state) * (1 - supervision)

        true_positive = positives.sum()
        total_positive = supervision.sum()
        true_negative = negatives.sum()
        total_negative = (1 - supervision).sum()
        print(f"True positives: {true_positive}/{total_positive} ({true_positive / total_positive:.5f})")
        print(f"True negatives: {true_negative}/{total_negative} ({true_negative / total_negative:.5f})")

load_seed = 43
if load_seed is None:
    pass
else:
    val_embeddings = np.load(f"embeddings/{load_seed}/embeddings.npy")
    val_interactions = np.load(f"embeddings/{load_seed}/interactions.npy")

print(f"Embedding seed={embedding_seed}, Load seed={load_seed}")
test_probe(model, val_embeddings, val_interactions)
