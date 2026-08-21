import os
import sys
SCRIPT_DIR = os.path.dirname(__file__)

if len(sys.argv) > 1:
    ROOT_DIR = os.path.expanduser(sys.argv[1])
else:
    ROOT_DIR = SCRIPT_DIR

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

from probe_network import MLPProbe

def train_probe(embeddings, val_embeddings):
    model = MLPProbe(out_dim=2).cuda()

    data = torch.tensor(embeddings.reshape((-1, embeddings.shape[-1])), dtype=torch.float32).cuda()
    val_data = torch.tensor(embeddings.reshape((-1, val_embeddings.shape[-1])), dtype=torch.float32).cuda()

    gt = []
    for xpos in np.linspace(0, 1, 100):
        for ypos in np.linspace(0, 1, 100):
            gt.append([xpos, ypos])
    gt = torch.tensor(gt, dtype=torch.float32).cuda()

    n_epochs = 10000
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    #scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=n_epochs)

    best_val_err = np.inf
    best_val_iter = 0
     
    for epoch in tqdm.trange(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()       # clear gradients from previous step
        pred_positions = model(data)          # forward pass
        loss = (pred_positions - gt).pow(2).mean()
        loss.backward()             # backprop
        optimizer.step()            # update weights
        #scheduler.step()

        running_loss = loss.item()

        model.eval()
        with torch.no_grad():
            pred_positions = model(val_data)
            val_err = (pred_positions - gt).pow(2).mean()
            if val_err < best_val_err:
                best_val_err = val_err
                best_val_iter = epoch
                torch.save(model.state_dict(), os.path.join(ROOT_DIR, "best.pth"))

        if epoch % 100 == 0:
            print(f"Epoch {epoch:2d} | train err: {running_loss:.4f} val err: {val_err:.4f}")


    print(f"Best: epoch {best_val_iter} err {best_val_err}")
    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "probe.pth"))

if __name__ == "__main__":
    train_seed = 42
    if train_seed is None:
        train_embeddings = np.load("embeddings.npy")
    else:
        train_embeddings = np.load(f"embeddings/{train_seed}/embeddings.npy")
    val_seed = 43
    if val_seed is None:
        val_embeddings = np.load("embeddings.npy")
    else:
        val_embeddings = np.load(f"embeddings/{val_seed}/embeddings.npy")

    train_probe(train_embeddings, val_embeddings)
    #test_probe(embeddings)
