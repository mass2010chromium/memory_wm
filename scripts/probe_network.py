import torch
import torch.nn as nn

class MLPProbe(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim*2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim*2, hidden_dim*2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim*2, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
 
    def forward(self, x):
        return self.net(x)

