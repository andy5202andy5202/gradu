import torch
import torch.nn as nn
import torch.nn.functional as F

class LowLevelDQN(nn.Module):
    """
    兩頭 Q 值：對每台車輸出 Q(s, a=0/1)
    obs shape: (B, V, 6)
    output:    (B, V, 2)
    """
    def __init__(self, input_dim=6, hidden_dim=128, max_vehicles=10):
        super().__init__()
        self.max_vehicles = max_vehicles

        self.vehicle_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_dim, 2)  # [不選, 選]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, V, 6)
        feat = self.vehicle_fc(obs)   # (B, V, H)
        q = self.q_head(feat)         # (B, V, 2)
        return q
