import torch
import torch.nn as nn

class LowLevelDQN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, max_vehicles=10):
        super(LowLevelDQN, self).__init__()
        self.max_vehicles = max_vehicles

        # 車輛特徵進 hidden
        self.vehicle_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 聚合所有車輛後的 decision layer
        self.decision_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        """
        obs: tensor, shape (batch_size, max_vehicles, 6)
        returns: tensor, shape (batch_size, max_vehicles)
        """
        batch_size = obs.shape[0]

        vehicle_feat = self.vehicle_fc(obs)  # shape: (batch_size, max_vehicles, hidden_dim)
        logits = self.decision_fc(vehicle_feat).squeeze(-1)  # shape: (batch_size, max_vehicles)

        return logits  # raw logits，之後用 sigmoid



