import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLevelDQN(nn.Module):
    def __init__(self, state_dim=6, action_dim=10):  # state_dim=6 for state1, action_dim=max_slots
        super(HighLevelDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)  # output Q value for each possible slot count

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
