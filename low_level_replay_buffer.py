import random
import torch

class LowLevelReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, obs, action, reward, next_obs, done):
        transition = {
            'obs': obs,                # tensor (max_vehicles, 6)
            'action': action,          # tensor (max_vehicles,)
            'reward': reward,          # float
            'next_obs': next_obs,      # tensor (max_vehicles, 6)
            'done': done               # bool
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        obs_batch = torch.stack([b['obs'] for b in batch])               # (B, V, 6)
        action_batch = torch.stack([b['action'] for b in batch])         # (B, V)
        reward_batch = torch.stack([b['reward'] for b in batch])  # shape (B,) → 不要再 .squeeze()
        next_obs_batch = torch.stack([b['next_obs'] for b in batch])     # (B, V, 6)
        done_batch = torch.stack([b['done'] for b in batch]).squeeze()   # (B,)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch


    def __len__(self):
        return len(self.buffer)
