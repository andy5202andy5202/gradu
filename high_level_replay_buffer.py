import random
import numpy as np
import torch

class HighLevelReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, transition):
        """ transition = {
            'obs': obs,             # dict of agent_id: {"global": np.array([6,])}
            'action': action,       # dict of agent_id: {"num_slots": int, ...}
            'reward': reward,       # dict of agent_id: float
            'next_obs': next_obs,    # same as obs
            'done': bool
        } """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []

        for transition in batch:
            obs_batch.append(transition['obs'])          # 已是 tensor
            action_batch.append(transition['action'])    # int
            reward_batch.append(transition['reward'])    # float
            next_obs_batch.append(transition['next_obs']) # 已是 tensor
            done_batch.append(transition.get('done', False))
        
        for i, obs in enumerate(obs_batch):
            if obs.shape != (6,):
                print(f"第 {i} 筆 obs shape 有問題：{obs.shape}")

        obs_batch = torch.stack(obs_batch)
        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_obs_batch = torch.stack(next_obs_batch)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch


    def __len__(self):
        return len(self.buffer)
