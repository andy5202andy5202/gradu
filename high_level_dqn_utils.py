import random
import torch
import torch.nn.functional as F

def select_action(model, state, epsilon, action_dim):
    """ 
    state: torch tensor, shape=(batch, 6) or (6,)
    epsilon: float, exploration rate
    action_dim: int, 最大可選的 slot 數（12）
    """
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
            return q_values.argmax(dim=1).item()

def train_dqn(model, target_model, optimizer, batch, gamma=0.99):
    """
    batch: (obs_batch, action_batch, reward_batch, next_obs_batch)
    obs_batch: (batch_size, 6)
    action_batch: (batch_size,)
    reward_batch: (batch_size,)
    next_obs_batch: (batch_size, 6)
    """

    obs, action, reward, next_obs, done = batch
    q_values = model(obs)  # shape=(batch_size, action_dim)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  # shape=(batch_size,)

    with torch.no_grad():
        next_q_values = target_model(next_obs)
        max_next_q_values, _ = next_q_values.max(dim=1)
        target = reward + gamma * max_next_q_values * (1 - done)

    loss = F.mse_loss(q_value, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()