# high_level_dqn_utils.py
import random
import torch
import torch.nn.functional as F
import os, time, torch
import torch.nn.functional as F
os.makedirs("logs", exist_ok=True)
_HL_LOG_PATH = os.path.join("logs", "high_level_training_log.txt")
_HL_STEP = 0

# def select_action(model, state, epsilon, action_dim):
#     """ 
#     state: torch tensor, shape=(batch, 6) or (6,)
#     epsilon: float, exploration rate
#     action_dim: int, 最大可選的 slot 數（12）
#     """
#     if random.random() < epsilon:
#         return random.randint(0, action_dim - 1)
#     else:
#         if len(state.shape) == 1:
#             state = state.unsqueeze(0)
#         with torch.no_grad():
#             q_values = model(state)
#             return q_values.argmax(dim=1).item()

def select_action(model, state, epsilon, action_dim, agent_id=None):
    """ 
    state: torch tensor, shape=(batch, 6) or (6,)
    epsilon: float, exploration rate
    action_dim: int, 最大可選的 slot 數（12）
    agent_id: int or None, 用來決定 log 檔案輸出路徑
    """
    if len(state.shape) == 1:
        state = state.unsqueeze(0)

    if random.random() < epsilon:
        action = random.randint(0, action_dim - 1)
        log_path = f"logs/high_level_rl_agent{agent_id}.log" if agent_id is not None else "logs/select_action.log"
        with open(log_path, "a") as f:
            f.write(f"[Eval select_action] ε={epsilon:.4f}, action={action} (Random)\n")
        return action
    else:
        with torch.no_grad():
            q_values = model(state)
            action = q_values.argmax(dim=1).item()
            log_path = f"logs/high_level_rl_agent{agent_id}.log" if agent_id is not None else "logs/select_action.log"
            with open(log_path, "a") as f:
                f.write(f"[Eval select_action] ε={epsilon:.4f}, action={action}, Q={q_values.cpu().numpy().tolist()}\n")
            return action


# def train_dqn(model, target_model, optimizer, batch, gamma=0.99):
#     """
#     batch: (obs_batch, action_batch, reward_batch, next_obs_batch)
#     obs_batch: (batch_size, 6)
#     action_batch: (batch_size,)
#     reward_batch: (batch_size,)
#     next_obs_batch: (batch_size, 6)
#     """

#     obs, action, reward, next_obs, done = batch
#     q_values = model(obs)  # shape=(batch_size, action_dim)
#     q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  # shape=(batch_size,)

#     with torch.no_grad():
#         next_q_values = target_model(next_obs)
#         max_next_q_values, _ = next_q_values.max(dim=1)
#         target = reward + gamma * max_next_q_values * (1 - done)

#     loss = F.mse_loss(q_value, target)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     return loss.item()


def train_dqn(model, target_model, optimizer, batch, gamma, tau=0.02, log_every=50):
    """
    返回 (loss, grad_norm)
    batch: (obs[B,feat], action[B], reward[B], next_obs[B,feat], done[B])
    """
    obs, action, reward, next_obs, done = batch
    device = next(model.parameters()).device
    obs      = obs.to(device).float()
    action   = action.to(device).long().view(-1, 1)
    reward   = reward.to(device).float().view(-1)
    next_obs = next_obs.to(device).float()
    done     = done.to(device).float().view(-1)

    # Q(s,a)
    q = model(obs)                           # (B, A)
    q_selected = q.gather(1, action).squeeze(1)  # (B,)

    # Double DQN target
    with torch.no_grad():
        next_act = model(next_obs).argmax(dim=1, keepdim=True)         # (B,1)
        q_next   = target_model(next_obs).gather(1, next_act).squeeze(1)  # (B,)
        target   = reward + gamma * q_next * (1.0 - done)

    loss = F.smooth_l1_loss(q_selected, target)

    optimizer.zero_grad()
    loss.backward()

    # 量化梯度是否真的在流動
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data
            grad_sq += float(g.norm(2).item() ** 2)
    grad_norm = grad_sq ** 0.5

    optimizer.step()

    # soft update
    with torch.no_grad():
        for tp, p in zip(target_model.parameters(), model.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

        global _HL_STEP
        _HL_STEP += 1
        if _HL_STEP % log_every == 0:
            q_mean = float(q_selected.mean().item())
            td_mean = float((target - q_selected).abs().mean().item())
            with open(_HL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[HL] step={_HL_STEP}, loss={loss.item():.6f}, grad={grad_norm:.6f}, q_mean={q_mean:.6f}, td={td_mean:.6f}, ts={int(time.time())}\n")

    return loss.item(), grad_norm
