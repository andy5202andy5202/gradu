# import torch
# import torch.nn.functional as F
# import os, time
# os.makedirs("logs", exist_ok=True)
# _LL_LOG_PATH = os.path.join("logs", "low_level_training_log.txt")
# _LL_STEP = 0

# def train_low_level_dqn(policy_net, target_net, optimizer, batch, gamma, tau, log_every=50):
#     # batch: (obs_batch[B,V,6], action_batch[B,V]{0/1}, reward_batch[B], next_obs_batch[B,V,6], done_batch[B])
#     obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

#     device = next(policy_net.parameters()).device
#     obs_batch       = obs_batch.to(device)            # (B, V, 6)
#     action_batch    = action_batch.to(device).long()  # (B, V)
#     reward_batch    = reward_batch.to(device).float() # (B,)
#     next_obs_batch  = next_obs_batch.to(device)       # (B, V, 6)
#     done_batch      = done_batch.to(device).float()   # (B,)

#     existence_mask = (obs_batch[:, :, 5] > 0).float() # (B, V)

#     # Q(s,a) for executed actions
#     q = policy_net(obs_batch)                         # (B, V, 2)
#     q_selected = q.gather(2, action_batch.unsqueeze(-1)).squeeze(-1)  # (B, V)

#     # Double DQN target
#     with torch.no_grad():
#         q_next_online = policy_net(next_obs_batch)    # (B, V, 2)
#         next_act = q_next_online.argmax(dim=2, keepdim=True)          # (B, V, 1)

#         q_next_target = target_net(next_obs_batch)    # (B, V, 2)
#         q_next_max = q_next_target.gather(2, next_act).squeeze(-1)    # (B, V)

#         reward_exp = reward_batch.view(-1, 1).expand_as(q_next_max)   # (B, V)
#         not_done   = (1.0 - done_batch.view(-1, 1)).expand_as(q_next_max)

#         target = reward_exp + gamma * q_next_max * not_done           # (B, V)

#     masked_pred   = q_selected * existence_mask
#     masked_target = target     * existence_mask

#     # Huber + 存在遮罩加權平均
#     loss = F.smooth_l1_loss(q_selected * existence_mask, target * existence_mask, reduction='sum')
#     denom = existence_mask.sum().clamp_min(1.0)
#     loss = loss / denom

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     global _LL_STEP
#     _LL_STEP += 1
    
#     with torch.no_grad():
#         td_abs = (masked_target - masked_pred).abs()
#         avg_td_error = (td_abs.sum() / denom).item()
#         if _LL_STEP % log_every == 0:
#             with open(_LL_LOG_PATH, "a", encoding="utf-8") as f:
#                 # step, loss, avg_td_error, ts
#                 f.write(f"{_LL_STEP},{loss.item():.6f},{avg_td_error:.6f},{time.time():.0f}\n")

#     # Soft update
#     with torch.no_grad():
#         for t, s in zip(target_net.parameters(), policy_net.parameters()):
#             t.data.mul_(1.0 - tau).add_(tau * s.data)

#     return loss.item()

import torch
import torch.nn.functional as F
import os, time
os.makedirs("logs", exist_ok=True)
_LL_LOG_PATH = os.path.join("logs", "low_level_training_log.txt")
_LL_STEP = 0

def train_low_level_dqn(policy_net, target_net, optimizer, batch, gamma, tau, log_every=50):
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

    device = next(policy_net.parameters()).device
    obs_batch       = obs_batch.to(device)
    action_batch    = action_batch.to(device).long()
    reward_batch    = reward_batch.to(device).float()
    next_obs_batch  = next_obs_batch.to(device)
    done_batch      = done_batch.to(device).float()

    existence_mask = (obs_batch[:, :, 5] > 0).float()  # (B, V)

    # Q(s,a)
    q = policy_net(obs_batch)                           # (B, V, 2)
    q_selected = q.gather(2, action_batch.unsqueeze(-1)).squeeze(-1)  # (B, V)

    # target
    with torch.no_grad():
        next_online = policy_net(next_obs_batch)       # (B, V, 2)
        next_act = next_online.argmax(dim=2, keepdim=True)             # (B, V, 1)
        next_target = target_net(next_obs_batch)       # (B, V, 2)
        q_next = next_target.gather(2, next_act).squeeze(-1)           # (B, V)

        reward_exp = reward_batch.view(-1,1).expand_as(q_next)
        not_done   = (1.0 - done_batch.view(-1,1)).expand_as(q_next)
        target = reward_exp + gamma * q_next * not_done                # (B, V)

    masked_pred   = q_selected * existence_mask
    masked_target = target     * existence_mask

    # Huber + 遮罩加權
    denom = existence_mask.sum().clamp_min(1.0)
    loss = F.smooth_l1_loss(masked_pred, masked_target, reduction='sum') / denom

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 週期性記錄 avg TD error
    global _LL_STEP
    _LL_STEP += 1
    with torch.no_grad():
        td_abs = (masked_target - masked_pred).abs()
        avg_td_error = (td_abs.sum() / denom).item()
        if _LL_STEP % log_every == 0:
            with open(_LL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{_LL_STEP},{loss.item():.6f},{avg_td_error:.6f},{int(time.time())}\n")

        for t, s in zip(target_net.parameters(), policy_net.parameters()):
            t.data.mul_(1.0 - tau).add_(tau * s.data)

    return loss.item()
