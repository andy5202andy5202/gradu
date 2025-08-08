import torch
import torch.nn.functional as F

def train_low_level_dqn(policy_net, target_net, optimizer, batch, gamma, tau):
    # batch: (obs_batch[B,V,6], action_batch[B,V]{0/1}, reward_batch[B], next_obs_batch[B,V,6], done_batch[B])
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

    device = next(policy_net.parameters()).device
    obs_batch       = obs_batch.to(device)            # (B, V, 6)
    action_batch    = action_batch.to(device).long()  # (B, V)
    reward_batch    = reward_batch.to(device).float() # (B,)
    next_obs_batch  = next_obs_batch.to(device)       # (B, V, 6)
    done_batch      = done_batch.to(device).float()   # (B,)

    existence_mask = (obs_batch[:, :, 5] > 0).float() # (B, V)

    # Q(s,a) for executed actions
    q = policy_net(obs_batch)                         # (B, V, 2)
    q_selected = q.gather(2, action_batch.unsqueeze(-1)).squeeze(-1)  # (B, V)

    # Double DQN target
    with torch.no_grad():
        q_next_online = policy_net(next_obs_batch)    # (B, V, 2)
        next_act = q_next_online.argmax(dim=2, keepdim=True)          # (B, V, 1)

        q_next_target = target_net(next_obs_batch)    # (B, V, 2)
        q_next_max = q_next_target.gather(2, next_act).squeeze(-1)    # (B, V)

        reward_exp = reward_batch.view(-1, 1).expand_as(q_next_max)   # (B, V)
        not_done   = (1.0 - done_batch.view(-1, 1)).expand_as(q_next_max)

        target = reward_exp + gamma * q_next_max * not_done           # (B, V)

    # Huber + 存在遮罩加權平均
    loss = F.smooth_l1_loss(q_selected * existence_mask, target * existence_mask, reduction='sum')
    denom = existence_mask.sum().clamp_min(1.0)
    loss = loss / denom

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update
    with torch.no_grad():
        for t, s in zip(target_net.parameters(), policy_net.parameters()):
            t.data.mul_(1.0 - tau).add_(tau * s.data)

    return loss.item()
