import torch
import torch.nn as nn

def train_low_level_dqn(policy_net, target_net, optimizer, batch, gamma, tau):
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

    device = next(policy_net.parameters()).device
    obs_batch = obs_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    next_obs_batch = next_obs_batch.to(device)
    done_batch = done_batch.to(device)

    existence_mask = obs_batch[:, :, 5]  # existence feature

    current_logits = policy_net(obs_batch)
    current_prob = torch.sigmoid(current_logits)

    with torch.no_grad():
        next_logits = target_net(next_obs_batch)
        next_prob = torch.sigmoid(next_logits)
        max_next_prob, _ = next_prob.max(dim=1)
        target_value = reward_batch + gamma * max_next_prob * (1 - done_batch)
        target_value = target_value.unsqueeze(1).repeat(1, current_prob.size(1))

    bce_loss_fn = nn.BCELoss(reduction='none')
    loss = bce_loss_fn(current_prob, action_batch.float())
    loss = loss * existence_mask
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    return loss.item()
