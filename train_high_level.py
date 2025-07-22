import torch
import torch.optim as optim
import numpy as np
from federated_gym_env import FederatedGymEnv
from server_factory import create_servers_fn
from high_level_dqn import HighLevelDQN
from high_level_replay_buffer import HighLevelReplayBuffer
from high_level_dqn_utils import select_action, train_dqn
import pickle
import os
import multiprocessing as mp
import gc
import csv
import pandas as pd
import sys
import traceback

def save_checkpoint(episode, dqn, target_dqn, optimizer, buffer, epsilon, train_rewards_log, train_loss_log):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': dqn.state_dict(),
        'target_model_state_dict': target_dqn.state_dict(),  # ✅ 加上 target_dqn
        'optimizer_state_dict': optimizer.state_dict()
    }, 'checkpoints/dqn_latest.pth')
    
    with open('checkpoints/replay_buffer.pkl', 'wb') as f:
        pickle.dump(buffer, f)
    
    state = {
        'episode': episode,
        'epsilon': epsilon,
        'train_rewards_log': train_rewards_log,
        'train_loss_log': train_loss_log
    }
    with open('checkpoints/training_state.pkl', 'wb') as f:
        pickle.dump(state, f)
    
    log_message = f"Checkpoint saved at Episode {episode+1}\n"
    print(log_message)
    with open('logs/high_level_training_log.txt', 'a') as f:
        f.write(log_message)


def load_checkpoint(dqn, target_dqn, optimizer):
    if not os.path.exists('checkpoints/training_state.pkl'):
        return None

    checkpoint = torch.load('checkpoints/dqn_latest.pth')
    dqn.load_state_dict(checkpoint['model_state_dict'])
    target_dqn.load_state_dict(checkpoint['target_model_state_dict'])  # ✅ 還原 target_dqn
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with open('checkpoints/replay_buffer.pkl', 'rb') as f:
        buffer = pickle.load(f)
    with open('checkpoints/training_state.pkl', 'rb') as f:
        state = pickle.load(f)

    return buffer, state


def evaluate_agent(dqn, env, episode):
    try:
        obs, _ = env.reset()
        done = False
        round_metrics = []

        while not done:
            action = {}
            for agent_id in range(env.num_agents):
                state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                num_slots = select_action(dqn, state1, epsilon=0, action_dim=env.max_slots)
                dummy_slot_actions = np.random.randint(0, 2, (env.max_slots, env.max_vehicles)).astype(np.int8)
                action[agent_id] = {"num_slots": num_slots, "slot_actions": dummy_slot_actions}

            obs, _, done, _, info = env.step(action)

            global_loss = info.get('global_loss', None)
            global_accuracy = info.get('global_accuracy', None)
            round_metrics.append({'round': len(round_metrics) + 1, 'global_loss': global_loss, 'global_accuracy': global_accuracy})

        os.makedirs('evaluation_logs', exist_ok=True)
        eval_filename = f'evaluation_logs/eval_episode_{episode+1}_full_rounds.csv'
        with open(eval_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'global_loss', 'global_accuracy'])
            writer.writeheader()
            writer.writerows(round_metrics)

        log_message = f"[Evaluate] Episode {episode+1}: Full evaluation results saved to {eval_filename}\n"
    except Exception as e:
        error_detail = traceback.format_exc()
        log_message = f"[Evaluate] Episode {episode+1} failed: {e}\n{error_detail}\n"

    with open('logs/high_level_training_log.txt', 'a') as log_file:
        log_file.write(log_message)

def safe_exit():
    # cleanup global_server, threads, GPU
    torch.cuda.empty_cache()
    gc.collect()
    print("[SAFE EXIT] Cleanup done. Exiting...")
    sys.exit(0)


def main():
    MAX_SLOTS = 10
    STATE_DIM = 6
    ACTION_DIM = MAX_SLOTS
    NUM_AGENTS = 4
    BUFFER_CAPACITY = 10000
    BATCH_SIZE = 16
    GAMMA = 0.99
    LR = 1e-3
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    EVAL_INTERVAL = 10
    NUM_EPISODES = 500
    TRAIN_REPEAT_PER_STEP = 5
    MAX_EPISODES_BEFORE_RESTART = 3
    TAU = 0.02
    MAX_VEHICLES = 10
    
    train_loss_records = []   # 每筆是 dict: {'episode': X, 'step': Y, 'loss': Z}


    os.makedirs("logs", exist_ok=True)

    env = FederatedGymEnv(create_servers_fn, max_slots=MAX_SLOTS, num_agents=NUM_AGENTS)

    # 加上 low-level agent 相關設定
    from low_level_dqn import LowLevelDQN
    from low_level_replay_buffer import ReplayBuffer as LowLevelReplayBuffer

    LOW_LEVEL_STATE_DIM = 6 * MAX_VEHICLES  # 看你的設計
    LOW_LEVEL_ACTION_DIM = MAX_VEHICLES
    LOW_LEVEL_BUFFER_CAPACITY = 10000
    low_level_dqn = LowLevelDQN(LOW_LEVEL_STATE_DIM, LOW_LEVEL_ACTION_DIM)
    low_level_replay_buffer = LowLevelReplayBuffer(LOW_LEVEL_BUFFER_CAPACITY)
    low_level_epsilon = 0.1
    low_level_device = 'cuda'

    env.low_level_agent = low_level_dqn
    env.low_level_replay_buffer = low_level_replay_buffer
    env.epsilon = low_level_epsilon
    env.device = low_level_device

    dqn = HighLevelDQN(STATE_DIM, ACTION_DIM)
    target_dqn = HighLevelDQN(STATE_DIM, ACTION_DIM)
    optimizer = optim.Adam(dqn.parameters(), lr=LR)

    buffer = HighLevelReplayBuffer(BUFFER_CAPACITY)
    train_rewards_log = []
    train_loss_log = []
    epsilon = EPSILON_START
    start_episode = 0

    checkpoint = load_checkpoint(dqn, target_dqn, optimizer)
    
    episode_metrics_path = 'logs/episode_metrics.csv'

    if checkpoint:
        buffer, state = checkpoint
        target_dqn.load_state_dict(dqn.state_dict())
        start_episode = state['episode'] + 1
        epsilon = state['epsilon']
        train_rewards_log = state['train_rewards_log']
        train_loss_log = state['train_loss_log']
        log_msg = f"Resumed from episode {start_episode}\n"
        print(log_msg)
        with open('logs/high_level_training_log.txt', 'a') as f:
            f.write(log_msg)
    else:
        target_dqn.load_state_dict(dqn.state_dict())
        start_episode = 0
        
    if start_episode == 0 or not os.path.exists(episode_metrics_path):
        with open(episode_metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_reward', 'avg_loss', 'epsilon'])

    for episode in range(start_episode, NUM_EPISODES):
        gc.collect()
        torch.cuda.empty_cache()

        try:
            obs, _ = env.reset()
        except Exception as e:
            log_msg = f"Crash during env.reset() at episode {episode+1}: {e}\n"
            print(log_msg)
            with open('logs/high_level_training_log.txt', 'a') as f:
                f.write(log_msg)
            continue

        done = False
        episode_reward = {i: 0.0 for i in range(NUM_AGENTS)}

        while not done:
            action = {}
            for agent_id in range(NUM_AGENTS):
                state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                num_slots = select_action(dqn, state1, epsilon, ACTION_DIM)
                dummy_slot_actions = np.random.randint(0, 2, (MAX_SLOTS, env.max_vehicles)).astype(np.int8)
                action[agent_id] = {"num_slots": num_slots, "slot_actions": dummy_slot_actions}

            next_obs, reward, done, _, _ = env.step(action)

            for agent_id in range(NUM_AGENTS):
                transition = {
                    "obs": torch.tensor(obs[agent_id]["global"], dtype=torch.float32),
                    "action": action[agent_id]["num_slots"],
                    "reward": reward[agent_id],
                    "next_obs": torch.tensor(next_obs[agent_id]["global"], dtype=torch.float32),
                    "done": done
                }
                buffer.add(transition)

                episode_reward[agent_id] += reward[agent_id]

            for step in range(TRAIN_REPEAT_PER_STEP):
                if len(buffer) >= BATCH_SIZE:
                    try:
                        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = buffer.sample(BATCH_SIZE)
                        loss = train_dqn(dqn, target_dqn, optimizer,
                                        (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch),
                                        gamma=GAMMA)
                        
                        train_loss_log.append(loss)
                        
                        log_msg = f"[訓練] Episode {episode+1} Global Round {env.round} Step {step+1}: Loss={loss:.4f}\n"

                        print(log_msg, end='')
                        with open('logs/high_level_training_log.txt', 'a') as f:
                            f.write(log_msg)
                    except Exception as e:
                        log_msg = f"Training error during step training at episode {episode+1}: {e}\n"
                        print(log_msg)
                        with open('logs/high_level_training_log.txt', 'a') as f:
                            f.write(log_msg)

            for target_param, param in zip(target_dqn.parameters(), dqn.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
            obs = next_obs

        avg_reward = np.mean(list(episode_reward.values()))
        train_rewards_log.append(avg_reward)
        num_steps_in_episode = env.round * TRAIN_REPEAT_PER_STEP
        recent_losses = train_loss_log[-num_steps_in_episode:]
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0


        # if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        #     target_dqn.load_state_dict(dqn.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        log_msg = f"Episode {episode+1}: Reward {episode_reward}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon {epsilon:.3f}\n"
        print(log_msg, end='')
        with open('logs/high_level_training_log.txt', 'a') as f:
            f.write(log_msg)
        
        with open('logs/episode_metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if episode == 0 and start_episode == 0:
                writer.writerow(['episode', 'avg_reward', 'avg_loss', 'epsilon'])
            writer.writerow([episode+1, avg_reward, avg_loss, epsilon])

        save_checkpoint(episode, dqn, target_dqn, optimizer, buffer, epsilon, train_rewards_log, train_loss_log)
        
        if (episode + 1) % EVAL_INTERVAL == 0:
            evaluate_agent(dqn, env, episode)

        
        if (episode + 1) % MAX_EPISODES_BEFORE_RESTART == 0:
            print(f"[INFO] Episode {episode+1} 達到上限，強制退出，請重新啟動程式")
            safe_exit()

    print("訓練完成。")
    with open('logs/high_level_training_log.txt', 'a') as f:
        f.write("訓練完成。\n")



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
