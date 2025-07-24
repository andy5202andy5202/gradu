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
from low_level_dqn import LowLevelDQN
from low_level_replay_buffer import LowLevelReplayBuffer

from low_level_dqn_utils import train_low_level_dqn


def save_checkpoint(
    episode, 
    dqn, target_dqn, optimizer, buffer, epsilon, train_rewards_log, train_loss_log,
    low_level_dqn, low_level_target_dqn, low_level_optimizer, low_level_buffer,low_level_epsilon
):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'high_level_dqn': dqn.state_dict(),
        'high_level_target_dqn': target_dqn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'low_level_dqn': low_level_dqn.state_dict(),
        'low_level_target_dqn': low_level_target_dqn.state_dict(),
        'low_level_optimizer_state_dict': low_level_optimizer.state_dict()
    }, 'checkpoints/dqn_latest.pth')
    
    with open('checkpoints/replay_buffer.pkl', 'wb') as f:
        pickle.dump(buffer, f)
    with open('checkpoints/low_level_replay_buffer.pkl', 'wb') as f:
        pickle.dump(low_level_buffer, f)
    
    state = {
        'episode': episode,
        'epsilon': epsilon,
        'low_level_epsilon': low_level_epsilon,
        'train_rewards_log': train_rewards_log,
        'train_loss_log': train_loss_log
    }
    with open('checkpoints/training_state.pkl', 'wb') as f:
        pickle.dump(state, f)
    
    log_message = f"Checkpoint saved at Episode {episode+1}\n"
    print(log_message)
    with open('logs/high_level_training_log.txt', 'a') as f:
        f.write(log_message)




def load_checkpoint(dqn, target_dqn, optimizer,
                    low_level_dqn, low_level_target_dqn, low_level_optimizer,
                    buffer, low_level_buffer):
    if not os.path.exists('checkpoints/training_state.pkl'):
        return None

    checkpoint = torch.load('checkpoints/dqn_latest.pth')
    dqn.load_state_dict(checkpoint['high_level_dqn'])
    target_dqn.load_state_dict(checkpoint['high_level_target_dqn'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    low_level_dqn.load_state_dict(checkpoint['low_level_dqn'])
    low_level_target_dqn.load_state_dict(checkpoint['low_level_target_dqn'])
    low_level_optimizer.load_state_dict(checkpoint['low_level_optimizer_state_dict'])

    with open('checkpoints/replay_buffer.pkl', 'rb') as f:
        buffer_data = pickle.load(f)
        buffer.buffer = buffer_data.buffer
        buffer.position = buffer_data.position

    with open('checkpoints/low_level_replay_buffer.pkl', 'rb') as f:
        low_buffer_data = pickle.load(f)
        low_level_buffer.buffer = low_buffer_data.buffer
        low_level_buffer.position = low_buffer_data.position

    with open('checkpoints/training_state.pkl', 'rb') as f:
        state = pickle.load(f)
    
    state['low_level_epsilon'] = state.get('low_level_epsilon', 1.0)

    return buffer, low_level_buffer, state



def evaluate_agent(dqn, env, episode):
    try:
        
        original_epsilons = [edge.low_level_epsilon for edge in env.edge_servers]
        for edge in env.edge_servers:
            edge.low_level_epsilon = 0.0
            
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

    finally:
        for edge, original_epsilon in zip(env.edge_servers, original_epsilons):
            edge.low_level_epsilon = original_epsilon
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
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 1e-3
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    EVAL_INTERVAL = 10
    NUM_EPISODES = 500
    TRAIN_REPEAT_PER_STEP = 5
    MAX_EPISODES_BEFORE_RESTART = 4
    TAU = 0.02
    MAX_VEHICLES = 10
    
    LOW_LEVEL_BATCH_SIZE = 16
    LOW_LEVEL_LR = 1e-4
    LOW_LEVEL_GAMMA = 0.99
    LOW_LEVEL_TAU = 0.02
    LOW_LEVEL_TRAIN_REPEAT = 25
    LOW_LEVEL_BUFFER_CAPACITY = 10000
    
    LOW_LEVEL_EPSILON_START = 1.0
    LOW_LEVEL_EPSILON_END = 0.05
    LOW_LEVEL_EPSILON_DECAY = 0.995
    low_level_epsilon = LOW_LEVEL_EPSILON_START


    
    train_loss_records = []   # 每筆是 dict: {'episode': X, 'step': Y, 'loss': Z}


    os.makedirs("logs", exist_ok=True)

    env = FederatedGymEnv(create_servers_fn, max_slots=MAX_SLOTS, num_agents=NUM_AGENTS)

    LOW_LEVEL_STATE_DIM = 6  # 看你的設計
    LOW_LEVEL_ACTION_DIM = MAX_VEHICLES
    LOW_LEVEL_BUFFER_CAPACITY = 10000
    low_level_dqn = LowLevelDQN(input_dim=LOW_LEVEL_STATE_DIM, hidden_dim=64, max_vehicles=MAX_VEHICLES).cuda()
    low_level_target_dqn = LowLevelDQN(input_dim=LOW_LEVEL_STATE_DIM, hidden_dim=64, max_vehicles=MAX_VEHICLES).cuda()

    low_level_target_dqn.load_state_dict(low_level_dqn.state_dict())

    low_level_optimizer = optim.Adam(low_level_dqn.parameters(), lr=LOW_LEVEL_LR)
    low_level_replay_buffer = LowLevelReplayBuffer(LOW_LEVEL_BUFFER_CAPACITY)

    low_level_epsilon = 0.1


    dqn = HighLevelDQN(STATE_DIM, ACTION_DIM)
    target_dqn = HighLevelDQN(STATE_DIM, ACTION_DIM)
    optimizer = optim.Adam(dqn.parameters(), lr=LR)

    buffer = HighLevelReplayBuffer(BUFFER_CAPACITY)
    train_rewards_log = []
    train_loss_log = []
    epsilon = EPSILON_START
    start_episode = 0
    
    env.low_level_agent = low_level_dqn
    env.low_level_replay_buffer = low_level_replay_buffer
    env.epsilon = low_level_epsilon
    env.device = 'cuda'

    checkpoint = load_checkpoint(
        dqn, target_dqn, optimizer,
        low_level_dqn, low_level_target_dqn, low_level_optimizer,
        buffer, low_level_replay_buffer
    )

    
    episode_metrics_path = 'logs/episode_metrics.csv'

    if checkpoint:
        buffer, low_level_replay_buffer, state = checkpoint
        start_episode = state['episode'] + 1
        epsilon = state['epsilon']
        low_level_epsilon = state['low_level_epsilon']
        train_rewards_log = state['train_rewards_log']
        train_loss_log = state['train_loss_log']
        
        log_msg = f"Resumed from episode {start_episode}\n"
        print(log_msg)
        with open('logs/high_level_training_log.txt', 'a') as f:
            f.write(log_msg)
    else:
        target_dqn.load_state_dict(dqn.state_dict())
        low_level_target_dqn.load_state_dict(low_level_dqn.state_dict())
        start_episode = 0

        
    if start_episode == 0 or not os.path.exists(episode_metrics_path):
        with open(episode_metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_reward', 'avg_loss', 'epsilon'])

    for episode in range(start_episode, NUM_EPISODES):
        gc.collect()
        torch.cuda.empty_cache()

        is_high_level_episode = (episode % 2 == 1)
        print(f"[{'HighLevel' if is_high_level_episode else 'LowLevel'}] Episode {episode+1}")
        
        try:
            obs, _ = env.reset()
        except Exception as e:
            log_msg = f"Crash during env.reset() at episode {episode+1}: {e}\n"
            print(log_msg)
            with open('logs/high_level_training_log.txt', 'a') as f:
                f.write(log_msg)
            continue
        
        for edge in env.edge_servers:
            edge.low_level_epsilon = low_level_epsilon
        
        done = False
        episode_reward = {i: 0.0 for i in range(NUM_AGENTS)}
        low_level_losses = []
        low_level_rewards = []

        while not done:
            action = {}

            for agent_id in range(NUM_AGENTS):
                state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                num_slots = select_action(dqn, state1, epsilon if is_high_level_episode else 0, ACTION_DIM)

                slot_actions = np.zeros((MAX_SLOTS, env.max_vehicles), dtype=np.int8)
                action[agent_id] = {"num_slots": num_slots, "slot_actions": slot_actions}

            next_obs, reward, done, _, _ = env.step(action)

            # for agent_id in range(NUM_AGENTS):
            #     # High-level buffer
            #     high_transition = {
            #         "obs": torch.tensor(obs[agent_id]["global"], dtype=torch.float32),
            #         "action": action[agent_id]["num_slots"],
            #         "reward": reward[agent_id],
            #         "next_obs": torch.tensor(next_obs[agent_id]["global"], dtype=torch.float32),
            #         "done": done
            #     }
            #     buffer.add(high_transition)
            #     episode_reward[agent_id] += reward[agent_id]
            
            for agent_id in range(NUM_AGENTS):
                obs_tensor = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                next_obs_tensor = torch.tensor(next_obs[agent_id]["global"], dtype=torch.float32)
                act = action[agent_id]["num_slots"]
                rew = reward[agent_id]
                done_flag = done

                # 加入 log（含中文說明欄位順序）
                normalized_time = env.round / env.max_rounds  # normalized ∈ [0,1]
                alpha = 2  # 時間加權指數
                weight = (1 + normalized_time) ** alpha
                scaled_reward = rew * weight
                
                with open(f'logs/high_level_rl_agent{agent_id}.log', 'a', encoding='utf-8') as logf:
                    logf.write(f"[Round {env.round}] 高層轉移紀錄\n")
                    logf.write(f"  obs（平均速度, 平均運算力, 平均剩餘距離, 車輛比例, 上一輪 slot 數比例, 正規化 round）= {obs_tensor.tolist()}\n")
                    logf.write(f"  action（slot 數）= {act}\n")
                    logf.write(f"  reward = {rew:.4f}, scaled = {scaled_reward:.4f}\n")
                    logf.write(f"  next_obs = {next_obs_tensor.tolist()}\n")
                    logf.write(f"  done = {done_flag}\n\n")
            
                high_transition = {
                    "obs": obs_tensor,
                    "action": act,
                    "reward": scaled_reward,
                    "next_obs": next_obs_tensor,
                    "done": done_flag
                }
                buffer.add(high_transition)
                episode_reward[agent_id] += rew


            # 訓練 high-level 或 low-level
            if is_high_level_episode:
                for step in range(TRAIN_REPEAT_PER_STEP):
                    if len(buffer) >= BATCH_SIZE:
                        try:
                            batch = buffer.sample(BATCH_SIZE)
                            loss = train_dqn(dqn, target_dqn, optimizer, batch, gamma=GAMMA)
                            train_loss_log.append(loss)

                            log_msg = f"[訓練] Episode {episode+1} Global Round {env.round} Step {step+1}: Loss={loss:.4f}\n"
                            print(log_msg, end='')
                            with open('logs/high_level_training_log.txt', 'a') as f:
                                f.write(log_msg)
                        except Exception as e:
                            log_msg = f"Training error at episode {episode+1}: {e}\n"
                            print(log_msg)
                            with open('logs/high_level_training_log.txt', 'a') as f:
                                f.write(log_msg)
            else:
                if len(low_level_replay_buffer) >= LOW_LEVEL_BATCH_SIZE:
                    for _ in range(LOW_LEVEL_TRAIN_REPEAT):
                        batch = low_level_replay_buffer.sample(LOW_LEVEL_BATCH_SIZE)
                        low_loss = train_low_level_dqn(
                            low_level_dqn, low_level_target_dqn,
                            low_level_optimizer, batch,
                            gamma=LOW_LEVEL_GAMMA, tau=LOW_LEVEL_TAU
                        )
                        low_level_losses.append(low_loss)
                        low_level_rewards.extend(batch[2].cpu().numpy())

                        log_msg = f"[LowLevel] Episode {episode+1}: Loss: {low_loss:.4f}\n"
                        print(log_msg, end='')
                        with open('logs/low_level_training_log.txt', 'a') as f:
                            f.write(log_msg)
                else:
                    log_msg = f"[LowLevel] Episode {episode+1}: Buffer 不足，跳過訓練\n"
                    print(log_msg, end='')
                    with open('logs/low_level_training_log.txt', 'a') as f:
                        f.write(log_msg)

            for t_p, p in zip(target_dqn.parameters(), dqn.parameters()):
                t_p.data.copy_(TAU * p.data + (1 - TAU) * t_p.data)

            for t_p, p in zip(low_level_target_dqn.parameters(), low_level_dqn.parameters()):
                t_p.data.copy_(LOW_LEVEL_TAU * p.data + (1 - LOW_LEVEL_TAU) * t_p.data)

            obs = next_obs

        # Episode 結束統計與儲存
        if is_high_level_episode:
            avg_reward = np.mean(list(episode_reward.values()))
            train_rewards_log.append(avg_reward)
            num_steps_in_episode = env.round * TRAIN_REPEAT_PER_STEP
            recent_losses = train_loss_log[-num_steps_in_episode:]
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            log_msg = f"Episode {episode+1}: Reward {episode_reward}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon {epsilon:.3f}\n"
            print(log_msg, end='')
            with open('logs/high_level_training_log.txt', 'a') as f:
                f.write(log_msg)
            with open('logs/episode_metrics.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode+1, avg_reward, avg_loss, epsilon])
        else:
            low_level_epsilon = max(LOW_LEVEL_EPSILON_END, low_level_epsilon * LOW_LEVEL_EPSILON_DECAY)
            for edge in env.edge_servers:
                edge.low_level_epsilon = low_level_epsilon

            avg_low_loss = np.mean(low_level_losses) if low_level_losses else None
            avg_low_reward = np.mean(low_level_rewards) if low_level_rewards else None

            # 第一次寫檔時補上 header
            low_level_metrics_path = 'logs/low_level_episode_metrics.csv'
            if episode == 0 or not os.path.exists(low_level_metrics_path):
                with open(low_level_metrics_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['episode', 'avg_low_loss', 'avg_low_reward', 'epsilon'])

            with open(low_level_metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode+1, avg_low_loss, avg_low_reward, low_level_epsilon])


        save_checkpoint(
            episode, dqn, target_dqn, optimizer, buffer, epsilon, train_rewards_log, train_loss_log,
            low_level_dqn, low_level_target_dqn, low_level_optimizer, low_level_replay_buffer,low_level_epsilon
        )

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
