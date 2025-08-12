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
import time

from low_level_dqn_utils import train_low_level_dqn
# >>> BEGIN PATCH (helpers in train_high_level.py)
import csv
from pathlib import Path

def append_eval_summary(eval_csv_path, summary_path="evaluation_logs/summary.csv",
                        tag="current", high_level_loss=None, low_level_loss=None):
    """
    從剛剛輸出的單次評估 CSV（包含 global_loss 欄位）彙整平均數，追加到 summary.csv。
    - eval_csv_path: 剛存好的評估檔路徑（單回或單次 run 的 df）
    - tag: 這次評估的名稱（如 'ep42'、'checkpoint_100k'）
    - high_level_loss/low_level_loss: 可選，若你有在評估階段計算到
    """
    eval_csv_path = Path(eval_csv_path)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # 讀入剛剛輸出的評估 CSV
    import pandas as pd
    df = pd.read_csv(eval_csv_path)

    # 盡量穩健的彙整方式：有 global_loss 就用尾端 5 步平均；否則用整體平均
    if "global_loss" in df.columns and len(df["global_loss"]) > 0:
        tail_n = min(5, len(df))
        avg_global_loss = float(df["global_loss"].tail(tail_n).mean())
    else:
        avg_global_loss = float(df.mean(numeric_only=True).mean())

    # 若有 slot_reward 欄位，就加個平均 reward 做參考
    avg_slot_reward = None
    for col in ["slot_reward", "reward", "slot_rewards"]:
        if col in df.columns and len(df[col]) > 0:
            avg_slot_reward = float(df[col].mean())
            break

    row = {
        "tag": str(tag),
        "avg_global_loss": avg_global_loss,
        "avg_slot_reward": avg_slot_reward,
        "high_level_loss": high_level_loss,
        "low_level_loss": low_level_loss,
        "timestamp": int(time.time()),
        "csv": str(eval_csv_path)
    }

    # 追加到 summary.csv（若不存在就寫 header）
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
# >>> END PATCH



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
        
    with open('logs/low_level_training_log.txt', 'a') as f:
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



# def evaluate_agent(dqn, env, episode):
#     try:
        
#         original_epsilons = [edge.low_level_epsilon for edge in env.edge_servers]
        
            
#         obs, _ = env.reset()
#         for edge in env.edge_servers:
#             edge.low_level_epsilon = 0.0
#         done = False
#         round_metrics = []

#         while not done:
#             action = {}
#             for agent_id in range(env.num_agents):
#                 state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
#                 num_slots = select_action(dqn, state1, epsilon=0, action_dim=env.max_slots, agent_id=agent_id)
#                 dummy_slot_actions = np.random.randint(0, 2, (env.max_slots, env.max_vehicles)).astype(np.int8)
#                 action[agent_id] = {"num_slots": num_slots, "slot_actions": dummy_slot_actions}

#             obs, _, done, _, info = env.step(action)

#             global_loss = info.get('global_loss', None)
#             global_accuracy = info.get('global_accuracy', None)
#             round_metrics.append({'round': len(round_metrics) + 1, 'global_loss': global_loss, 'global_accuracy': global_accuracy})

#         os.makedirs('evaluation_logs', exist_ok=True)
#         eval_filename = f'evaluation_logs/eval_episode_{episode+1}_full_rounds.csv'
#         with open(eval_filename, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=['round', 'global_loss', 'global_accuracy'])
#             writer.writeheader()
#             writer.writerows(round_metrics)

#         log_message = f"[Evaluate] Episode {episode+1}: Full evaluation results saved to {eval_filename}\n"
        
#         append_eval_summary(
#             eval_csv_path=eval_csv_path,
#             summary_path="evaluation_logs/summary.csv",
#             tag=f"ep{episode_idx}_greedy",          # 依你的情境改：當前 episode、或 checkpoint 名稱
#             high_level_loss=hloss if 'hloss' in locals() else None,
#             low_level_loss=lloss if 'lloss' in locals() else None
#         )
#     except Exception as e:
#         error_detail = traceback.format_exc()
#         log_message = f"[Evaluate] Episode {episode+1} failed: {e}\n{error_detail}\n"

#     finally:
#         for edge, original_epsilon in zip(env.edge_servers, original_epsilons):
#             edge.low_level_epsilon = original_epsilon
#         with open('logs/high_level_training_log.txt', 'a') as log_file:
#             log_file.write(log_message)

def evaluate_agent(dqn, env, episode):
    try:
        # 備份並將 low-level epsilon 與 env.epsilon 全部設為 0（確保 reset 時也為 0）
        original_env_eps = getattr(env, "epsilon", None)

        env.epsilon = 0.0
        obs, _ = env.reset()
        for edge in env.edge_servers:
            edge.low_level_epsilon = 0.0

        done = False
        round_metrics = []

        while not done:
            action = {}
            for agent_id in range(env.num_agents):
                state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                num_slots = select_action(dqn, state1, epsilon=0, action_dim=env.max_slots, agent_id=agent_id)
                # 低層在 eval 用 greedy 由 Edge 決策，不需要在這裡指定 slot_actions
                action[agent_id] = {"num_slots": num_slots, "slot_actions": np.zeros((env.max_slots, env.max_vehicles), dtype=np.int8)}

            obs, _, done, _, info = env.step(action)

            global_loss = info.get('global_loss', None)
            global_accuracy = info.get('global_accuracy', None)
            round_metrics.append({
                'round': len(round_metrics) + 1,
                'global_loss': global_loss,
                'global_accuracy': global_accuracy
            })

        os.makedirs('evaluation_logs', exist_ok=True)
        eval_filename = f'evaluation_logs/eval_episode_{episode+1}_full_rounds.csv'
        with open(eval_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'global_loss', 'global_accuracy'])
            writer.writeheader()
            writer.writerows(round_metrics)

        log_message = f"[Evaluate] Episode {episode+1}: Full evaluation results saved to {eval_filename}\n"

        # 寫入 summary（**修正未定義變數**）
        append_eval_summary(
            eval_csv_path=eval_filename,
            summary_path="evaluation_logs/summary.csv",
            tag=f"ep{episode+1}_greedy",
            high_level_loss=None,   # 若你在 eval 同時計算 loss 可填入
            low_level_loss=None
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        log_message = f"[Evaluate] Episode {episode+1} failed: {e}\n{error_detail}\n"

    finally:
        if original_env_eps is not None:
            env.epsilon = original_env_eps
        for edge in env.edge_servers:
            edge.low_level_epsilon = env.epsilon  # 用 env 的值作為單一真相來源

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
    EPSILON_START = 0.7
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.99
    EVAL_INTERVAL = 10
    NUM_EPISODES = 500
    TRAIN_REPEAT_PER_STEP = 5
    MAX_EPISODES_BEFORE_RESTART = 4
    TAU = 0.01
    MAX_VEHICLES = 10
    
    LOW_LEVEL_BATCH_SIZE = 32
    LOW_LEVEL_LR = 1e-4
    LOW_LEVEL_GAMMA = 0.99
    LOW_LEVEL_TAU = 0.01
    LOW_LEVEL_TRAIN_REPEAT = 50
    LOW_LEVEL_BUFFER_CAPACITY = 10000
    
    LOW_LEVEL_EPSILON_START = 1.0
    LOW_LEVEL_EPSILON_END = 0.05
    LOW_LEVEL_EPSILON_DECAY = 0.99
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
    os.makedirs("logs", exist_ok=True)

    
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
            
    if start_episode == 1:
        print("[Baseline] : 執行一次隨機 baseline 評估...")

        # 建立環境
        env = FederatedGymEnv(create_servers_fn, max_slots=MAX_SLOTS, num_agents=NUM_AGENTS)
        env.low_level_agent = low_level_dqn
        env.low_level_replay_buffer = low_level_replay_buffer
        env.epsilon = 1.0  
        env.device = 'cuda'

        obs, _ = env.reset()

        # 設定 low-level epsilon
        for edge in env.edge_servers:
            edge.low_level_epsilon = 1.0

        done = False
        baseline_metrics = []

        while not done:
            action = {}
            for agent_id in range(NUM_AGENTS):
                num_slots = np.random.randint(1, MAX_SLOTS + 1)
                slot_actions = np.random.randint(0, 2, (num_slots + 1, env.max_vehicles)).astype(np.int8)
                action[agent_id] = {
                    "num_slots": num_slots,
                    "slot_actions": slot_actions
                }

            obs, _, done, _, info = env.step(action)

            loss = info.get('global_loss', None)
            acc = info.get('global_accuracy', None)
            baseline_metrics.append({'round': len(baseline_metrics)+1, 'global_loss': loss, 'global_accuracy': acc})

            print(f"[Baseline] Round {len(baseline_metrics)}, Loss: {loss:.4f}")
            with open('logs/high_level_training_log.txt', 'a') as f:
                f.write(f"[Baseline] 使用 high ε = {env.epsilon}, low ε = {[edge.low_level_epsilon for edge in env.edge_servers]}\n")

        os.makedirs('evaluation_logs', exist_ok=True)
        eval_filename = 'evaluation_logs/eval_episode_0_full_rounds.csv'
        with open(eval_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'global_loss', 'global_accuracy'])
            writer.writeheader()
            writer.writerows(baseline_metrics)

        print(f"[Baseline] 完成 baseline，結果寫入 {eval_filename}")

        start_episode = 1


    # 清空buffer
    # buffer.buffer = []
    # buffer.position = 0
    # if hasattr(low_level_replay_buffer, "buffer"):
    #     low_level_replay_buffer.buffer = []
    # if hasattr(low_level_replay_buffer, "position"):
    #     low_level_replay_buffer.position = 0
    # epsilon = EPSILON_START

    for episode in range(start_episode, NUM_EPISODES):
        gc.collect()
        torch.cuda.empty_cache()

        is_high_level_episode = (episode % 2 == 0)
        # ---- 高層訓練回合：低層改用 greedy，降低回饋噪聲 ----
        if is_high_level_episode:
            env.epsilon = 0.0
            eps_for_hl = epsilon
        else:
            env.epsilon = low_level_epsilon
            eps_for_hl = 0.0 
            
            
        print(f"[{'HighLevel' if is_high_level_episode else 'LowLevel'}] Episode {episode+1}")
        
        # for edge in env.edge_servers:
        #     edge.low_level_epsilon = low_level_epsilon

        
        try:
            obs, _ = env.reset()
        except Exception as e:
            log_msg = f"Crash during env.reset() at episode {episode+1}: {e}\n"
            print(log_msg)
            with open('logs/high_level_training_log.txt', 'a') as f:
                f.write(log_msg)
            continue
        
        for edge in env.edge_servers:
            edge.low_level_epsilon = env.epsilon
            
        done = False
        episode_reward = {i: 0.0 for i in range(NUM_AGENTS)}
        low_level_losses = []
        low_level_rewards = []
        with open('logs/high_level_training_log.txt', 'a') as f:
            f.write(f"[Episode {episode+1}] high ε(used) = {eps_for_hl:.4f}, low ε(env) = {env.epsilon:.4f}\n")

        # if episode == 112:
        #     buffer.buffer = [] 
        #     buffer.position = 0
        #     print(f"[Episode {episode}] 清空 high-level replay buffer")
        #     with open('logs/high_level_training_log.txt', 'a') as f:
        #         f.write(f"[Episode {episode}] 清空 high-level replay buffer\n")
 
        #     low_level_replay_buffer.buffer = []
        #     low_level_replay_buffer.position = 0
        #     print(f"[Episode {episode}] 清空 low-level replay buffer")
        #     with open('logs/low_level_training_log.txt', 'a') as f:
        #         f.write(f"[Episode {episode}] 清空 low-level replay buffer\n")
        updates_this_ep = 0
        while not done:
            action = {}

            for agent_id in range(NUM_AGENTS):
                state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                num_slots = select_action(dqn, state1, eps_for_hl, ACTION_DIM, agent_id=agent_id)

                slot_actions = np.zeros((MAX_SLOTS, env.max_vehicles), dtype=np.int8)
                action[agent_id] = {"num_slots": num_slots, "slot_actions": slot_actions}

            next_obs, reward, done, _, _ = env.step(action)
            
            for agent_id in range(NUM_AGENTS):
                obs_tensor = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                next_obs_tensor = torch.tensor(next_obs[agent_id]["global"], dtype=torch.float32)
                act = action[agent_id]["num_slots"]
                rew = reward[agent_id]
                done_flag = done

                # 加入 log（含中文說明欄位順序）
                normalized_time = env.round / env.max_rounds
                w_out = 0.5 + 0.5 * normalized_time      
                scaled_reward = float(np.tanh(rew / 3.0)) * w_out
                # 最後可保險再夾一下（理論上 w_out≤1 不會超過 ±1）
                scaled_reward = max(-1.0, min(1.0, scaled_reward))
                high_reward_log_path = f"logs/high_reward_{agent_id}.log"
                if not os.path.exists(high_reward_log_path):
                    with open(high_reward_log_path, 'w', encoding='utf-8') as f:
                        f.write("round,scaled_reward,raw_reward,normalized_round,action_slot\n")
                with open(high_reward_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{env.round},{scaled_reward:.6f},{rew:.6f},{normalized_time:.6f},{act}\n")
                
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
                HIGH_STEP_CSV = 'logs/high_level_step_rewards.csv'
                if not os.path.exists(HIGH_STEP_CSV):
                    with open(HIGH_STEP_CSV, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['episode', 'round', 'agent_id', 'scaled_reward', 'raw_reward', 'normalized_round', 'action_slot'])

                with open(HIGH_STEP_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode + 1,
                        env.round,
                        agent_id,
                        f"{scaled_reward:.6f}",
                        f"{rew:.6f}",
                        f"{normalized_time:.6f}",
                        act
                    ])
                buffer.add(high_transition)
                episode_reward[agent_id] += rew


            # 訓練 high-level 或 low-level
            if is_high_level_episode:
                # epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                if len(buffer) < 1000:
                    repeat = 3
                elif len(buffer) < 3000:
                    repeat = 5
                else:
                    repeat = 8
                for step in range(repeat):
                    if len(buffer) >= BATCH_SIZE:
                        try:
                            batch = buffer.sample(BATCH_SIZE)
                            ret = train_dqn(dqn, target_dqn, optimizer, batch, gamma=GAMMA, tau=TAU)
                            if isinstance(ret, (tuple, list)):
                                loss, grad_norm = ret[0], ret[1]
                            else:
                                loss, grad_norm = ret, None

                            train_loss_log.append(loss)
                            updates_this_ep += 1

                            log_msg = f"[訓練] Episode {episode+1} Global Round {env.round} Step {step+1}: Loss={loss:.4f}"
                            if grad_norm is not None:
                                log_msg += f", GradNorm={grad_norm:.4f}"
                            log_msg += "\n"
                            print(log_msg, end='')
                            with open('logs/high_level_training_log.txt', 'a') as f:
                                f.write(log_msg)

                        except Exception as e:
                            log_msg = f"Training error at episode {episode+1}: {e}\n"
                            print(log_msg)
                            with open('logs/high_level_training_log.txt', 'a') as f:
                                f.write(log_msg)
            else:
                # low_level_epsilon = max(LOW_LEVEL_EPSILON_END, low_level_epsilon * LOW_LEVEL_EPSILON_DECAY)
                if len(low_level_replay_buffer) >= LOW_LEVEL_BATCH_SIZE:
                    effective_repeat = min(20, len(low_level_replay_buffer) // LOW_LEVEL_BATCH_SIZE)
                    for _ in range(effective_repeat):
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

            

            obs = next_obs

        # Episode 結束統計與儲存
        if is_high_level_episode:
            avg_reward = np.mean(list(episode_reward.values()))
            train_rewards_log.append(avg_reward)
            
            recent_losses = train_loss_log[-updates_this_ep:] if updates_this_ep > 0 else []
            avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.
            
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
