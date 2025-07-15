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


def main():
    # --- Hyperparameters ---
    MAX_SLOTS = 12
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
    TARGET_UPDATE_FREQ = 10
    SAVE_INTERVAL = 20
    EVAL_INTERVAL = 50
    EVAL_EPISODES = 5
    NUM_EPISODES = 500
    TRAIN_ITERS_PER_EPISODE = 10

    env = FederatedGymEnv(create_servers_fn, max_slots=MAX_SLOTS, num_agents=NUM_AGENTS)
    buffer = HighLevelReplayBuffer(BUFFER_CAPACITY)
    train_rewards_log = []
    train_loss_log = []

    dqn = HighLevelDQN(STATE_DIM, ACTION_DIM)
    target_dqn = HighLevelDQN(STATE_DIM, ACTION_DIM)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=LR)

    epsilon = EPSILON_START

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/high_level_training_log.txt"
    with open(log_path, "w") as f:
        f.write("[High-Level DQN 訓練 Log]\n")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = {i:0.0 for i in range(NUM_AGENTS)}

        while not done:
            action = {}
            for agent_id in range(NUM_AGENTS):
                state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                num_slots = select_action(dqn, state1, epsilon, ACTION_DIM)
                dummy_slot_actions = np.random.randint(0, 2, (MAX_SLOTS, env.max_vehicles)).astype(np.int8)
                action[agent_id] = {"num_slots": num_slots, "slot_actions": dummy_slot_actions}

            next_obs, reward, done, _, info = env.step(action)

            for agent_id in range(NUM_AGENTS):
                transition = {
                    "obs": torch.tensor(obs[agent_id]["global"], dtype=torch.float32),
                    "action": action[agent_id]["num_slots"],
                    "reward": reward[agent_id],
                    "next_obs": torch.tensor(next_obs[agent_id]["global"], dtype=torch.float32)
                }
                buffer.add(transition)

                episode_reward[agent_id] += reward[agent_id]
            # with open(log_path, "a") as f:
            #     f.write(f"[Episode {episode+1} - Global Round {env.round}] Transition:\n")
            #     for agent_id in range(NUM_AGENTS):
            #         num_slots = action[agent_id]["num_slots"] + 1
            #         f.write(f"  Edge{agent_id}:\n")
            #         f.write(f"    action num_slots: {num_slots}\n")
            #         f.write(f"    reward: {reward[agent_id]:.4f}\n")

            #         obs_labels = ["平均速度", "平均運算能力", "平均剩餘距離比例", "有效車輛比例", "上一輪 slot 數比例", "global round 進度"]
            #         obs_values = obs[agent_id]['global'].tolist()
            #         f.write(f"    obs (global):\n")
            #         for label, val in zip(obs_labels, obs_values):
            #             f.write(f"      {label}: {val:.4f}\n")

            #         next_obs_values = next_obs[agent_id]['global'].tolist()
            #         f.write(f"    next_obs (global):\n")
            #         for label, val in zip(obs_labels, next_obs_values):
            #             f.write(f"      {label}: {val:.4f}\n")
            #     f.write("\n")

            obs = next_obs

        # 每集結束後訓練多次
        for train_iter in range(TRAIN_ITERS_PER_EPISODE * episode):
            if len(buffer) >= BATCH_SIZE:
                obs_batch, action_batch, reward_batch, next_obs_batch = buffer.sample(BATCH_SIZE)
                loss = train_dqn(dqn, target_dqn, optimizer,
                                (obs_batch, action_batch, reward_batch, next_obs_batch),
                                gamma=GAMMA)
                train_loss_log.append(loss)
                print(f"[訓練] Episode {episode+1} Iteration {train_iter+1}: Loss={loss:.4f}")
                with open(log_path, "a") as f:
                    f.write(f"[訓練] Episode {episode+1} Iteration {train_iter+1}: Loss={loss:.4f}\n")



        avg_reward = np.mean(list(episode_reward.values()))
        train_rewards_log.append(avg_reward)

        if 'loss' in locals():
            train_loss_log.append(loss)
        else:
            train_loss_log.append(0.0)

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        if (episode + 1) % SAVE_INTERVAL == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(dqn.state_dict(), f"checkpoints/high_level_dqn_episode_{episode+1}.pth")
            print(f"模型已儲存：checkpoints/high_level_dqn_episode_{episode+1}.pth")

        if (episode + 1) % EVAL_INTERVAL == 0:
            eval_rewards = {i:0.0 for i in range(NUM_AGENTS)}
            for _ in range(EVAL_EPISODES):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = {}
                    for agent_id in range(NUM_AGENTS):
                        state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
                        num_slots = select_action(dqn, state1, epsilon=0, action_dim=ACTION_DIM)
                        dummy_slot_actions = np.random.randint(0, 2, (MAX_SLOTS, env.max_vehicles)).astype(np.int8)
                        action[agent_id] = {"num_slots": num_slots, "slot_actions": dummy_slot_actions}

                    obs, reward, done, _, _ = env.step(action)
                    for agent_id in range(NUM_AGENTS):
                        eval_rewards[agent_id] += reward[agent_id]
            
            avg_eval_rewards = {i: eval_rewards[i] / EVAL_EPISODES for i in eval_rewards}
            print(f"[評估] Episode {episode+1}: Average Eval Reward {avg_eval_rewards}")

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        log_msg = f"Episode {episode+1}: Reward {episode_reward}, Avg Reward: {avg_reward:.2f}, Epsilon {epsilon:.3f}"
        print(log_msg)
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")

    print("訓練完成。")
    os.makedirs("checkpoints", exist_ok=True)
    with open('checkpoints/high_level_training_logs.pkl', 'wb') as f:
        pickle.dump({
            'rewards': train_rewards_log,
            'losses': train_loss_log
        }, f)
    print("訓練 log 已儲存：checkpoints/high_level_training_logs.pkl")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
