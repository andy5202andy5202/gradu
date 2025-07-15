import torch
import numpy as np
from federated_gym_env import FederatedGymEnv
from server_factory import create_servers_fn
from high_level_dqn import HighLevelDQN
from high_level_dqn_utils import select_action
import os

# --- 設定 ---
CHECKPOINT_DIR = 'checkpoints'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'high_level_dqn_episode_500.pth')
MAX_SLOTS = 12
STATE_DIM = 6
ACTION_DIM = MAX_SLOTS
NUM_AGENTS = 4
EVAL_EPISODES = 10

# --- 載入模型 ---
dqn = HighLevelDQN(STATE_DIM, ACTION_DIM)
dqn.load_state_dict(torch.load(MODEL_PATH))
dqn.eval()

env = FederatedGymEnv(create_servers_fn, max_slots=MAX_SLOTS, num_agents=NUM_AGENTS)

for ep in range(EVAL_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = {i:0.0 for i in range(NUM_AGENTS)}

    while not done:
        action = {}
        for agent_id in range(NUM_AGENTS):
            state1 = torch.tensor(obs[agent_id]["global"], dtype=torch.float32)
            num_slots = select_action(dqn, state1, epsilon=0, action_dim=ACTION_DIM)
            dummy_slot_actions = np.random.randint(0, 2, (MAX_SLOTS, env.max_vehicles)).astype(np.int8)
            action[agent_id] = {"num_slots": num_slots, "slot_actions": dummy_slot_actions}

        obs, reward, done, _, info = env.step(action)

        for agent_id in range(NUM_AGENTS):
            total_reward[agent_id] += reward[agent_id]

    print(f"[Inference] Episode {ep+1} Total Reward: {total_reward}")
