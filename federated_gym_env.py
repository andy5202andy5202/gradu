import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vehicle_manager_rl import init_environment
import threading
import traci
import time


class FederatedGymEnv(gym.Env):
    def __init__(self, create_servers_fn, max_slots=12, max_vehicles=8, max_rounds=100, loss_threshold=0.2, num_agents=4):
        super(FederatedGymEnv, self).__init__()
        self.create_servers_fn = create_servers_fn
        self.max_slots = max_slots
        self.max_vehicles = max_vehicles
        self.max_rounds = max_rounds
        self.loss_threshold = loss_threshold
        self.num_agents = num_agents

        self.round = 0
        self.prev_loss = [1.0] * self.num_agents
        self.curr_loss = [1.0] * self.num_agents

        self.global_server = None
        self.edge_servers = None
        self.global_clock = None
        self.sim_thread = None
        self.vehicle_thread = None

        single_obs_space = spaces.Dict({
            "global": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),
            "vehicles": spaces.Box(low=0.0, high=1.0, shape=(self.max_vehicles, 4), dtype=np.float32)
        })
        single_action_space = spaces.Dict({
            "num_slots": spaces.Discrete(self.max_slots),
            "slot_actions": spaces.MultiBinary((self.max_slots, self.max_vehicles))
        })

        self.observation_space = spaces.Dict({i: single_obs_space for i in range(self.num_agents)})
        self.action_space = spaces.Dict({i: single_action_space for i in range(self.num_agents)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.round = 0
        self.prev_loss = [1.0] * self.num_agents
        self.curr_loss = [1.0] * self.num_agents
        
        if self.vehicle_thread and self.vehicle_thread.is_alive():
            self.sim_thread.stop()  
            self.sim_thread.join(timeout=2)
            self.vehicle_thread.join(timeout=2)


        # 停掉上一輪 GlobalServer 與 SUMO
        if self.global_server and self.global_server.is_alive():
            self.global_server.stop()
            self.global_server.join()

        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.stop()
            self.sim_thread.join(timeout=2)

        try:
            traci.close()
        except:
            pass

        # 重新初始化環境
        self.env_components = self.create_servers_fn()
        self.global_server = self.env_components["global_server"]
        self.edge_servers = self.env_components["edge_servers"]
        self.global_clock = self.env_components["global_clock"]
        self.sim_thread = self.env_components["sim_thread"]
        self.vehicle_thread = self.env_components["vehicle_thread"]


        # 啟動新的 global server
        self.global_server.start()

        # 等待 GlobalClock 確實重啟完成
        time.sleep(0.5)

        return self._get_observation(), {}


    def step(self, action):
        self.round += 1
        for i in range(self.num_agents):
            self.prev_loss[i] = self.curr_loss[i]

        threads = []

        for i, edge in enumerate(self.edge_servers):
            agent_action = action[i]
            num_slots = agent_action["num_slots"] + 1
            slot_actions = agent_action["slot_actions"][:num_slots]

            def run_edge_slots(edge=edge, num_slots=num_slots, slot_actions=slot_actions):
                edge.run_slots(num_slots, slot_actions)

            t = threading.Thread(target=run_edge_slots)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        self.curr_loss = self.global_server.get_per_edge_loss()
        reward = {
            i: self.prev_loss[i] - self.curr_loss[i]
            for i in range(self.num_agents)
        }
        done = any(
            self.curr_loss[i] <= self.loss_threshold or self.round >= self.max_rounds
            for i in range(self.num_agents)
        )
        obs = self._get_observation()
        info = {"round": self.round, "loss": self.curr_loss}
        return obs, reward, done, False, info

    def _get_observation(self):
        obs = {}
        for i in range(self.num_agents):
            avg_speed, avg_compute, avg_remain = self.edge_servers[i].get_avg_info()
            vehicle_state = self.edge_servers[i].get_vehicle_state(self.max_vehicles)

            vehicle_ratio = len(vehicle_state) / self.max_vehicles
            prev_slots = 0
            normalized_round = self.round / self.max_rounds

            state1 = np.array([avg_speed, avg_compute, avg_remain, vehicle_ratio, prev_slots, normalized_round], dtype=np.float32)
            state2 = np.zeros((self.max_vehicles, 4), dtype=np.float32)
            for j, v in enumerate(vehicle_state):
                state2[j] = np.array(v, dtype=np.float32)

            obs[i] = {"global": state1, "vehicles": state2}
        return obs
