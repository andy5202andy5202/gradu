import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vehicle_manager_rl import init_environment
import threading
import traci
import time
import psutil



class FederatedGymEnv(gym.Env):
    def __init__(self, create_servers_fn, max_slots=10, max_vehicles=10, max_rounds=15, loss_threshold=0.02, num_agents=4):
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
        self.low_level_agent = None
        self.low_level_replay_buffer = None
        self.epsilon = None
        self.device = None


        single_obs_space = spaces.Dict({
            "global": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        })

        single_action_space = spaces.Dict({
            "num_slots": spaces.Discrete(self.max_slots),
            "slot_actions": spaces.MultiBinary((self.max_slots, self.max_vehicles))
        })

        self.observation_space = spaces.Dict({i: single_obs_space for i in range(self.num_agents)})
        self.action_space = spaces.Dict({i: single_action_space for i in range(self.num_agents)})
        self.prev_num_slots = {i: 1 for i in range(self.num_agents)}  # 初始預設都是 1 slot


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.round = 0
        self.prev_loss = [1.0] * self.num_agents
        self.curr_loss = [1.0] * self.num_agents
        self.prev_num_slots = {i: 1 for i in range(self.num_agents)}

        if self.vehicle_thread and self.vehicle_thread.is_alive():
            self.sim_thread.stop()  
            self.sim_thread.join(timeout=2)
            self.vehicle_thread.join(timeout=2)


        # 停掉上一輪 GlobalServer 與 SUMO
        if self.global_server and self.global_server.is_alive():
            self.global_server.stop()
            self.global_server.join(timeout=2)

        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.stop()
            self.sim_thread.join(timeout=2)

        try:
            traci.close()
        except:
            pass
        
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] in ('sumo', 'sumo-gui'):
                try:
                    print(f"[reset] Killing residual SUMO process PID {proc.info['pid']}")
                    proc.kill()
                except Exception as e:
                    print(f"[reset] Failed to kill SUMO PID {proc.info['pid']}: {e}")

        time.sleep(1)  # 等待 process 完全釋放
        
        # 重新初始化環境
        try:
            print("[reset] 正在執行 create_servers_fn()...")
            self.env_components = self.create_servers_fn()
            print("[reset] create_servers_fn() 成功")
        except Exception as e:
            print(f"[reset] create_servers_fn() 發生錯誤：{e}")
            import traceback
            traceback.print_exc()
            return None, {}

        # self.env_components = self.create_servers_fn()
        if self.env_components is None:
            raise RuntimeError("[reset] create_servers_fn() 回傳 None")
        # self.global_server = self.env_components["global_server"]
        # self.edge_servers = self.env_components["edge_servers"]
        # self.global_clock = self.env_components["global_clock"]
        # self.sim_thread = self.env_components["sim_thread"]
        # self.vehicle_thread = self.env_components["vehicle_thread"]
        # self.global_server = self.env_components.get("global_server", None)
        # self.edge_servers = self.env_components.get("edge_servers", None)
        # self.global_clock = self.env_components.get("global_clock", None)
        # self.sim_thread = self.env_components.get("sim_thread", None)
        # self.vehicle_thread = self.env_components.get("vehicle_thread", None)
        # if self.global_server is None or self.edge_servers is None:
        #     print("[reset] ERROR: create_servers_fn() 回傳不完整")
        #     return None, {}
        try:
            self.global_server = self.env_components.get("global_server", None)
            self.edge_servers = self.env_components.get("edge_servers", None)
            self.global_clock = self.env_components.get("global_clock", None)
            self.sim_thread = self.env_components.get("sim_thread", None)
            self.vehicle_thread = self.env_components.get("vehicle_thread", None)

            if self.global_server is None or self.edge_servers is None:
                print("[reset] ERROR: create_servers_fn() 回傳不完整")
                return None, {}
            print(f"[reset] 新 sim_thread ID: {id(self.sim_thread)}")

        except Exception as e:
            print(f"[reset] 錯誤：env_components 欄位缺失或格式錯誤：{e}")
            return None, {}

        self.global_server.max_rounds = self.max_rounds
        for edge_server in self.edge_servers:
            edge_server.attach_low_level_agent(
                low_level_dqn=self.low_level_agent,
                replay_buffer=self.low_level_replay_buffer,
                epsilon=self.epsilon,
                device=self.device
            )
        

        # 啟動新的 global server
        self.global_server.start()

        # 等待 GlobalClock 確實重啟完
        time.sleep(0.5)
        return self._get_observation(), {}


    def step(self, action):
        self.round += 1
        self.global_server.global_round = self.round
        for i in range(self.num_agents):
            self.prev_loss[i] = self.curr_loss[i]

        self.latest_num_slots = {}
        threads = []

        for i, edge in enumerate(self.edge_servers):
            agent_action = action[i]
            num_slots = agent_action["num_slots"] + 1
            slot_actions = agent_action["slot_actions"][:num_slots]
            self.latest_num_slots[i] = num_slots

            def run_edge_slots(edge=edge, num_slots=num_slots, slot_actions=slot_actions):
                edge.run_slots(
                    num_slots=num_slots,
                    slot_actions=slot_actions,
                    max_slots=self.max_slots,
                    max_vehicles=self.max_vehicles,
                    max_rounds=self.max_rounds
                )

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

        # 在取得 obs 前才更新 prev_num_slots
        self.prev_num_slots = self.latest_num_slots
        obs = self._get_observation(self.prev_num_slots)

        info = {
            "round": self.round,
            "loss": self.curr_loss,
            "global_loss": self.global_server.current_loss,
            "global_accuracy": self.global_server.current_accuracy
        }

        return obs, reward, done, False, info



    def _get_observation(self, latest_num_slots=None):
        if self.edge_servers is None:
            print("[_get_observation] self.edge_servers 是 None")
            return None
        if latest_num_slots is None:
            latest_num_slots = {i: 1 for i in range(self.num_agents)} 
        obs = {}
        for i in range(self.num_agents):
            avg_speed, avg_compute, avg_remain = self.edge_servers[i].get_avg_info()
            vehicle_state = self.edge_servers[i].get_vehicle_state(self.max_vehicles)

            vehicle_ratio = len(vehicle_state) / self.max_vehicles
            prev_slots = self.prev_num_slots.get(i, 1) / self.max_slots
            normalized_num_slots = latest_num_slots[i] / self.max_slots
            normalized_round = self.round / self.max_rounds
            
            state1 = np.array([
                avg_speed,
                avg_compute,
                avg_remain,
                vehicle_ratio,
                prev_slots,           # ← 用 self.prev_num_slots[i]
                normalized_round
            ], dtype=np.float32)
            obs[i] = {"global": state1}
        return obs
