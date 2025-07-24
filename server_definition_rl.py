import os
import threading
import torch
from train_utils import aggregate_models, calculate_loss_and_accuracy, create_dataloader
# from models.resnet import SmallResNet
import time
from trainer import VehicleTrainer
import copy
from global_server import GlobalServer
import logging
import multiprocessing
from models.resnet import CIFAR_CNN
from low_level_replay_buffer import LowLevelReplayBuffer as ReplayBuffer
import numpy as np





class EdgeServer(threading.Thread):
    def __init__(self, server_id, covered_edges,cached_node_data, 
                global_data_path, active_training_threads, 
                global_server,upload_due_to_position,
                upload_due_to_early_stop,
                upload_due_to_global_timeout_counter, 
                global_clock=None, global_time=120, waiting_time=30, device='cuda',
                position_status_dict=None,
                vehicle_current_edge=None,
                vehicle_exit_edge=None):
        super().__init__(daemon=True)
        self.server_id = server_id
        self.covered_edges = covered_edges
        self.cached_node_data = cached_node_data
        self.global_data_path = global_data_path
        self.active_training_threads = active_training_threads
        self.global_server = global_server
        self.global_clock = global_clock
        self.logger = self.setup_logger()
        self.global_time = global_time
        self.waiting_time = waiting_time
        self.device = device
        self.upload_due_to_position = upload_due_to_position
        self.upload_due_to_early_stop = upload_due_to_early_stop
        self.upload_due_to_global_timeout_counter = upload_due_to_global_timeout_counter
        # self.model = SmallResNet(num_classes=10).to('cpu')
        self.model = CIFAR_CNN(num_classes=10).to('cpu')
        self.low_level_dqn = None
        self.low_level_epsilon = 1  # 可調整，探索率
        self.low_level_device = 'cuda'  # 或依 device 設定
        self.low_level_replay_buffer = None



        self.received_models = []
        self.last_selection_time = time.time()
        self.model_version = 1
        self.training_semaphore = multiprocessing.Semaphore(25)
        self.received_models_lock = threading.Lock()
        self.update_model(self.global_server.model.state_dict(), self.global_server.model_version)
        self.position_status_dict = position_status_dict
        self.vehicle_current_edge = vehicle_current_edge
        self.vehicle_exit_edge = vehicle_exit_edge
        self.enable_auto_run = False
        # self.replay_buffer = ReplayBuffer()
        self.rl_logger = logging.getLogger(self.server_id + "_rl")
        self.rl_logger.setLevel(logging.INFO)
        
        if self.rl_logger.hasHandlers():
            self.rl_logger.handlers.clear()

        rl_handler = logging.FileHandler(f"{self.server_id}_rl.log", mode='w', encoding='utf-8')
        rl_handler.setFormatter(logging.Formatter('%(message)s'))
        self.rl_logger.addHandler(rl_handler)

    def attach_low_level_agent(self, low_level_dqn, replay_buffer, epsilon=0.1, device='cuda'):
        self.low_level_dqn = low_level_dqn.to(device)
        self.low_level_replay_buffer = replay_buffer
        self.low_level_epsilon = epsilon
        self.low_level_device = device

    
    def collect_uploaded_models(self):
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            return

        for fname in os.listdir(upload_dir):
            if not fname.endswith(".pkl"):
                continue
            if not fname.startswith(self.server_id + "_"):
                continue

            path = os.path.join(upload_dir, fname)
            try:
                model_info = torch.load(path)
                model_state = model_info["model_state_dict"]
                version = model_info["model_version"]
                global_version = model_info["global_version"]
                current_global = self.global_server.model_version

                if global_version == current_global:
                    self.logger.info(f"[接受] 載入來自 {fname} 的模型（v={version}）")
                    self.received_models.append((model_state, version))
                else:
                    self.logger.info(f"[丟棄] {fname} → global version 不符（{global_version} ≠ {current_global}）")
                    self.global_server.discarded_model_upload_total += 1
                    self.global_server.discarded_model_uploads_this_round += 1

            except Exception as e:
                self.logger.warning(f"讀取 {fname} 發生錯誤：{e}")

            finally:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    def get_avg_info(self):
        speeds, computes, remains = [], [], []
        snapshot = list(self.active_training_threads.items())
        for vid, info in snapshot:
            if info.get("trainer") is None:
                edge_id = self.vehicle_current_edge.get(vid, None)
                if edge_id is not None and self.is_in_range(edge_id):
                    speed = info.get("max_speed", 0)
                    compute = info.get("compute_power", 0)
                    remain = info.get("remaining_steps", 0)
                    speeds.append(speed)
                    computes.append(compute)
                    remains.append(remain)

        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        avg_compute = sum(computes) / len(computes) if computes else 0.0
        avg_remain = sum(remains) / len(remains) if remains else 0.0

        return avg_speed / 20, avg_compute / 10, avg_remain / 100
    
    def get_vehicle_state(self, max_k=10, slot_ratio=1.0, normalized_round=0.0):
        vehicles = []
        snapshot_keys = list(self.active_training_threads.keys())
        for vid in snapshot_keys:
            info = self.active_training_threads.get(vid)
            if info is None:
                continue
            if info.get("trainer") is None:
                edge_id = self.vehicle_current_edge.get(vid, None)
                if edge_id is not None and self.is_in_range(edge_id):
                    speed = info.get("max_speed", 0) / 20.0
                    compute = info.get("compute_power", 0) / 4.0
                    remain = info.get("remaining_steps", 0) / 100.0
                    vehicles.append([speed, compute, remain, slot_ratio, normalized_round, 1.0])

        while len(vehicles) < max_k:
            vehicles.append([0.0, 0.0, 0.0,slot_ratio, normalized_round, 0.0])

        return vehicles[:max_k]
    
    def setup_logger(self):
        logger = logging.getLogger(self.server_id)
        logger.setLevel(logging.INFO)

        #這邊不需要檢查 handler，直接清除所有 handler（確保每次 clean）
        if logger.hasHandlers():
            logger.handlers.clear()

        class GlobalClockFormatter(logging.Formatter):
            def __init__(self, fmt=None, datefmt=None, global_clock=None):
                super().__init__(fmt, datefmt)
                self.global_clock = global_clock

            def format(self, record):
                try:
                    record.custom_time = f"[GlobalClock] {self.global_clock.get_time():.1f}s"
                except:
                    record.custom_time = "[GlobalClock] ??s"
                return super().format(record)

        formatter = GlobalClockFormatter('%(custom_time)s - %(message)s', global_clock=self.global_clock)

        file_handler = logging.FileHandler(f"{self.server_id}.log", mode='w')  # overwrite mode
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    
    def get_model(self):
        """提供最新的模型與版本號"""
        return self.model.state_dict(), self.model_version
    
    
    def update_model(self, new_state_dict, new_version, alpha=0.7):
        """
        Edge Server 對 Global Server 傳下來的模型進行 momentum 融合更新。
        θ_edge ← (1 - α) * θ_edge + α * θ_global
        """
        old_state_dict = self.model.state_dict()
        new_state_dict_smooth = {}

        for key in old_state_dict:
            if isinstance(old_state_dict[key], torch.Tensor) and torch.is_floating_point(old_state_dict[key]):
                new_param = (1 - alpha) * old_state_dict[key] + alpha * new_state_dict[key].to(old_state_dict[key].device)
                new_state_dict_smooth[key] = new_param
            else:
                new_state_dict_smooth[key] = new_state_dict[key]

        self.model.load_state_dict(new_state_dict_smooth)
        self.model_version = new_version
        self.logger.info(f"{self.server_id} 使用 α={alpha} momentum 更新模型 → 版本 {new_version}")


    def is_in_range(self, edge_id):
        return edge_id in self.covered_edges
    
    def get_data_for_vehicle(self, vehicle_id):
        try:
            info = self.active_training_threads[vehicle_id]
            group = info.get('data_group')
            max_speed = info.get('max_speed', 10.0)
            if group is None:
                self.logger.warning(f"[{vehicle_id}] 缺少 data_group")
                return None
            speed_int = int(round(max(1, min(20, max_speed))))
            return self.cached_node_data[group][speed_int]
        except Exception as e:
            self.logger.error(f"[{vehicle_id}] 取得資料錯誤：{e}")
            return None
    
    def run_single_slot(self, slot_obs, expected_slot_start, slot_time, global_deadline, slot_ratio, normalized_round):

        expected_slot_end = expected_slot_start + slot_time
        self.logger.info(f"{self.server_id} slot 預計執行 {slot_time:.1f}s，等待至 GlobalClock={expected_slot_end:.1f}s")
        
        # slot_obs_flat = np.array(slot_obs, dtype=np.float32).flatten()
        # slot_obs_tensor = torch.tensor(slot_obs_flat, dtype=torch.float32).unsqueeze(0).to(self.low_level_device)  # shape (1, 60)
        slot_obs_tensor = torch.tensor(slot_obs, dtype=torch.float32).unsqueeze(0).to(self.low_level_device)  # shape (1, V, F)

        existence_mask = torch.tensor([v[5] for v in slot_obs], dtype=torch.float32).to(self.low_level_device)  # shape (max_vehicles,)
        with torch.no_grad():
            logits = self.low_level_dqn(slot_obs_tensor)
            probs = torch.sigmoid(logits).squeeze(0)  # shape: (max_vehicles,)

        # Epsilon-greedy
        
        random_mask = torch.randint(0, 2, probs.shape, device=self.low_level_device)
        action_mask = torch.where(torch.rand_like(probs) < self.low_level_epsilon, random_mask, (probs >= 0.5).int())

        # existence mask
        
        action_mask = action_mask * existence_mask.int()

        action_mask = action_mask.cpu().numpy().tolist()

        selected_vehicles = []
        snapshot_keys = list(self.active_training_threads.keys())
        for idx, bit in enumerate(action_mask):
            if bit == 1 and idx < len(snapshot_keys):
                vid = snapshot_keys[idx]
                edge_id = self.vehicle_current_edge.get(vid, None)
                if edge_id is not None and self.is_in_range(edge_id):
                    selected_vehicles.append(vid)

        self.logger.info(f"[{self.server_id}] Slot action mask: {action_mask}")
        if selected_vehicles:
            self.logger.info(f"[{self.server_id}] Slot 選中的車輛:")
            for vid in selected_vehicles:
                info = self.active_training_threads.get(vid, {})
                speed = info.get('max_speed', 0)
                compute = info.get('compute_power', 0)
                remain = info.get('remaining_steps', 0)
                self.logger.info(f"  車輛 {vid} → 速度={speed}, 運算力={compute}, 剩餘距離={remain}")
        else:
            self.logger.info(f"[{self.server_id}] Slot 無車輛被選中")

        
        for vid in selected_vehicles:
            if vid not in self.active_training_threads:
                self.logger.warning(f"{self.server_id} 車輛 {vid} 已離開系統，跳過。")
                continue

            vehicle_info = self.active_training_threads[vid]
            if vehicle_info.get("trainer") is not None and vehicle_info["trainer"].is_alive():
                self.logger.warning(f"{self.server_id} 車輛 {vid} 的 trainer 還在跑，跳過。")
                continue

            if 'data' not in vehicle_info:
                data = self.get_data_for_vehicle(vid)
                if data is None:
                    self.logger.warning(f"{self.server_id} 車輛 {vid} 找不到對應資料，跳過。")
                    continue
                vehicle_info['data'] = data

            if not self.training_semaphore.acquire(timeout=1.0):
                self.logger.info(f"{self.server_id} 車輛 {vid} 無法取得 GPU slot，略過此次訓練。")
                continue

            try:
                # pos_dict_copy = dict(self.position_status_dict)
                trainer = VehicleTrainer(
                    vehicle_id=vid,
                    data_for_vehicle=vehicle_info['data'],
                    edge_server_id=self.server_id,
                    model_state_dict=copy.deepcopy(self.model.state_dict()),
                    model_version=self.model_version,
                    global_version=self.global_server.model_version,
                    upload_due_to_position_counter=self.upload_due_to_position,
                    upload_due_to_early_stop=self.upload_due_to_early_stop,
                    upload_due_to_global_timeout_counter=self.upload_due_to_global_timeout_counter,
                    global_deadline=global_deadline,
                    global_clock=self.global_clock.time_value,
                    compute_power=vehicle_info['compute_power'],
                    max_speed=vehicle_info['max_speed'],
                    remaining_steps=vehicle_info['remaining_steps'],
                    gpu_fraction=1.0 / 100,
                    position_status_dict=self.position_status_dict,
                    vehicle_current_edge=self.vehicle_current_edge,
                    vehicle_exit_edge=self.vehicle_exit_edge,
                    device=self.device
                )
                trainer.start()
                self.active_training_threads[vid]['trainer'] = trainer
                self.logger.info(f"{self.server_id} 啟動 {vid} 訓練")

                def release_after_done():
                    try:
                        trainer.join(timeout=20)  # 最多等5分鐘
                        self.logger.info(f"{self.server_id} {vid} 訓練完成，釋放 GPU slot")
                    except Exception as e:
                        self.logger.error(f"{self.server_id} {vid} join 發生錯誤：{e}")
                    finally:
                        self.training_semaphore.release()
                        if vid in self.active_training_threads:
                            self.active_training_threads[vid]['trainer'] = None
                            
                threading.Thread(target=release_after_done, daemon=True).start()
            except Exception as e:
                self.logger.error(f"{self.server_id} 車輛 {vid} trainer 啟動失敗：{e}")
                self.training_semaphore.release()

        while self.global_clock.get_time() < expected_slot_end:
            time.sleep(0.1)

        self.collect_uploaded_models()
        with self.received_models_lock:
            if self.received_models:
                aggregated = aggregate_models(self.received_models, self)
                self.model.load_state_dict({
                    k: v for k, v in aggregated.items() if k in self.model.state_dict()
                })
                self.received_models.clear()
                self.model_version += 1
                self.logger.info(f"{self.server_id} 聚合完成，模型版本 {self.model_version}")
            else:
                self.logger.info(f"{self.server_id} 此 slot 無收到模型，版本不變")
           
        next_slot_obs = self.get_vehicle_state(slot_ratio=slot_ratio, normalized_round=normalized_round)
        return action_mask, next_slot_obs
         
    def run_slots(self, num_slots, slot_actions, max_slots=10, max_vehicles=10, max_rounds=30):
        self.logger.info(f"{self.server_id} 開始執行 run_slots()，slots = {num_slots}")
        normalized_num_slots = num_slots / max_slots
        self.slot_rewards = []
        try:
            self.prev_slot_loss = self.global_server.get_loss_for_edge(self.server_id)
            self.logger.info(f"{self.server_id} 初始 loss（slot 0 前）= {self.prev_slot_loss:.4f}")
        except Exception as e:
            self.logger.warning(f"{self.server_id} 初始 loss 記錄失敗: {e}")
            self.prev_slot_loss = 1.0  # fallback

        if len(slot_actions) < num_slots:
            self.logger.error(f"{self.server_id} slot_actions 長度不足（{len(slot_actions)} < {num_slots}），結束此輪")
            return
        
        round_start = self.global_clock.get_time()
        slot_time = self.global_time / num_slots
        global_deadline = round_start + self.global_time

        for i in range(num_slots):
            
            expected_slot_start = round_start + i * slot_time
            
            while self.global_clock.get_time() < expected_slot_start:
                time.sleep(0.05)

            normalized_round = self.global_server.global_round / max_rounds
            slot_obs = self.get_vehicle_state(max_vehicles, normalized_num_slots, normalized_round)
            
            try:
                action_mask, next_slot_obs = self.run_single_slot(
                    slot_obs, expected_slot_start, slot_time, global_deadline, normalized_num_slots, normalized_round
                )
            except Exception as e:
                self.logger.error(f"{self.server_id} slot {i} 執行失敗：{e}")
                action_mask = [0] * max_vehicles
                # next_slot_obs = self.get_vehicle_state(max_vehicles, normalized_num_slots, normalized_round)

            try:
                after_loss = self.global_server.get_loss_for_edge(self.server_id)
                raw_reward = self.prev_slot_loss - after_loss

                alpha = 2  # 權重
                time_ratio = (self.global_server.global_round + 1) / max_rounds  # e.g., 1~20 normalized
                weight = (1 + time_ratio) ** alpha
                slot_reward = raw_reward * weight

                self.slot_rewards.append(slot_reward)
                self.logger.info(
                    f"{self.server_id} Slot {i+1} reward = {slot_reward:.4f}（raw={raw_reward:.4f}, weight={weight:.2f}, loss {self.prev_slot_loss:.4f} → {after_loss:.4f}）"
                )
                self.prev_slot_loss = after_loss
            except Exception as e:
                self.logger.warning(f"{self.server_id} 無法計算 Slot {i+1} reward: {e}")
                slot_reward = 0.0
                self.slot_rewards.append(0.0)

            #Log slot transition 詳細資訊
            self.rl_logger.info(f"\n[{self.server_id}] Slot {i+1} transition:")
            self.rl_logger.info(f"觀察 obs（每列為車輛）: [速度, 運算力, 剩餘距離, slot比例, 輪數比例, 是否存在]")
            for idx, v in enumerate(slot_obs):
                self.rl_logger.info(f"  車輛 {idx}: {v}")
            self.rl_logger.info(f"動作 action mask（0:不選，1:選）: {action_mask}")
            self.rl_logger.info(f"回饋 reward: {slot_reward}")
            self.rl_logger.info(f"下一狀態 next_obs:")
            for idx, v in enumerate(next_slot_obs):
                self.rl_logger.info(f"  車輛 {idx}: {v}")


            self.low_level_replay_buffer.add(
                torch.tensor(slot_obs, dtype=torch.float32).unsqueeze(0),          
                torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0),      
                torch.tensor([slot_reward], dtype=torch.float32),                 
                torch.tensor(next_slot_obs, dtype=torch.float32).unsqueeze(0),     
                torch.tensor([False])                                          
            )





        self.global_server.received_models.append((self.model.state_dict(), self.model_version))
        self.logger.info(f"{self.server_id} 已上傳模型 v{self.model_version} 給 Global Server")

        current_version = self.global_server.model_version
        while self.global_server.model_version <= current_version:
            time.sleep(0.1)

        self.update_model(self.global_server.model.state_dict(), self.global_server.model_version)

        self.model_version = 1





