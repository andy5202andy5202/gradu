import os
import pickle
import threading
import torch
import random
import traci
from train_utils import aggregate_models, calculate_loss_and_accuracy, create_dataloader
# from models.resnet import SmallResNet
import time
from trainer import VehicleTrainer
import copy
from global_server import GlobalServer
import logging
import multiprocessing
from models.resnet import CIFAR_CNN


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


        self.received_models = []
        self.last_selection_time = time.time()
        self.model_version = 1
        self.training_semaphore = multiprocessing.Semaphore(25)
        self.received_models_lock = threading.Lock()
        self.update_model(self.global_server.model.state_dict(), self.global_server.model_version)
        self.position_status_dict = position_status_dict
        self.vehicle_current_edge = vehicle_current_edge
        self.vehicle_exit_edge = vehicle_exit_edge

    
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
        for vid, info in self.active_training_threads.items():
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
    
    def get_vehicle_state(self, max_k=8):
        vehicles = []
        for vid, info in self.active_training_threads.items():
            if info.get("trainer") is None:
                edge_id = self.vehicle_current_edge.get(vid, None)
                if edge_id is not None and self.is_in_range(edge_id):
                    speed = info.get("max_speed", 0) / 20.0
                    compute = info.get("compute_power", 0) / 10.0
                    remain = info.get("remaining_steps", 0) / 100.0
                    vehicles.append([speed, compute, remain, 1.0])

        while len(vehicles) < max_k:
            vehicles.append([0.0, 0.0, 0.0, 0.0])

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


        
    def run(self):
        try:
            while True:
                round_start = self.global_clock.get_time()  # 換用全局時間
                global_deadline = round_start + self.global_time
                total_slots = int(self.global_time / self.waiting_time)
                
                for current_slot in range(total_slots):  
                    expected_slot_start = round_start + current_slot * self.waiting_time
                    expected_slot_end = expected_slot_start + self.waiting_time

                    # 立即選車
                    # print(f"[GlobalClock] {self.global_clock.get_time()}s - {self.server_id} Slot {current_slot + 1} 選擇車輛...")
                    self.logger.info(f"{self.server_id} Slot {current_slot + 1} 選擇車輛...")

                    # 1. 車輛選擇
                    vehicles_in_area = []
                    vehicle_infos = []
                    
                    active_threads_copy = self.active_training_threads.copy()
                    for vid, vehicle_info in active_threads_copy.items():
                        if vehicle_info.get('trainer') is None:  # 還沒開始訓練的車輛
                            if vid not in traci.vehicle.getIDList():
                                continue  # 該車輛已離開模擬，不要呼叫 getRoadID
                            try:
                                position = traci.vehicle.getRoadID(vid)
                                if position and position.startswith("n_") and self.is_in_range(position):  # position 是 edge_id
                                    compute_power = vehicle_info.get('compute_power', -1)
                                    route_length = vehicle_info.get('route_length', -1)
                                    
                                    try:
                                        route_index = traci.vehicle.getRouteIndex(vid)
                                        remaining_steps = route_length - route_index
                                    except Exception as e:
                                        remaining_steps = -1
                                        self.logger.warning(f"{self.server_id} 無法取得 {vid} 的 route index：{e}")
                                    vehicle_info["remaining_steps"] = remaining_steps
                                    vehicles_in_area.append(vid)
                                    vehicle_infos.append((vid, compute_power, route_length, remaining_steps))
                            except Exception as e:
                                # print(f'取得車輛 {vid} 位置時發生錯誤：{e}')
                                self.logger.info(f'取得車輛 {vid} 位置時發生錯誤：{e}')

                    # print(f'{self.server_id} 範圍內的車輛: {vehicles_in_area}')
                    self.logger.info(f'{self.server_id} 範圍內的車輛: {vehicles_in_area}')
                    if vehicle_infos:
                        self.logger.info(f"{self.server_id} 可選車輛資訊如下：")
                        for vid, cp, rl, rs in vehicle_infos:
                            self.logger.info(f"{vid} | compute={cp} | route_len={rl} | remain={rs}")
                    else:
                        self.logger.info(f"{self.server_id} 此 slot 無可選車輛。")

                    # 隨機選擇最多三輛車來訓練
                    num_to_select = random.randint(0, len(vehicles_in_area))
                    # selected_vehicles = random.sample(vehicles_in_area,min(3,len(vehicles_in_area)))
                    selected_vehicles = random.sample(vehicles_in_area,num_to_select)
                    # print(f'{self.server_id} 選中的車輛: {selected_vehicles}')
                    # selected_vehicles = vehicles_in_area  # 全選
                    self.logger.info(f'{self.server_id} 選中的車輛: {selected_vehicles}')
                    self.logger.info(f"目前系統 thread 數量: {threading.active_count()}")
                    
                    # 2. 啟動選中的車輛進行訓練
                    
                    for vid in selected_vehicles:
                        
                        if vid not in self.active_training_threads:
                            self.logger.warning(f"{self.server_id} 車輛 {vid} 在選中後已離開系統，跳過。")
                            continue

                        vehicle_info = self.active_training_threads[vid]
                        trainer_obj = vehicle_info.get('trainer')

                        if trainer_obj is not None and trainer_obj.is_alive():
                            self.logger.warning(f"{self.server_id} 車輛 {vid} 的 trainer 還在跑，跳過這輛車。")
                            continue

                        if 'data' not in vehicle_info:
                            data_for_vehicle = self.get_data_for_vehicle(vid)
                            if data_for_vehicle is None:
                                self.logger.warning(f"{self.server_id} 車輛 {vid} 找不到對應資料，跳過。")
                                continue
                            vehicle_info['data'] = data_for_vehicle
                            
                        if not self.training_semaphore.acquire(timeout=1.0):
                            self.logger.info(f"{self.server_id} 車輛 {vid} 無法取得 GPU slot，略過此次訓練。")
                            continue

                        self.logger.info(f"{self.server_id} 準備啟動車輛 {vid} 的 trainer 進行訓練")

                        gpu_fraction = 1.0 / 100  #上限調整
                        try:
                            
                            trainer = VehicleTrainer(vehicle_id=vid, 
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
                                                    gpu_fraction=gpu_fraction,
                                                    position_status_dict=self.position_status_dict,
                                                    vehicle_current_edge=self.vehicle_current_edge,
                                                    vehicle_exit_edge=self.vehicle_exit_edge,
    
                                                    device=self.device)
                            trainer.start()
                            self.active_training_threads[vid]['trainer'] = trainer
                            self.logger.info(f"{self.server_id} 車輛 {vid} 的 trainer 訓練已啟動")
                            
                            def release_after_done():
                                trainer.join()
                                self.training_semaphore.release()
                                self.logger.info(f"{self.server_id} 車輛 {vid} 的 trainer 結束，GPU slot 已釋放")
                                try:
                                    if vid in self.active_training_threads:
                                        self.active_training_threads[vid]['trainer'] = None
                                        self.logger.info(f"{self.server_id} 車輛 {vid} 標記為可重新訓練")
                                    else:
                                        self.logger.info(f"{self.server_id} 車輛 {vid} 已不在 active_training_threads，略過 trainer 重設")
                                except Exception as e:
                                    self.logger.warning(f"{self.server_id} 設定 {vid} 為可重訓時發生錯誤：{e}")

                            threading.Thread(target=release_after_done, daemon=True).start()

                        except Exception as e:
                            self.logger.error(f"{self.server_id} 啟動車輛 {vid} 的 trainer 失敗，錯誤：{str(e)}，跳過這台車。")
                            self.training_semaphore.release()
                            continue

                        
                    while self.global_clock.get_time() < expected_slot_end:
                        time.sleep(0.1)
                    
                    self.collect_uploaded_models()
                    
                    with self.received_models_lock:
                        if self.received_models:
                            self.logger.info(f"{self.server_id} 正在聚合收到的車輛模型...")
                            aggregated_state_dict = aggregate_models(self.received_models, self)
                            filtered_state_dict = {
                                k: v for k, v in aggregated_state_dict.items()
                                if k in self.model.state_dict()
                            }
                            self.model.load_state_dict(filtered_state_dict)
                            self.received_models = []  # 清空
                            self.model_version += 1
                            self.logger.info(f"{self.server_id} 完成聚合，本地模型版本更新為 {self.model_version}")
                        else:
                            self.logger.info(f"{self.server_id} 本輪沒有收到車輛模型，版本不變。")
                            
        
                # 結束後，將模型上傳到 Global Server
                self.global_server.received_models.append((self.model.state_dict(), self.model_version))
                # print(f'{self.server_id} 已將本地模型版本 {self.model_version} 上傳給 Global Server')
                self.logger.info(f'{self.server_id} 已將本地模型版本 {self.model_version} 上傳給 Global Server')


                # busy waiting 等 Global Server 聚合並更新版本
                current_version = self.global_server.model_version
                while self.global_server.model_version <= current_version:
                    time.sleep(0.1)

                # 收到更新後同步
                self.update_model(self.global_server.model.state_dict(), self.global_server.model_version)
                # print(f'{self.server_id} 已更新全局模型為版本 {self.model_version} 從 Global Server')
                # self.logger.info(f'{self.server_id} 已更新全局模型為版本 {self.model_version} 從 Global Server')
                self.model_version = 1    
                   
        except Exception as e:
            self.logger.exception(f"[致命錯誤] {self.server_id} run() 發生例外 → {e}")
    
    def run_single_slot(self, selected_vehicles, expected_slot_start):
        expected_slot_end = expected_slot_start + self.waiting_time
        self.logger.info(f"{self.server_id} slot 選中的車輛: {selected_vehicles}")

        global_deadline = expected_slot_start + self.global_time

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
                    trainer.join()
                    self.training_semaphore.release()
                    self.logger.info(f"{self.server_id} {vid} 訓練完成，釋放 GPU slot")
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
                
    def run_slots(self, num_slots, slot_actions):
        round_start = self.global_clock.get_time()

        for i in range(num_slots):
            expected_slot_start = round_start + i * self.waiting_time
            self.run_single_slot(slot_actions[i], expected_slot_start)

        # 最後將模型上傳給 Global Server
        self.global_server.received_models.append((self.model.state_dict(), self.model_version))
        self.logger.info(f"{self.server_id} 已上傳模型 v{self.model_version} 給 Global Server")

        # 等待 Global Server 聚合後發佈新版
        current_version = self.global_server.model_version
        while self.global_server.model_version <= current_version:
            time.sleep(0.1)

        # 拿新版模型
        self.update_model(self.global_server.model.state_dict(), self.global_server.model_version)
        self.model_version = 1




