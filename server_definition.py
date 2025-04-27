import os
import pickle
import threading
import torch
import random
import traci
from train_utils import aggregate_models, calculate_loss_and_accuracy, create_dataloader
from models.resnet import SmallResNet
import time
from trainer import VehicleTrainer
import copy
from global_server import GlobalServer
import logging

class EdgeServer(threading.Thread):
    def __init__(self, server_id, covered_edges,cached_node_data, global_data_path, active_training_threads, global_server, global_clock=None, global_time=120, waiting_time=20, device='cuda'):
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
        self.model = SmallResNet(num_classes=10).to(self.device)
        self.received_models = []
        self.last_selection_time = time.time()
        self.model_version = 1

        # # 載入所有資料
        # self.global_data = self.load_global_data()
        # self.global_dataloader = create_dataloader(self.global_data, batch_size=64)  # 使用 DataLoader
        self.update_model(self.global_server.model.state_dict(), self.global_server.model_version)
    
    def setup_logger(self):
        logger = logging.getLogger(self.server_id)
        logger.setLevel(logging.INFO)

        # 🔥 這邊不需要檢查 handler，直接清除所有 handler（確保每次 clean）
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

        file_handler = logging.FileHandler(f"{self.server_id}.log", mode='w')  # ✅ overwrite mode
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    
    def get_model(self):
        """提供最新的模型與版本號"""
        return self.model.state_dict(), self.model_version
    
    def update_model(self, new_state_dict, new_version):
        """
        從 Global Server 收到新的模型參數並更新版本
        """
        self.model.load_state_dict(new_state_dict)
        self.model_version = new_version
        # print(f"{self.server_id} 已更新模型到全局版本 {self.model_version}")
        self.logger.info(f"{self.server_id} 已更新模型到全局版本 {self.model_version}")

    def is_in_range(self, edge_id):
        return edge_id in self.covered_edges
    
    def get_data_for_vehicle(self,road_id,vehicle_id,mode='train'):
    # 解析成真正的入口節點名稱
        try:
            node = '_'.join(road_id.split('_')[:3])
        except:
            return None

        mapping = {
            'n_1_5': 'g0', 'n_3_5': 'g1', 'n_5_5': 'g2',
            'n_1_3': 'g3', 'n_3_3': 'g4', 'n_5_3': 'g5',
            'n_1_1': 'g6', 'n_3_1': 'g7', 'n_5_1': 'g8'
        }
        group = mapping.get(node)
        if group:
            simple_node = node.replace('_', '')
            print(f"車輛 {vehicle_id} 從 {simple_node} 進入 → 分配資料集 {group}")
            # print(self.cached_node_data.keys())
            if group not in self.cached_node_data or not self.cached_node_data[group]:
                print(f"[警告] 車輛 {vehicle_id} 分配到 {group} 但資料沒載入或是空資料！")
                return None

            return self.cached_node_data.get(f"{group}")
        
        return None
        
    def run(self):
        while True:
            round_start = self.global_clock.get_time()  # 換用全局時間
            total_slots = int(self.global_time / self.waiting_time)
            
            for current_slot in range(total_slots):  
                expected_slot_start = round_start + current_slot * self.waiting_time
                expected_slot_end = expected_slot_start + self.waiting_time

                # 立即選車
                # print(f"[GlobalClock] {self.global_clock.get_time()}s - {self.server_id} Slot {current_slot + 1} 選擇車輛...")
                self.logger.info(f"{self.server_id} Slot {current_slot + 1} 選擇車輛...")

                # 1. 車輛選擇
                vehicles_in_area = []
                active_threads_copy = self.active_training_threads.copy()
                for vid, vehicle_info in active_threads_copy.items():
                    if vehicle_info.get('trainer') is None:  # 還沒開始訓練的車輛
                        if vid not in traci.vehicle.getIDList():
                            continue  # 該車輛已離開模擬，不要呼叫 getRoadID
                        try:
                            position = traci.vehicle.getRoadID(vid)
                            if position and position.startswith("n_") and self.is_in_range(position):  # position 是 edge_id
                                vehicles_in_area.append(vid)
                        except Exception as e:
                            # print(f'取得車輛 {vid} 位置時發生錯誤：{e}')
                            self.logger.info(f'取得車輛 {vid} 位置時發生錯誤：{e}')

                # print(f'{self.server_id} 範圍內的車輛: {vehicles_in_area}')
                self.logger.info(f'{self.server_id} 範圍內的車輛: {vehicles_in_area}')

                # 隨機選擇最多三輛車來訓練
                selected_vehicles = random.sample(vehicles_in_area, min(len(vehicles_in_area), 3))
                # print(f'{self.server_id} 選中的車輛: {selected_vehicles}')
                self.logger.info(f'{self.server_id} 選中的車輛: {selected_vehicles}')
                self.logger.info(f"目前系統 thread 數量: {threading.active_count()}")
                
                # 2. 啟動選中的車輛進行訓練
                for vid in selected_vehicles:
                    if vid not in self.active_training_threads:
                        self.logger.warning(f"{self.server_id} 車輛 {vid} 在選中後已離開系統，跳過。")
                        continue  # 防止 KeyError
                    
                    vehicle_info = self.active_training_threads[vid]
                    trainer_obj = vehicle_info.get('trainer')
                    
                    if trainer_obj is not None and trainer_obj.is_alive():
                        self.logger.warning(f"{self.server_id} 車輛 {vid} 的 trainer 還在跑，跳過這輛車。")
                        continue  # 不要重新開一個 thread
                    
                    
                    if 'data' not in vehicle_info:  # 車輛被選到時才載入 data
                        entry_node = vehicle_info['entry_node']
                        data_for_vehicle = self.get_data_for_vehicle(entry_node,vid)
                        
                        if data_for_vehicle is None:
                            self.logger.warning(f"{self.server_id} 車輛 {vid} 找不到對應資料，跳過。")
                            continue
                        vehicle_info['data'] = data_for_vehicle
                        
                    self.logger.info(f"{self.server_id} 準備啟動車輛 {vid} 的 trainer 進行訓練")
                    if vid in self.active_training_threads:
                        trainer = VehicleTrainer(vid, vehicle_info['data'], self, device=self.device)
                        
                        trainer.start()
                        if not trainer.started_event.wait(timeout=1.0):
                            self.logger.error(f"{self.server_id} 車輛 {vid} 的 trainer 啟動超時（超過1秒），跳過這輛車。")
                            # 不把這個卡住的 trainer 放進 active_training_threads
                            continue
                        self.active_training_threads[vid]['trainer'] = trainer
                        self.logger.info(f"車輛 {vid} 的 trainer 訓練已啟動")
                    else:
                        self.logger.warning(f"{self.server_id} 要訓練的車輛 {vid} 已被移除，略過。")

                self.logger.info(f"{self.server_id} 選完車輛後，目前系統 thread 數量: {threading.active_count()}")
                # 4. Slot 結束時進行聚合
                while self.global_clock.get_time() < expected_slot_end:
                    time.sleep(0.1)
                    
                if self.received_models:
                    # print(f'{self.server_id} 正在聚合模型...')
                    self.logger.info(f'{self.server_id} 正在聚合模型...')
                    aggregated_state_dict = aggregate_models(self.received_models,self)
                    self.model.load_state_dict(aggregated_state_dict)
                    self.received_models = []
                    self.model_version += 1
                    # print(f'{self.server_id} 完成聚合，模型版本更新為全局版本{self.global_server.model_version},本地版本{self.model_version}')
                    self.logger.info(f'{self.server_id} 完成聚合，模型版本更新為全局版本{self.global_server.model_version},本地版本{self.model_version}')
                else:
                    # print(f'{self.server_id} 本輪未收到參數，版本不更新')
                    self.logger.info(f'{self.server_id} 本輪未收到參數，版本不更新')

                # 5. Slot 完成，更新模型給下一輪車輛使用
                
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
