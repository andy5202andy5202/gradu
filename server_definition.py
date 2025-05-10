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
    def __init__(self, server_id, covered_edges,cached_node_data, global_data_path, active_training_threads, global_server,upload_due_to_position, global_clock=None, global_time=120, waiting_time=40, device='cuda'):
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
        self.model = SmallResNet(num_classes=9).to(self.device)
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
    
    def get_data_for_vehicle(self, vehicle_id):
        try:
            group = self.active_training_threads[vehicle_id].get('data_group')
            if group is None:
                print(f"[警告] 車輛 {vehicle_id} 沒有 data_group")
                return None
            if group not in self.cached_node_data:
                print(f"[警告] 車輛 {vehicle_id} 分配到 {group} 但資料沒載入！")
                return None
            return self.cached_node_data[group]
        except Exception as e:
            print(f"[錯誤] EdgeServer 拿車輛 {vehicle_id} 的資料時錯誤：{e}")
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
                selected_vehicles = random.sample(vehicles_in_area,len(vehicles_in_area))
                # print(f'{self.server_id} 選中的車輛: {selected_vehicles}')
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

                    self.logger.info(f"{self.server_id} 準備啟動車輛 {vid} 的 trainer 進行訓練")

                    try:
                        trainer = VehicleTrainer(vid, vehicle_info['data'], self, upload_due_to_position_counter=self.upload_due_to_position, device=self.device)
                        trainer.start()

                        success = trainer.started_event.wait(timeout=1.0)
                        if success:
                            self.logger.info(f"{self.server_id} 車輛 {vid} 的 trainer 訓練已啟動")
                            self.active_training_threads[vid]['trainer'] = trainer
                        else:
                            self.logger.error(f"{self.server_id} 車輛 {vid} 的 trainer 啟動超時（超過1秒），強制跳過這台車。")
                            continue

                    except Exception as e:
                        self.logger.error(f"{self.server_id} 啟動車輛 {vid} 的 trainer 失敗，錯誤：{str(e)}，跳過這台車。")
                        continue

                    
                while self.global_clock.get_time() < expected_slot_end:
                    time.sleep(0.1)
                    
                if self.received_models:
                    self.logger.info(f"{self.server_id} 正在聚合收到的車輛模型...")
                    aggregated_state_dict = aggregate_models(self.received_models, self)
                    self.model.load_state_dict(aggregated_state_dict)
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
