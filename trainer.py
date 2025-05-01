# trainer.py
import traci
import threading
import torch
from train_utils import train_model
from models.resnet import SmallResNet
import time
import logging
import os 

class VehicleTrainer(threading.Thread):
    
    def __init__(self, vehicle_id, data_for_vehicle, edge_server, device='cuda'):
        super().__init__()
        self.vehicle_id = vehicle_id
        self.data_for_vehicle = data_for_vehicle
        self.device = device
        self.edge_server = edge_server
        self.global_clock = edge_server.global_clock
        self.started_event = threading.Event()
        
        self.logger = self.setup_shared_logger()
        self.logger.info(f"{self.vehicle_id} __init__ 開始")
        
        model_state_dict, self.model_version = self.edge_server.get_model()
        
        self.logger.info(f"{self.vehicle_id} 成功取得 model_state_dict (版本 {self.model_version})")
        
        self.model = SmallResNet(num_classes=10).to(self.device)
        
        self.logger.info(f"{self.vehicle_id} SmallResNet 模型建好並移到 {self.device}")
        
        self.model.load_state_dict(model_state_dict)
        
        self.logger.info(f"{self.vehicle_id} state_dict 載入完成")
        
        self.epoch = 30
        self.loss_threshold = 0.1
        self.batch_size = 32
        self.learning_rate = 0.005
        self.trained = False
        self.global_version = self.edge_server.global_server.model_version
        print(f"車輛 {self.vehicle_id} 拿到 {self.edge_server.server_id} 的模型版本 {self.model_version}（全局版本 {self.global_version}），開始訓練。")
        self.logger.info(f"{self.vehicle_id} __init__ 完成，trainer 準備啟動")


    def setup_shared_logger(self):
        logger = logging.getLogger("veh_shared_logger")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # 確保只設定一次 handler
        if not logger.handlers:
            os.makedirs('veh', exist_ok=True)

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

            file_handler = logging.FileHandler(os.path.join('veh', 'veh.log'), mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger




    def run(self):
        self.started_event.set()
        self.model, loss, finished = train_model(
            self.model,
            self.data_for_vehicle,
            vehicle_id=self.vehicle_id,
            epochs=self.epoch,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            loss_threshold=self.loss_threshold,
            logger=self.logger
        )

        if finished:
            self.trained = True

        # 訓練完成後回傳模型參數（上傳邏輯放這裡）
        self.model.cpu()
        torch.cuda.empty_cache()

        if self.trained:
            try:
                while self.edge_server.global_server.aggregating:
                    print(f"[等待] Global 正在聚合，車輛 {self.vehicle_id} 等待中...")
                    self.logger.info(f"[等待] Global 正在聚合，車輛 {self.vehicle_id} 等待中...")
                    time.sleep(0.1)  # 等 Global 聚合完再繼續

                current_global_version = self.edge_server.global_server.model_version
                if current_global_version == self.global_version:
                    self.edge_server.received_models.append((self.model.state_dict(), self.model_version))
                    print(f"車輛 {self.vehicle_id} 完成訓練並回傳參數給 {self.edge_server.server_id}（版本 {self.model_version}）")
                    self.logger.info(f"車輛 {self.vehicle_id} 完成訓練並回傳參數給 {self.edge_server.server_id}（版本 {self.model_version}）")
                else:
                    print(f"[棄用模型] 車輛 {self.vehicle_id} 的模型版本 {self.global_version} ≠ {self.edge_server.server_id} 當前全局版本 {current_global_version} → 不上傳")
                    self.logger.info(f"[棄用模型] 車輛 {self.vehicle_id} 的模型版本 {self.global_version} ≠ {self.edge_server.server_id} 當前全局版本 {current_global_version} → 不上傳")
               
                print(f"車輛 {self.vehicle_id} 訓練完成，回復為可選對象")
                self.logger.info(f"車輛 {self.vehicle_id} 訓練完成，回復為可選對象")
                    
            except KeyError:
                print(f"車輛 {self.vehicle_id} 已經從 active_training_threads 中被移除，無法回傳模型。")
                self.logger.info(f"車輛 {self.vehicle_id} 已經從 active_training_threads 中被移除，無法回傳模型。")
            except Exception as e:
                print(f"[{self.vehicle_id}] 上傳模型時發生未知錯誤：{e}")
                self.logger.info(f"[{self.vehicle_id}] 上傳模型時發生未知錯誤：{e}")

            finally:
                # 不管有沒有正常結束，都釋放 trainer 欄位
                if self.vehicle_id in self.edge_server.active_training_threads:
                    self.edge_server.active_training_threads[self.vehicle_id]['trainer'] = None
                    print(f"[清理] 車輛 {self.vehicle_id} 的 trainer 已經釋放完成")
                    self.logger.info(f"[清理] 車輛 {self.vehicle_id} 的 trainer 已經釋放完成")



    def stop(self):
        pass
