# trainer.py
import traci
import threading
import torch
from train_utils import train_model
from models.resnet import SmallResNet
import time

class VehicleTrainer(threading.Thread):
    def __init__(self, vehicle_id, data_for_vehicle, edge_server, device='cuda'):
        super().__init__()
        self.vehicle_id = vehicle_id
        self.data_for_vehicle = data_for_vehicle
        self.device = device
        self.edge_server = edge_server
        
        model_state_dict, self.model_version = self.edge_server.get_model()
        self.model = SmallResNet(num_classes=10).to(self.device)
        self.model.load_state_dict(model_state_dict)
        self.loss_threshold = 0.001
        self.trained = False
        self.global_version = self.edge_server.global_server.model_version
        print(f"車輛 {self.vehicle_id} 拿到 {self.edge_server.server_id} 的模型版本 {self.model_version}（全局版本 {self.global_version}），開始訓練。")

    def run(self):
        self.model, loss, finished = train_model(
            self.model,
            self.data_for_vehicle,
            vehicle_id=self.vehicle_id,
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            device=self.device,
            loss_threshold=self.loss_threshold
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
                    time.sleep(0.1)  # 等 Global 聚合完再繼續

                current_global_version = self.edge_server.global_server.model_version
                if current_global_version == self.global_version:
                    self.edge_server.received_models.append((self.model.state_dict(), self.model_version))
                    print(f"車輛 {self.vehicle_id} 完成訓練並回傳參數給 {self.edge_server.server_id}（版本 {self.model_version}）")
                else:
                    print(f"[棄用模型] 車輛 {self.vehicle_id} 的模型版本 {self.global_version} ≠ {self.edge_server.server_id} 當前全局版本 {current_global_version} → 不上傳")
            except KeyError:
                print(f"車輛 {self.vehicle_id} 已經從 active_training_threads 中被移除，無法回傳模型。")
            except Exception as e:
                print(f"[{self.vehicle_id}] 上傳模型時發生未知錯誤：{e}")


    def stop(self):
        pass
