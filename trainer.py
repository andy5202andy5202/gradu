# trainer.py
import traci
import threading
import torch
from train_utils import train_model
from models.resnet import SmallResNet
import time
import logging
import os 
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import cv2

def unnormalize(image):
    image = image * 0.5 + 0.5
    return np.clip(image, 0, 1)

def save_image_pair(original, blurred, vehicle_id, speed, index=0):
    os.makedirs("debug_blur", exist_ok=True)

    original_img = unnormalize(original)
    blurred_img = unnormalize(blurred)

    if original_img.shape != (3, 32, 32) or blurred_img.shape != (3, 32, 32):
        raise ValueError(f"save_image_pair() → shape 錯誤：original={original_img.shape}, blurred={blurred_img.shape}")

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(np.transpose(original_img, (1, 2, 0)))
    axs[0].set_title("Original")
    axs[1].imshow(np.transpose(blurred_img, (1, 2, 0)))
    axs[1].set_title(f"Blurred (v={speed:.1f})")
    for ax in axs: ax.axis('off')

    plt.tight_layout()
    save_path = f"debug_blur/veh{vehicle_id}_img{index}.png"
    plt.savefig(save_path)
    plt.close()



def apply_motion_blur(data, max_speed, vehicle_id=None, logger=None):
    """
    根據 max_speed 對圖像資料套用高斯模糊（使用 OpenCV），保留 shape 與 dtype。
    並檢查模糊差異、是否出現 NaN。
    """
    sigma = max(0, 0.02 * max_speed - 0.5)

    # 不模糊（完全跳過 OpenCV）
    if sigma == 0:
        if logger:
            logger.info(f"{vehicle_id} sigma=0 → 跳過模糊處理")
        return data.copy()

    # 確保 shape 正確
    if data.ndim == 2 and data.shape[1] == 3072:
        data = data.reshape(-1, 3, 32, 32)
    elif data.ndim == 1 and data.shape[0] == 3072:
        data = data.reshape(1, 3, 32, 32)
    elif data.ndim == 4 and data.shape[1:] == (3, 32, 32):
        pass
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    blurred_list = []
    total_diff = []
    nan_flag = False
    ksize = int(2 * round(sigma * 3) + 1)  # 推薦 kernel size = 6*sigma+1

    for img_idx, image in enumerate(data):
        # (3, 32, 32) → (32, 32, 3) & scale to [0, 255]
        img_uint8 = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)

        blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        blurred = blurred.transpose(2, 0, 1).astype(np.float32)
        blurred = (blurred / 255.0 - 0.5) / 0.5  # 回到 [-1, 1]
        blurred_list.append(blurred)

        # 差異偵測
        diff = np.abs(blurred - image)
        total_diff.append(diff.mean())
        if np.isnan(blurred).any():
            nan_flag = True
    if logger:
        logger.info(
            f"{vehicle_id} 模糊處理完成：σ={sigma:.2f}，樣本數={len(data)}，平均差異={np.mean(total_diff):.6f}"
        )
        if nan_flag:
            logger.warning(f"{vehicle_id} 模糊後存在 NaN")

    return np.array(blurred_list, dtype=np.float32)


class VehicleTrainer(threading.Thread):
    
    def __init__(self, vehicle_id, data_for_vehicle, edge_server, upload_due_to_position_counter, device='cuda'):
        super().__init__()
        self.vehicle_id = vehicle_id
        self.data_for_vehicle = data_for_vehicle
        self.device = device
        self.edge_server = edge_server
        self.upload_due_to_position_counter = upload_due_to_position_counter
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
        
        self.epoch = 150
        self.loss_threshold = 0.1
        self.batch_size = 64
        self.learning_rate = 0.01
        self.early_stop_patience = 5
        self.min_delta = 0.005
        self.trained = False
        self.global_version = self.edge_server.global_server.model_version
        print(f"車輛 {self.vehicle_id} 拿到 {self.edge_server.server_id} 的模型版本 {self.model_version}（全局版本 {self.global_version}），開始訓練。")
        self.logger.info(f"{self.vehicle_id} __init__ 完成，trainer 準備啟動")


    def setup_shared_logger(self):
        logger = logging.getLogger("veh_shared_logger")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

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
        
        # # 模糊處理
        # try:
        #     max_speed = self.edge_server.active_training_threads[self.vehicle_id]['max_speed']
        #     original = self.data_for_vehicle['data']

        #     # 必要的轉換
        #     if original.ndim == 1 and original.shape[0] == 3072:
        #         original = original.reshape(1, 3, 32, 32)
        #     elif original.ndim == 2 and original.shape[1] == 3072:
        #         original = original.reshape(-1, 3, 32, 32)

        #     original = (original / 255.0 - 0.5) / 0.5  
        #     original = original.astype(np.float32)
            
            
        #     blurred = apply_motion_blur(original, max_speed, vehicle_id=self.vehicle_id, logger=self.logger)

        #     # 只在模糊成功後儲存圖片
        #     try:
        #         vid_num = int(self.vehicle_id.replace("veh", ""))
        #         if vid_num < 50:
        #             save_image_pair(original[0], blurred[0], self.vehicle_id, max_speed)
        #     except Exception as e:
        #         self.logger.warning(f"{self.vehicle_id} 儲存模糊圖像時出錯：{e}")
            
        #     blurred = blurred.reshape(len(blurred), -1)
        #     self.data_for_vehicle['data'] = blurred
        #     self.logger.info(f"{self.vehicle_id} 模糊處理完成（maxSpeed = {max_speed:.1f}, σ={max(0, 0.08 * max_speed - 0.5):.2f}）")

        # except Exception as e:
        #     self.logger.warning(f"{self.vehicle_id} 模糊處理失敗：{e}")


        self.model, loss, finish_reason = train_model(
            self.model,
            self.data_for_vehicle,
            vehicle_id=self.vehicle_id,
            epochs=self.epoch,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            loss_threshold=self.loss_threshold,
            logger=self.logger,
            early_stop_patience=self.early_stop_patience, 
            min_delta=self.min_delta
        )

        if finish_reason == 'position':
            self.upload_due_to_position_counter['count'] += 1
        
        if finish_reason in ('early_stop', 'position'):
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
                    # self.edge_server.received_models.append((self.model.state_dict(), self.model_version))
                    model_state_dict = self.model.state_dict()
                    model_state_dict["label_count"] = len(set(self.data_for_vehicle["labels"])) 
                    self.edge_server.received_models.append((model_state_dict, self.model_version))
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
