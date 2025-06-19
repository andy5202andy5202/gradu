# train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import traci
from collections import defaultdict
import time

def train_model(model, train_data, vehicle_id, epochs=10, batch_size=32, learning_rate=0.001,
    device='cuda', loss_threshold=0.001, logger=None, early_stop_patience=5, delay_scale=None
    , global_clock=None, global_deadline=None, position_status_dict=None):
    model.train()

    if logger and global_clock and global_deadline:
        with global_clock.get_lock():
            current_time = global_clock.value
        logger.info(
            f"[{vehicle_id}] 收到 global deadline = {global_deadline:.2f}s，"
            f"目前時間 = {current_time:.2f}s"
        )


    images = torch.tensor(train_data['data']).reshape(-1, 3, 32, 32).float()
    # images = (images / 255.0 - 0.5) / 0.5
    labels = torch.tensor(train_data['labels']).long()
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # drop_last_flag = len(dataset) >= batch_size
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_flag)


    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #sGD + StepLR
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    best_loss = float('inf')
    no_improve_epochs = 0
    
    # logger.info(f"[{vehicle_id}] 訓練前張量範圍：min={images.min()}, max={images.max()}")
    logger.info(f"[{vehicle_id}] 訓練前張量：min={images.min()}, max={images.max()}, contig={images.is_contiguous()}, dtype={images.dtype}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        
        #SGD + StepLR
        scheduler.step()
        if delay_scale is not None:
            time.sleep(delay_scale)

        # print(f"車輛 {vehicle_id} - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        if logger:
            logger.info(f"[{vehicle_id}] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f} Delay: {delay_scale:.1f}")
        
        # ===== Loss 判斷 =====
        # if avg_loss < loss_threshold:
        #     print(f"車輛 {vehicle_id} 達到 Loss 門檻 {loss_threshold}，提前結束訓練。")
        #     if logger:
        #         logger.info(f"車輛 {vehicle_id} 達到 Loss 門檻 {loss_threshold}，提前結束訓練。")
        #     return model, avg_loss, 'loss'
        
        # ===== Early Stopping 判斷 =====
        dynamic_delta = 0.01 * best_loss if best_loss != float('inf') else 0.005

        # if best_loss - avg_loss >= min_delta:
        if best_loss - avg_loss >= dynamic_delta:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= early_stop_patience:
            print(f"車輛 {vehicle_id} 連續 {early_stop_patience} 次 Loss 無明顯改善（Δ < {dynamic_delta}），提前 Early Stop")
            if logger:
                logger.info(f"車輛 {vehicle_id} 連續 {early_stop_patience} 次 Loss 無明顯改善（Δ < {dynamic_delta}），提前 Early Stop")
            return model, avg_loss, 'early_stop', epoch + 1
        
        # Global 時間判斷
        if global_clock and global_deadline:
            with global_clock.get_lock():
                current_time = global_clock.value
            this_epoch_time = time.time() - epoch_start
            if current_time + this_epoch_time >= global_deadline:
                print(f"車輛 {vehicle_id} 預估下一輪將超過 global deadline（{global_deadline:.1f}s），提前上傳")
                if logger:
                    logger.info(f"車輛 {vehicle_id} 預估下一輪將超過 global deadline（{global_deadline:.1f}s），提前上傳")
                return model, avg_loss, 'global_timeout', epoch + 1


        # ===== 位置判斷 =====
        try:
            if position_status_dict is not None:
                pos_info = position_status_dict.get(vehicle_id, None)
                if pos_info:
                    route_index, route_length = pos_info
                    if route_index >= route_length - 1:
                        logger.info(f"車輛 {vehicle_id} 完成路徑，預估位置 {route_index}/{route_length} → 結束訓練")
                        return model, avg_loss, 'position', epoch + 1
                                    
            else:
                print(f"車輛 {vehicle_id} 已離開模擬 → 結束訓練")
                if logger:
                    logger.info(f"車輛 {vehicle_id} 已離開模擬 → 結束訓練")
                return model, avg_loss, 'position', epoch + 1

        except traci.exceptions.TraCIException:
            print(f"TraCIException: 無法取得車輛 {vehicle_id} 的位置。該車輛已離開模擬環境→ 結束訓練")
            if logger:
                logger.info(f"TraCIException: 無法取得車輛 {vehicle_id} 的位置。該車輛已離開模擬環境→ 結束訓練")
            return model, avg_loss, 'position'

    return model, avg_loss, 'False', epoch + 1 # 沒提前結束，跑滿 epochs


def aggregate_models(models, self):
    if not models:
        self.logger.info(f"{getattr(self, 'server_id', 'GlobalServer')} 本輪沒收到參數")
        return None

    model_state_dicts = [m[0] for m in models]
    weights = []

    # ============ Global Server 聚合邏輯 ============
    if getattr(self, "server_id", None) is None:
        Ke_list = [m[1] for m in models]
        weight_sum = sum(Ke_list)
        weights = [k / weight_sum for k in Ke_list]
        alpha = 0.1
        old_state_dict = self.model.state_dict()

        self.logger.info(
            f"GlobalServer 使用 Ke 加權聚合：\n"
            f"→ 收到版本號 (Ke) = {Ke_list}\n"
            f"→ 權重 = {[f'{w:.3f}' for w in weights]}"
        )

        aggregated_state_dict = {}
        for key in old_state_dict:
            if isinstance(old_state_dict[key], torch.Tensor) and torch.is_floating_point(old_state_dict[key]):
                agg = torch.zeros_like(old_state_dict[key])
                for i, state_dict in enumerate(model_state_dicts):
                    agg += weights[i] * state_dict[key].to(agg.device)
                aggregated_state_dict[key] = (1 - alpha) * old_state_dict[key] + alpha * agg
            else:
                aggregated_state_dict[key] = old_state_dict[key]

        self.logger.info(
            f"GlobalServer 使用 Ke 加權聚合 + α={alpha} 平滑：\n"
            f"→ Ke = {Ke_list}\n"
            f"→ 權重 = {[f'{w:.3f}' for w in weights]}"
        )

        return aggregated_state_dict

    # ============ Edge Server 聚合邏輯（加入 momentum 平滑） ============
    else:
        versions = [m[1] for m in models]
        current_version = getattr(self, 'model_version', 0)
        delta = 1.0

        filtered_models = []
        staleness_list = []
        for i, v in enumerate(versions):
            s = current_version - v
            if s >= 0:
                filtered_models.append(models[i])
                staleness_list.append(s)

        if not filtered_models:
            self.logger.info(f"{self.server_id} 本輪沒有合法模型參與聚合")
            self.model_version -= 1
            return self.model.state_dict()

        label_counts = []
        for model_dict, version in filtered_models:
            label_count = model_dict.get("label_count", 1)
            label_counts.append(label_count)

        C_max = max(label_counts)
        raw_weights = [
            (c / C_max) * (1 / (1 + delta * s))
            for c, s in zip(label_counts, staleness_list)
        ]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]

        
        alpha = 1  # Edge Server 的平滑係數
        
        self.logger.info(
            f"{self.server_id} 使用 label-aware + staleness 聚合 + α({alpha}) 平滑：\n"
            f"→ 使用的模型版本 = {[m[1] for m in filtered_models]}\n"
            f"→ Staleness = {staleness_list}\n"
            f"→ Label Count = {label_counts}\n"
            f"→ 權重 = {[f'{w:.3f}' for w in weights]}"
        )

        model_state_dicts = [m[0] for m in filtered_models]
        old_state_dict = self.model.state_dict()
        

        aggregated_state_dict = {}
        for key in old_state_dict:
            if isinstance(old_state_dict[key], torch.Tensor) and torch.is_floating_point(old_state_dict[key]):
                weighted_avg = torch.zeros_like(old_state_dict[key])
                for i, state_dict in enumerate(model_state_dicts):
                    weighted_avg += weights[i] * state_dict[key].to(weighted_avg.device)
                aggregated_state_dict[key] = (1 - alpha) * old_state_dict[key] + alpha * weighted_avg
            else:
                aggregated_state_dict[key] = old_state_dict[key]

        return aggregated_state_dict





def create_dataloader(global_data, batch_size=32):
    """ 將整個資料集轉成 DataLoader 格式 """
    images = np.array(global_data['data'])
    labels = np.array(global_data['labels'])
    
    # 確認資料存在
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Global data is empty. 確認載入的資料檔案正確。")

    # 將資料轉換為 Tensor
    images = torch.tensor(images).reshape(-1, 3, 32, 32).float()
    images = (images / 255.0 - 0.5) / 0.5
    labels = torch.tensor(labels).long()
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader

def calculate_loss_and_accuracy(model, dataloader, criterion, device='cuda'):
    """根據完整資料集來計算 Loss 和 Accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    average_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    per_class_accuracy = {
        label: 100 * class_correct[label] / class_total[label]
        for label in sorted(class_total.keys())
    }

    model.train()  # 切回訓練模式
    return average_loss, accuracy, per_class_accuracy

