# train_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import traci

def train_model(model, train_data, vehicle_id, epochs=10, batch_size=32, learning_rate=0.001,
    device='cuda', loss_threshold=0.001, logger=None, early_stop_patience=5, min_delta=0.002):
    model.train()

    images = torch.tensor(train_data['data']).reshape(-1, 3, 32, 32).float()
    images = (images / 255.0 - 0.5) / 0.5
    labels = torch.tensor(train_data['labels']).long()
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
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
        # print(f"車輛 {vehicle_id} - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        if logger:
            logger.info(f"[{vehicle_id}] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
        
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
            return model, avg_loss, 'early_stop'

        # ===== 位置判斷 =====
        try:
            if vehicle_id in traci.vehicle.getIDList():
                route = traci.vehicle.getRoute(vehicle_id)
                index = traci.vehicle.getRouteIndex(vehicle_id)

                if index == len(route) - 1:
                    lane_pos = traci.vehicle.getLanePosition(vehicle_id)
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    lane_length = traci.lane.getLength(lane_id)

                    if lane_pos >= lane_length - 5:
                        print(f"車輛 {vehicle_id} 完成路徑，位置 {lane_pos:.1f}/{lane_length:.1f} → 結束訓練")
                        if logger:
                            logger.info(f"車輛 {vehicle_id} 完成路徑，位置 {lane_pos:.1f}/{lane_length:.1f} → 結束訓練")
                        return model, avg_loss, 'position'
                        
            else:
                print(f"車輛 {vehicle_id} 已離開模擬 → 結束訓練")
                if logger:
                    logger.info(f"車輛 {vehicle_id} 已離開模擬 → 結束訓練")
                return model, avg_loss, 'position'

        except traci.exceptions.TraCIException:
            print(f"TraCIException: 無法取得車輛 {vehicle_id} 的位置。該車輛已離開模擬環境→ 結束訓練")
            if logger:
                logger.info(f"TraCIException: 無法取得車輛 {vehicle_id} 的位置。該車輛已離開模擬環境→ 結束訓練")
            return model, avg_loss, 'position'

    return model, avg_loss, 'False'  # 沒提前結束，跑滿 epochs



def aggregate_models(models, self):
    if not models:
        self.logger.info(f"{getattr(self, 'server_id', 'GlobalServer')} 本輪沒收到參數")
        return None

    model_state_dicts = [m[0] for m in models]
    weights = []

    # ---------------------------------------------------
    # Global Server 聚合：使用版本號 (Ke) 作為權重
    # ---------------------------------------------------
    if getattr(self, "server_id", None) is None:
        Ke_list = [m[1] for m in models]  # Edge Server 上傳的 model_version 當作 Ke
        weight_sum = sum(Ke_list)
        weights = [k / weight_sum for k in Ke_list]

        self.logger.info(
            f"GlobalServer 使用 Ke 加權聚合：\n"
            f"→ 收到版本號 (Ke) = {Ke_list}\n"
            f"→ 權重 = {[f'{w:.3f}' for w in weights]}"
        )

    # ---------------------------------------------------
    # Edge Server 聚合：使用 staleness-aware 加權
    # ---------------------------------------------------
    else:
        versions = [m[1] for m in models]  # 每輛車的模型版本
        current_version = getattr(self, 'model_version', 0)
        delta = 1.0

        # 計算 staleness 並過濾掉比 server 還新的模型（s < 0）
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

        raw_weights = [1 / (1 + delta * s) for s in staleness_list]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]

        self.logger.info(
            f"{self.server_id} 使用 staleness-aware 聚合:\n"
            f"→ 使用的模型版本 = {[m[1] for m in filtered_models]}\n"
            f"→ Staleness = {staleness_list}\n"
            f"→ 權重 = {[f'{w:.3f}' for w in weights]}"
        )

        model_state_dicts = [m[0] for m in filtered_models]

    # ---------------------
    # 聚合模型
    # ---------------------
    aggregated_state_dict = copy.deepcopy(model_state_dicts[0])
    for key in aggregated_state_dict:
        if torch.is_floating_point(aggregated_state_dict[key]):
            aggregated_state_dict[key] = torch.zeros_like(aggregated_state_dict[key])
            for i, state_dict in enumerate(model_state_dicts):
                aggregated_state_dict[key] += weights[i] * state_dict[key].to(aggregated_state_dict[key].device)
        else:
            aggregated_state_dict[key] = model_state_dicts[0][key].to(aggregated_state_dict[key].device)

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

def calculate_loss_and_accuracy(model, dataloader, criterion, device='cuda'):
    """根據完整資料集來計算 Loss 和 Accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    average_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    model.train()  # 切回訓練模式
    return average_loss, accuracy
