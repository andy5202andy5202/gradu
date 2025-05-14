import os
import pickle
import threading
import torch
import time
from train_utils import aggregate_models, calculate_loss_and_accuracy, create_dataloader
from models.resnet import SmallResNet
import logging
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import atexit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



class GlobalServer(threading.Thread):
    def __init__(self, global_data_path, total_edge_servers, upload_due_to_position, T=120, device='cuda', global_clock=None):
        super().__init__(daemon=True)
        self.T = T
        self.device = device
        self.global_clock = global_clock
        self.logger = self.setup_logger()
        self.model = SmallResNet(num_classes=10).to(self.device)
        self.received_models = []
        self.model_version = 1
        self.upload_due_to_position = upload_due_to_position
        self.position_upload_history = []
        self.global_data = self.load_global_data(global_data_path)
        self.global_dataloader = create_dataloader(self.global_data, batch_size=64)
        self.total_edge_servers = total_edge_servers
        self.loss_history = []
        self.per_class_accuracy_log = []
        self.accuracy_history = []
        self.aggregating = False
        self.aggregate_lock = threading.Lock()

        atexit.register(self.safe_shutdown)

        
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            _ = self.model(dummy_input)
        
        self.logger.info("Global Server 已初始化模型，版本為 0")
        self.check_model_parameters()  #檢查初始模型參數

    def setup_logger(self):
        logger = logging.getLogger("GlobalServer")
        logger.setLevel(logging.INFO)

        # 清空已有 handler，確保每次乾淨
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

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 每次啟動覆蓋（mode='w'）
        file_handler = logging.FileHandler(os.path.join(log_dir, "GlobalServer.log"), mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger


    
    def load_global_data(self, global_data_path):
        all_data = {'data': [], 'labels': []}
        for group_id in range(4):
            file_path = os.path.join(global_data_path, f'g{group_id}_test.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    all_data['data'].extend(data['data'])
                    all_data['labels'].extend(data['labels'])
        self.logger.info(f"Global Server 測試資料集大小：{len(all_data['data'])}")
        self.logger.info(f"Global Server 測試資料標籤種類：{set(all_data['labels'])}")
        return all_data
    
    def safe_shutdown(self):
        if self.loss_history and self.accuracy_history:
            self.logger.info("程式結束，觸發 atexit → 儲存 Loss/Accuracy 圖表")
            self.save_training_plot()

    def check_model_parameters(self):
        """檢查模型參數的統計資料"""
        with torch.no_grad():
            sample_param = list(self.model.state_dict().items())[0]  # 隨便選一個參數
            param_name, param_value = sample_param
            self.logger.info(f"模型參數檢查 - {param_name}: 平均值 = {param_value.mean().item()}, 標準差 = {param_value.std().item()}")
    
    def save_training_plot(self):
        rounds = range(1, len(self.loss_history) + 1)
                # 根據總輪數動態決定 tick 間距
        num_rounds = len(rounds)
        if num_rounds <= 20:
            tick_step = 1
        elif num_rounds <= 50:
            tick_step = 2
        elif num_rounds <= 100:
            tick_step = 5
        elif num_rounds <= 200:
            tick_step = 10
        else:
            tick_step = 20
            
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Loss (左 y 軸)
        ax1.set_xlabel("Global Aggregation Round")
        ax1.set_ylabel("Loss", color='tab:blue')
        ax1.plot(rounds, self.loss_history, label="Loss", marker='.', color='tab:blue', linestyle='--')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # 讓 x 軸不要每輪都顯示 → 每隔 5 輪顯示一次 tick（或視 round 數量而定）
        # ax1.set_xticks(rounds[::5])  # 若 rounds 很多，這樣較美觀
        # ax1.set_xticklabels([str(r) for r in rounds[::5]])
    
        ax1.set_xticks(rounds[::tick_step])
        ax1.set_xticklabels([str(r) for r in rounds[::tick_step]])

        # Accuracy (右 y 軸)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy (%)", color='tab:red')
        ax2.plot(rounds, self.accuracy_history, label="Accuracy", marker='.', color='tab:red', linestyle='-')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # 額外軸顯示「非 loss 上傳」車輛數（右側第二個 y 軸）
        if hasattr(self, 'position_upload_history'):
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.1))  # 把第 2 條 y 軸右移
            ax3.set_ylabel("#Upload due to position", color='tab:green')
            ax3.plot(rounds, self.position_upload_history, label="Upload due to position", marker='.', color='tab:green', linestyle=':')
            ax3.tick_params(axis='y', labelcolor='tab:green')

        plt.title("Global Model Training Metrics Over Rounds")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("logs", exist_ok=True)
        plt.savefig("logs/global_training_metrics.png")
        plt.close()
        
    def save_per_class_accuracy_heatmap(self):
        if not self.per_class_accuracy_log:
            self.logger.warning("無 per_class_accuracy_log 資料，跳過 heatmap 繪圖")
            return

        os.makedirs("logs", exist_ok=True)

        try:
            # 強制所有類別都存在（補 0）
            full_log = []
            for acc_dict in self.per_class_accuracy_log:
                row = {i: acc_dict.get(i, 0.0) for i in range(10)}  # 確保 0~9 類都存在
                full_log.append(row)

            df = pd.DataFrame(full_log)
            df.index.name = "Global Round"
            df.columns = [f"Label {i}" for i in df.columns]

            self.logger.info(f"繪圖資料維度: {df.shape}")
            self.logger.info(f"欄位: {df.columns.tolist()}")

            plt.figure(figsize=(12, 6))
            sns.heatmap(
                df.T,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'Accuracy (%)'}
            )
            num_rounds = len(df)
            if num_rounds <= 20:
                tick_step = 1
            elif num_rounds <= 50:
                tick_step = 2
            elif num_rounds <= 100:
                tick_step = 5
            elif num_rounds <= 200:
                tick_step = 10
            else:
                tick_step = 20

            ax = plt.gca()
            ax.set_xticks(np.arange(0, num_rounds, step=tick_step))
            ax.set_xticklabels([str(i + 1) for i in range(0, num_rounds, tick_step)])
            plt.title("Per-Class Accuracy Across Global Rounds")
            plt.xlabel("Global Round")
            plt.ylabel("Class Label")
            plt.tight_layout()
            out_path = "logs/per_class_accuracy_heatmap.png"
            plt.savefig(out_path)
            plt.close()
            self.logger.info(f"Heatmap 成功儲存至 {out_path}")

        except Exception as e:
            self.logger.error(f"Heatmap 繪圖失敗：{str(e)}")

    def run(self):
        try:
            while True:
                # 等待 T 秒 (正常等待)
                round_start = self.global_clock.get_time()
                
                while self.global_clock.get_time() < round_start + self.T:
                    time.sleep(0.1)
                    
                self.logger.info("Global Server 準備開始聚合")

                self.logger.info("Global Server 開始等待所有 Edge Server 的參數上傳...")

                # Busy waiting 等待模型
                while len(self.received_models) < self.total_edge_servers:
                    self.logger.info(f"Global Server 等待中... 已收到 {len(self.received_models)}/{self.total_edge_servers} 個模型")
                    time.sleep(0.1)

                # self.logger.info("Global Server 已收到所有 Edge Server 的參數，開始聚合模型...")
                
                self.logger.info("Global Server 已收到所有 Edge Server 的參數，開始聚合模型...")

                self.aggregating = True  # <<< 標記正在聚合
                with self.aggregate_lock:
                    # 檢查收到的模型版本
                    received_versions = [version for _, version in self.received_models]
                    self.logger.info(f"Global Server 收到的模型更新次數：{received_versions}")
                    aggregated_state_dict = aggregate_models(self.received_models, self)
                    self.model.load_state_dict(aggregated_state_dict)
                    self.received_models = []
                self.aggregating = False  # <<< 聚合結束

                # 確認模型參數有沒有改變
                self.check_model_parameters()

                # 計算 Loss 和 Accuracy
                loss, accuracy, per_class_accuracy = calculate_loss_and_accuracy(
                    self.model, 
                    self.global_dataloader, 
                    criterion=torch.nn.CrossEntropyLoss(), 
                    # criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
                    device=self.device
                )
                self.logger.info(f'Global Server 聚合後模型 - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
                
                for i in range(10):  # 假設是 10 類
                    acc = per_class_accuracy.get(i, 0.0)
                    self.logger.info(f"→ 類別 {i}: Accuracy = {acc:.2f}%")

                    
                self.per_class_accuracy_log.append(per_class_accuracy)
                self.save_per_class_accuracy_heatmap()
                
                self.model_version += 1
                self.logger.info(f"Global Server 更新模型版本為 {self.model_version}")
                self.position_upload_history.append(self.upload_due_to_position['count'])
                self.logger.info(f"第 {self.model_version} 輪：{self.position_upload_history[-1]} 輛車是因為 position 而上傳")
                self.upload_due_to_position['count'] = 0  # 重置統計器
                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)
                self.save_training_plot()

                if accuracy >= 85.0:
                    try:
                        self.logger.info(f"Accuracy 達到 {accuracy:.2f}%，觸發停止條件，結束訓練流程。")
                        self.save_training_plot()
                    finally:
                        os._exit(0)

        
        finally:
            # 確保即使中斷也會執行
            self.logger.info("Global Server 儲存 Loss/Accuracy 曲線圖...")
            self.save_training_plot()

