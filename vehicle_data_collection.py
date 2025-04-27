import traci
import time
import os
import pickle
import torch
from models.resnet import SmallResNet
from train_utils import train_model
from server_definition import EdgeServer
from global_server import GlobalServer
from global_clock import GlobalClock  
from simulation_thread import SimulationThread
from edge_server_init import init_edge_servers
import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 的 backend（不要 Tkinter / TkAgg）
import matplotlib.pyplot as plt



SUMO_BINARY = 'sumo'
CONFIG_FILE = 'grid7x7.sumocfg'
DATA_PATH = os.path.join(os.getcwd(), 'cifar_non_iid')

active_training_threads = {}  # 在這裡初始化 active_training_threads
cached_node_data = {}

def preload_node_data():
    group_names = [f'g{i}' for i in range(9)]
    for group_name in group_names:
        file_path = os.path.join(DATA_PATH, f'{group_name}_train.pkl')
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    cached_node_data[group_name] = pickle.load(f)
                    print(f"預載成功：{group_name}_train.pkl")
            except Exception as e:
                print(f"[錯誤] 載入 {group_name}_train.pkl 時失敗：{e}")

# def get_data_for_vehicle(road_id):
#     # 解析成真正的入口節點名稱
#     try:
#         node = '_'.join(road_id.split('_')[:3])
#     except:
#         return None

#     mapping = {
#         'n_1_5': 'g0', 'n_3_5': 'g1', 'n_5_5': 'g2',
#         'n_1_3': 'g3', 'n_3_3': 'g4', 'n_5_3': 'g5',
#         'n_1_1': 'g6', 'n_3_1': 'g7', 'n_5_1': 'g8'
#     }
#     group = mapping.get(node)
#     if group:
#         simple_node = node.replace('_', '')
#         print(f"車輛從 {simple_node} 進入 → 分配資料集 {group}")
#         return cached_node_data.get(group)
#     return None

import traci

def get_entry_node_from_edge(edge_id):
    """
    根據 edge_id 取得 from node。
    例如：'n_5_5_n_5_6' → 'n_5_5'
    """
    try:
        return '_'.join(edge_id.split('_')[:3])
    except Exception as e:
        print(f"[錯誤] edge_id 解析失敗：{edge_id}, error: {e}")
        return None



if __name__ == '__main__':
    traci.start([SUMO_BINARY, '-c', CONFIG_FILE, '--collision.action', 'none'])
    pre_step = 0
    step = 0
    real_time_step = 1.0
    preload_node_data()
    global_clock = GlobalClock()
    global_clock.start()  # 啟動全域時鐘
    
    global_server = GlobalServer(global_data_path=DATA_PATH, total_edge_servers=9, T=120, global_clock = global_clock)
    # 定義 Edge Servers
    edge_servers = init_edge_servers(cached_node_data,DATA_PATH, active_training_threads, global_server, global_clock)

        
    real_time_step = 1.0
        
    sim_thread = SimulationThread(step_limit=50000, real_time_step=1.0)

    sim_thread.start()
    
    global_server.start()
    
    for server in edge_servers.values():
        server.start()
        
    while sim_thread.step < 50000:
        start_time = time.time()
        
        vehicle_ids = traci.vehicle.getIDList()

        # 紀錄目前存在的車輛 ID
        existing_vehicles = set(vehicle_ids)
        
        for vid in vehicle_ids:
            try:
                if vid not in active_training_threads:
                    route = traci.vehicle.getRoute(vid)
                    if not route:
                        continue  # 確保有 route 再繼續
                    start_node = get_entry_node_from_edge(route[0])
                    # data_for_vehicle = get_data_for_vehicle(start_node)
                    if start_node not in ['n_1_5', 'n_3_5', 'n_5_5', 'n_1_3', 'n_3_3', 'n_5_3', 'n_1_1', 'n_3_1', 'n_5_1']:
                        print(f"[警告] 車輛 {vid} 生成時的 entry_node {start_node} 不在資料 mapping 裡！")
                        continue

                    active_training_threads[vid] = {
                        "entry_node": start_node,
                        "trainer": None
                    }
                    compact_id = start_node.replace('_', '')  # 把 n_3_5_n_2_5 變成 n35n25
                    print(f"車輛 {vid} 成功從 {compact_id} 產生並加入 active_training_threads。")

            except traci.exceptions.TraCIException:
                print(f"[錯誤] 無法取得車輛 {vid} 的位置")
                continue
            except Exception as e:
                print(f"[未知錯誤] 處理車輛 {vid} 時發生例外：{e}")
                continue
        # ----------------------------
        # 移除已離開的車輛（try 保護版本）
        # ----------------------------
        for vid in list(active_training_threads.keys()):
            try:
                if vid not in existing_vehicles:
                    print(f"車輛 {vid} 離開模擬環境，移除 active_training_threads。")

                    vehicle_info = active_training_threads.pop(vid, None)

                    if vehicle_info:
                        if 'trainer' in vehicle_info and vehicle_info['trainer'] is not None:
                            vehicle_info['trainer'].stop()
                            vehicle_info['trainer'].join()

                        if 'data' in vehicle_info:
                            del vehicle_info['data']

                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[錯誤] 移除車輛 {vid} 時發生例外：{e}")
                
    sim_thread.join()
    traci.close()

