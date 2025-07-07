import traci
import time
import os
import pickle
import torch
# from models.resnet import SmallResNet
from train_utils import train_model
from server_definition import EdgeServer
from global_server import GlobalServer
from global_clock import GlobalClock  
from simulation_thread import SimulationThread
from edge_server_init import init_edge_servers
import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 的 backend（不要 Tkinter / TkAgg）
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
from multiprocessing import Manager
import threading


def preload_blurred_data(cached_blurred_data):
    groups = ['g0', 'g1', 'g2', 'g3']
    for group in groups:
        cached_blurred_data[group] = {}
        for speed in range(1, 21):
            path = os.path.join("cifar_noniid_blurred", group, f"speed{speed:02d}_train.pkl")
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        cached_blurred_data[group][speed] = pickle.load(f)
                        print(f"載入：{group} speed={speed}")
                except Exception as e:
                    print(f"[錯誤] 無法載入 {path}：{e}")



def load_vehicle_data(cached_blurred_data, group_name, max_speed):
    speed_int = int(round(max(1, min(20, max_speed))))
    try:
        return cached_blurred_data[group_name][speed_int]
    except KeyError:
        raise FileNotFoundError(f"[cache miss] {group_name} @ speed={speed_int} 尚未載入")

def get_entry_node_from_edge(edge_id):
    try:
        return '_'.join(edge_id.split('_')[:3])
    except Exception as e:
        print(f"[錯誤] edge_id 解析失敗：{edge_id}, error: {e}")
        return None

def log(global_clock, message):
    try:
        timestamp = f"[GlobalClock] {global_clock.get_time():.1f}s"
    except:
        timestamp = "[GlobalClock] ??s"
    full_msg = f"{timestamp} - {message}"
    print(full_msg, flush=True)
    with open("env.log", "a") as f:
        f.write(full_msg + "\n")
        f.flush()
    
def vehicle_manager_loop(sim_thread, active_training_threads, cached_blurred_data, 
                         upload_due_to_position, upload_due_to_early_stop, upload_due_to_global_timeout,
                         vehicle_position_status, vehicle_current_edge, vehicle_exit_edge,
                         global_clock):

    entry_to_group = {
        'n_0_1': 'g0', 'n_0_2': 'g0', 'n_0_3': 'g0', 'n_0_4': 'g0', 'n_0_5': 'g0',
        'n_1_6': 'g1', 'n_2_6': 'g1', 'n_3_6': 'g1', 'n_4_6': 'g1', 'n_5_6': 'g1',
        'n_6_1': 'g2', 'n_6_2': 'g2', 'n_6_3': 'g2', 'n_6_4': 'g2', 'n_6_5': 'g2',
        'n_1_0': 'g3', 'n_2_0': 'g3', 'n_3_0': 'g3', 'n_4_0': 'g3', 'n_5_0': 'g3',
    }

    while sim_thread.step < 7200:
        sim_thread.step_event.wait()
        sim_thread.step_event.clear()
        vehicle_ids = traci.vehicle.getIDList()
        existing_vehicles = set(vehicle_ids)

        for vid in vehicle_ids:
            try:
                if vid not in active_training_threads:
                    route = traci.vehicle.getRoute(vid)
                    if not route:
                        continue
                    start_node = get_entry_node_from_edge(route[0])
                    group = entry_to_group.get(start_node, 'g0')
                    max_speed = traci.vehicle.getMaxSpeed(vid)
                    compute_power = random.randint(1, 4)
                    route_length = len(route)
                    route_index = traci.vehicle.getRouteIndex(vid)
                    try:
                        data_for_vehicle = load_vehicle_data(cached_blurred_data, group, max_speed)
                    except Exception as e:
                        log(global_clock, f"[錯誤] 無法載入 {group} 的模糊版資料（速度 {max_speed:.1f}）→ {e}")
                        continue
                    active_training_threads[vid] = {
                        "entry_node": start_node,
                        "trainer": None,
                        "data_group": group,
                        "max_speed": max_speed,
                        "compute_power": compute_power,
                        "remaining_steps": route_length,
                        "data": data_for_vehicle
                    }
                    exit_edge = route[-1] if route else None
                    vehicle_exit_edge[vid] = exit_edge
                    vehicle_position_status[vid] = (route_index, route_length)
                    compact_id = start_node.replace('_', '')
                    log(global_clock,
                        f"車輛 {vid} 成功從 {compact_id} 產生並加入 active_training_threads，"
                        f"data={group}，速度={max_speed}，能力={compute_power}，路徑長度={route_length}")
                else:
                    route_index = traci.vehicle.getRouteIndex(vid)
                    route_length = len(traci.vehicle.getRoute(vid))
                    vehicle_position_status[vid] = (route_index, route_length)
                    try:
                        current_edge = traci.vehicle.getRoadID(vid)
                        vehicle_current_edge[vid] = current_edge
                    except traci.exceptions.TraCIException:
                        vehicle_current_edge[vid] = None
            except Exception as e:
                log(global_clock, f"[錯誤] 處理車輛 {vid} 時發生例外：{e}")

        for vid in list(active_training_threads.keys()):
            try:
                if vid not in existing_vehicles:
                    log(global_clock, f"車輛 {vid} 離開模擬環境，移除 active_training_threads。")
                    vehicle_info = active_training_threads.pop(vid, None)
                    if vehicle_info:
                        if vehicle_info.get('trainer'):
                            vehicle_info['trainer'].stop()
                            vehicle_info['trainer'].join()
                        if 'data' in vehicle_info:
                            del vehicle_info['data']
                    torch.cuda.empty_cache()
            except Exception as e:
                log(global_clock, f"[錯誤] 移除車輛 {vid} 發生例外：{e}")

    
def init_environment():
    try:
        if traci.isLoaded():
            print("[Init] traci 已啟動，關閉舊連線")
            traci.close()
    except:
        pass
    manager = Manager()
    SUMO_BINARY = 'sumo'
    CONFIG_FILE = 'grid7x7.sumocfg'
    DATA_PATH = os.path.join(os.getcwd(), 'cifar_noniid_4groups')
    

    if os.path.exists("env.log"):
        os.remove("env.log")
    veh_log_path = os.path.join("veh", "veh.log")
    if os.path.exists(veh_log_path):
        os.remove(veh_log_path)
    if os.path.exists("uploads"):
        for f in os.listdir("uploads"):
            os.remove(os.path.join("uploads", f))
    else:
        os.makedirs("uploads", exist_ok=True)


    active_training_threads = {}
    cached_blurred_data = {}
    upload_due_to_position = manager.dict({'count': 0})
    upload_due_to_early_stop = manager.dict({'count': 0})
    upload_due_to_global_timeout = manager.dict({'count': 0})
    vehicle_position_status = manager.dict()
    vehicle_current_edge = Manager().dict()
    vehicle_exit_edge = manager.dict()

    traci.start([SUMO_BINARY, '-c', CONFIG_FILE, '--collision.action', 'none'])
    preload_blurred_data(cached_blurred_data)
    global_clock = GlobalClock()
    global_clock.start()

    global_server = GlobalServer(
        global_data_path=DATA_PATH, 
        total_edge_servers=4,
        upload_due_to_position=upload_due_to_position, 
        upload_due_to_early_stop=upload_due_to_early_stop, 
        upload_due_to_global_timeout=upload_due_to_global_timeout,
        T=120, global_clock=global_clock)

    edge_servers = init_edge_servers(
        cached_blurred_data, DATA_PATH, active_training_threads, 
        global_server, global_clock, 
        upload_due_to_position, upload_due_to_early_stop, upload_due_to_global_timeout,
        vehicle_position_status, vehicle_current_edge, vehicle_exit_edge)

    global_server.edge_server_map = edge_servers

    sim_thread = SimulationThread(step_limit=7200, real_time_step=1.0)
    sim_thread.start()
    
    vehicle_thread = threading.Thread(
        target=vehicle_manager_loop,
        args=(sim_thread, active_training_threads, cached_blurred_data,
            upload_due_to_position, upload_due_to_early_stop, upload_due_to_global_timeout,
            vehicle_position_status, vehicle_current_edge, vehicle_exit_edge,
            global_clock),
        daemon=True
    )
    
    vehicle_thread.start()

    return {
        "global_server": global_server,
        "edge_servers": [edge_servers[f"Edge{i}"] for i in range(4)],
        "global_clock": global_clock,
        "sim_thread": sim_thread,
        "vehicle_thread": vehicle_thread
    }




# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)
#     main()
