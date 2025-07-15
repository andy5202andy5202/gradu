import numpy as np
from federated_gym_env import FederatedGymEnv
from server_factory import create_servers_fn
import multiprocessing as mp
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    env = FederatedGymEnv(create_servers_fn=create_servers_fn)

    # 建立 log 檔案
    log_path = "logs/rl_log.txt"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("[Federated RL Log]\n")

    obs, _ = env.reset()
    print("Reset 完成")

    for step in range(100):
        # 隨機產生 action
        action = {
            i: {
                "num_slots": np.random.randint(0, env.max_slots),
                "slot_actions": np.random.randint(0, 2, (env.max_slots, env.max_vehicles)).astype(np.int8)
            }
            for i in range(env.num_agents)
        }

        # 執行環境 step
        obs, reward, done, _, info = env.step(action)

        # 印出 summary
        print(f"[Round {info['round']}] Loss: {info['loss']} | Reward: {reward}")

        with open(log_path, "a") as f:
            f.write(f"--- Round {info['round']} ---\n")
            f.write(f"Global Loss: {info['loss']}\n")
            for i in range(env.num_agents):
                f.write(f"[Edge{i}] Obs:\n")

                # === Global state ===
                global_labels = [
                    "平均速度",              
                    "平均運算能力",          
                    "平均剩餘路徑比例",      
                    "有效車輛比例",          
                    "上一輪 slot 數",        
                    "global round 進度"     
                ]
                global_values = obs[i]["global"].tolist()

                f.write("  global:\n")
                for label, val in zip(global_labels, global_values):
                    f.write(f"    {label}: {val:.4f}\n")

                # 額外印出這一輪的 slot 數（action 中記錄）
                this_round_slots = action[i]["num_slots"] + 1
                f.write(f"  [說明] 本輪 action slot 數: {this_round_slots}\n")




        if (step + 1) % 20 == 0:
            print(f"\n[Round {info['round']}] 測試 reset() 開始...")
            with open(log_path, "a") as f:
                f.write(f"\n[Round {info['round']}] 測試 reset() 開始...\n")
            obs, _ = env.reset()
            print(f"[Round {info['round']}] 測試 reset() 完成\n")
            with open(log_path, "a") as f:
                f.write(f"[Round {info['round']}] 測試 reset() 完成\n\n")

        if done:
            print("模擬結束條件達成")
            break
        
    env.sim_thread.join(timeout=1)
    try:
        import traci
        traci.close()
    except:
        pass

    import torch
    torch.cuda.empty_cache()
    print("模擬結束，資源已釋放")
