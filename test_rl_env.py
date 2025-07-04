import numpy as np
from federated_gym_env import FederatedGymEnv
from server_factory import create_servers_fn
import multiprocessing as mp
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    env = FederatedGymEnv(create_servers_fn=create_servers_fn)

    obs, _ = env.reset()
    print("Reset 完成")

    for step in range(5):
        action = {
            i: {
                "num_slots": np.random.randint(0, env.max_slots),
                "slot_actions": np.random.randint(0, 2, (env.max_slots, env.max_vehicles)).astype(np.int8)
            }
            for i in range(env.num_agents)
        }

        obs, reward, done, _, info = env.step(action)
        print(f"[Round {info['round']}] Loss: {info['loss']} | Reward: {reward}")

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
