# simulation_thread.py
import threading
import time
import traci
import torch

class SimulationThread(threading.Thread):
    def __init__(self, step_limit=10000, real_time_step=1.0):
        super().__init__(daemon=True)
        self.step = 0
        self.step_limit = step_limit
        self.real_time_step = real_time_step
        self.running = True
        self.step_event = threading.Event()

    def run(self):
        while self.step < self.step_limit and self.running:
            print(f"[SIM STEP] {self.step}")
            start_time = time.time()
            try:
                traci.simulationStep()
            except traci.exceptions.TraCIException as e:
                print(f"[SimulationThread] TraCIException: {e}")
                break

            self.step += 1
            self.step_event.set() 
            elapsed = time.time() - start_time
            if elapsed < self.real_time_step:
                time.sleep(self.real_time_step - elapsed)
            show_gpu_usage()


        print("[SimulationThread] 結束 SUMO 模擬。")
        self.running = False
        traci.close()

    def stop(self):
        self.running = False

def show_gpu_usage():
    print("="*30, " GPU Memory Usage ", "="*30)
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)}")
        print(f"  Allocated: {round(torch.cuda.memory_allocated(i) / 1024**2, 1)} MB")
        print(f"  Cached:    {round(torch.cuda.memory_reserved(i) / 1024**2, 1)} MB")
    print("="*75)