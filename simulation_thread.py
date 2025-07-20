# simulation_thread.py
import threading
import time
import traci
import torch
import subprocess
import os

class SimulationThread(threading.Thread):
    def __init__(self, step_limit=10000, real_time_step=1.0):
        super().__init__(daemon=True)
        self.step = 0
        self.step_limit = step_limit
        self.real_time_step = real_time_step
        self.running = True
        self.step_event = threading.Event()

    def run(self):
        try:
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

        finally:
            self.cleanup()  # ← 保證最後呼叫 cleanup，安全關掉 traci

    def stop(self):
        self.running = False

    def cleanup(self):
        try:
            traci.close()
        except Exception as e:
            print(f"[SimulationThread] traci.close() 發生錯誤: {e}")



def show_gpu_usage(top_k=3):
    print("="*30, " GPU 使用狀況 ", "="*30)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        print(f"[GPU {i}] {name}")
        print(f"  使用中: {alloc:.1f} MB | 緩存: {reserved:.1f} MB")

    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']
        ).decode().strip().split('\n')

        current_pid = str(os.getpid())
        gpu_processes = [(pid.strip(), mem.strip()) for pid, mem in (line.split(',') for line in result)]
        
        print(f"[INFO] 共 {len(gpu_processes)} 個 GPU Process，前 {top_k} + 本程式：")

        shown = 0
        for pid, mem in gpu_processes:
            is_self = pid == current_pid
            if is_self or shown < top_k:
                flag = "(本程式)" if is_self else ""
                print(f"  PID {pid} → {mem} MB {flag}")
                if not is_self:
                    shown += 1

        if len(gpu_processes) > top_k + 1:
            print(f"  ...（略過 {len(gpu_processes) - top_k - 1} 筆）")

    except Exception as e:
        print(f"[ERROR] 讀取 nvidia-smi 失敗：{e}")

    print("="*70)
