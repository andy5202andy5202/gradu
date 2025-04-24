# simulation_thread.py
import threading
import time
import traci

class SimulationThread(threading.Thread):
    def __init__(self, step_limit=10000, real_time_step=1.0):
        super().__init__(daemon=True)
        self.step = 0
        self.step_limit = step_limit
        self.real_time_step = real_time_step
        self.running = True

    def run(self):
        while self.step < self.step_limit and self.running:
            start_time = time.time()
            try:
                traci.simulationStep()
            except traci.exceptions.TraCIException as e:
                print(f"[SimulationThread] TraCIException: {e}")
                break

            self.step += 1
            elapsed = time.time() - start_time
            if elapsed < self.real_time_step:
                time.sleep(self.real_time_step - elapsed)

        print("[SimulationThread] 結束 SUMO 模擬。")
        self.running = False
        traci.close()

    def stop(self):
        self.running = False
