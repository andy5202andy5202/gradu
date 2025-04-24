# global_clock.py
import threading
import time

class GlobalClock(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.current_time = 0
        self.lock = threading.Lock()

    def run(self):
        while True:
            time.sleep(1)
            with self.lock:
                self.current_time += 1

    def get_time(self):
        with self.lock:
            return self.current_time
