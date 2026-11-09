"""Open-mic watcher: after listen() returned, keeps reading the mic (100 ms blocks) through ASR + routing + brain prefill. New speech -> `speech` event; collect() returns all audio since start."""
import threading, time
import numpy as np


class Watch(threading.Thread):
    def __init__(self, st, thr, t_end, max_s=1.5, quiet_s=0.4, cap_s=15):
        super().__init__(daemon=True)
        self.st, self.thr, self.t_end, self.max_s, self.quiet_s, self.cap_s = st, thr, t_end, max_s, quiet_s, cap_s
        self.speech, self.done, self._stop = threading.Event(), threading.Event(), threading.Event()
        self.blocks, self.extra_s, self.t_last = [], 0.0, None  # t_last: perf_counter of the last loud block after the resume = real end of the continued speech
        self.start()

    def run(self):
        loud, quiet, t0 = 0, 0.0, time.perf_counter()
        try:
            while True:
                if not self.speech.is_set() and (self._stop.is_set() or time.perf_counter() - self.t_end > self.max_s): break
                x = self.st.read(1600)[0][:, 0].copy(); self.blocks.append(x)
                lo = float(np.sqrt((x ** 2).mean())) > self.thr
                if lo and self.speech.is_set(): self.t_last = time.perf_counter()
                if not self.speech.is_set():
                    loud = loud + 1 if lo else 0
                    if loud >= 2: self.speech.set(); quiet = 0.0
                else:
                    quiet = 0.0 if lo else quiet + 0.1
                    if quiet >= self.quiet_s or time.perf_counter() - t0 > self.cap_s: break
        except Exception: pass
        finally:
            try: self.st.stop(); self.st.close()
            except Exception: pass
            self.extra_s = sum(len(b) for b in self.blocks) / 16000 if self.speech.is_set() else 0.0
            self.done.set()

    def wait_until(self, t_perf):
        """block until perf_counter t_perf; True as soon as the user resumed talking."""
        while time.perf_counter() < t_perf:
            if self.speech.is_set(): return True
            time.sleep(0.02)
        return self.speech.is_set()

    def close(self):
        """end of the watch window (audio is about to start). No-op if the user is talking (collect() still works)."""
        if not self.speech.is_set(): self._stop.set()

    def collect(self):
        self.done.wait(self.cap_s + 5)
        return np.concatenate(self.blocks) if self.blocks else np.zeros(0, np.float32)
