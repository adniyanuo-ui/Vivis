"""One optional job at a time, with no waiting queue or shutdown dependency."""
import threading


class BestEffortJob:
    def __init__(self, name):
        self.name=name;self.lock=threading.Lock();self.closed=threading.Event();self.thread=None

    def offer(self, function, *args):
        if self.closed.is_set() or not self.lock.acquire(blocking=False):return False
        def run():
            try:
                if not self.closed.is_set():function(*args)
            finally:self.lock.release()
        self.thread=threading.Thread(target=run,name=self.name,daemon=True)
        self.thread.start();return True

    def wait(self, timeout=None):
        if self.thread:self.thread.join(timeout)

    def close(self):
        # An optional inference/HTTP request must never delay capture/control shutdown.
        self.closed.set()
