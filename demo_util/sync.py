import threading


class SharedObject:
    def __init__(self, value=None):
        self.lock = threading.Lock()
        with self.lock:
            self._value = value

    def get(self):
        with self.lock:
            return self._value

    def set(self, value):
        with self.lock:
            self._value = value
