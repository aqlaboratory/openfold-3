"""Utility context manager to measure runtime of block"""

import logging
import time


class PerformanceTimer:
    # https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        self.time_in_ms = self.time * 1000
        self.readout = f"{self.msg}... Time: {self.time:.6f} seconds"
        logging.warning(self.readout)
