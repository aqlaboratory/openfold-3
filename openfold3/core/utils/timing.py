# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility context manager to measure runtime of block"""

import logging
import time

logger = logging.getLogger(__name__)


class PerformanceTimer:
    # https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time

    def __init__(self, msg, logger=logger, level=logging.INFO):
        self.msg = msg
        self.logger = logger
        self.level = level

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        self.time_in_ms = self.time * 1000
        self.readout = f"{self.msg}... Time: {self.time:.6f} seconds"
        self.logger.log(self.level, self.readout)
