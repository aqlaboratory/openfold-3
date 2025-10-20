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

import logging
import platform
import resource

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.WARNING)


def set_ulimits():
    """Set ulimits for the process"""

    # TODO: Do this for now since this may fail on non-Linux systems,
    #  i.e. when running unit tests locally on a macOS machine.
    if platform.system() != "Linux":
        logger.info(
            f"Operating system is {platform.system()}, not Linux. "
            f"Skipping ulimit adjustments."
        )
        return

    # Set the maximum number of open files to a high value
    # This is more than is needed,
    no_file_limit = 32768

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    if no_file_limit > hard:
        logger.warning(
            f"RLIMIT_NOFILE hard limit {hard} is less than the desired limit "
            "{no_file_limit}. Lower the number of dataloader workers if issues arise."
        )
        no_file_limit = hard

    if soft < no_file_limit:
        logger.info(f"Setting RLIMIT_NOFILE to {no_file_limit} (was {soft})")
        resource.setrlimit(resource.RLIMIT_NOFILE, (no_file_limit, hard))

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    logger.info(f"Setting RLIMIT_STACK to {hard} (was {soft})")
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
