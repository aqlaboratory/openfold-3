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

import json
from pathlib import Path

import click
import numpy as np


def encode_numpy_types(obj: object):
    """An encoding function for NumPy -> standard types.

    This is useful for JSON serialisation for example, which can't deal with NumPy
    types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def is_intlike_string(value: str) -> bool:
    """Check if a string represents an integer.

    Args:
        value:
            The string to check.

    Returns:
        Whether the string represents an integer.
    """
    try:
        int(value)
        return True
    except ValueError:
        return False


class JsonStrOrFile(click.ParamType):
    name = "path_or_json"

    def convert(self, value, param, ctx):
        if isinstance(value, Path):
            return value

        value = str(value)

        try:
            if value.startswith("{") or value.startswith("["):
                return json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"Invalid JSON string: {value}", param, ctx)

        with open(value) as f:
            return json.load(f)
