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

from itertools import cycle, islice

import pandas as pd


def pad_to_world_size(df: pd.DataFrame, world_size: int | None = None) -> pd.DataFrame:
    """Pads a dataframe containing examples to match the world size.

    To avoid the default DistributedSampler behavior of repeating samples
    to match the world size, artificially inflate the dataset and flag the
    repeated samples so that they are ignored in the metrics.

    Args:
        df: starting collection of examples
        world_size: world_size in a distributed setting
    Returns:
        collection of examples padded to world size, with the first examples repeated.
    """
    num_examples = len(df)

    if not world_size or num_examples % world_size == 0:
        padded_df = df.copy()
        padded_df["repeated_sample"] = [False] * num_examples
        return padded_df

    # otherwise we need to pad the dataframe
    num_repeated_examples = world_size - num_examples % world_size
    repeated_indices = list(islice(cycle(range(num_examples)), num_repeated_examples))
    padded_df = pd.concat([df, df.iloc[repeated_indices]], ignore_index=True)
    padded_df["repeated_sample"] = [False] * num_examples + [
        True
    ] * num_repeated_examples

    return padded_df
