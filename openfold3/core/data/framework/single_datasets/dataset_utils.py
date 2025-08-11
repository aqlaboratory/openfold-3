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

    if world_size and num_examples % world_size != 0:
        num_repeated_examples = world_size - num_examples % world_size
        padded_df = pd.concat([df, df.iloc[:num_repeated_examples]], ignore_index=True)
        padded_df["repeated_sample"] = [False] * num_examples + [
            True
        ] * num_repeated_examples
    else:
        padded_df = df.copy()
        padded_df["repeated_sample"] = [False] * num_examples

    return padded_df
