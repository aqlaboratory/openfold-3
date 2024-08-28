"""Utilities for PyTorch Lightning."""


def _generate_seed_sequence(
    base_seed: int, worker_id: int, global_rank: int, count: int
) -> list[int]:
    """Generates seed sequence.

    Taken from Pytorch Lightning 2.4.1 source code:
    https://github.com/Lightning-AI/pytorch-lightning/blob/f3f10d460338ca8b2901d5cd43456992131767ec/src/lightning/fabric/utilities/seed.py#L110C1-L120C17


    Generates a sequence of seeds from a base seed, worker id and rank using the linear
    congruential generator (LCG) algorithm.

    Args:
        base_seed (int):
            The initial seed used to generate the sequence.
        worker_id (int):
            The identifier for the worker.
        global_rank (int):
            The global rank of the worker in a distributed system.
        count (int):
            The number of seeds to generate.

    Returns:
        list[int]:
            A list of seeds generated
    """
    # Combine base seed, worker id and rank into a unique 64-bit number
    combined_seed = (base_seed << 32) | (worker_id << 16) | global_rank
    seeds = []
    for _ in range(count):
        # x_(n+1) = (a * x_n + c) mod m. With c=1, m=2^64 and a is D. Knuth's constant
        combined_seed = (combined_seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
        seeds.append(combined_seed)
    return seeds
