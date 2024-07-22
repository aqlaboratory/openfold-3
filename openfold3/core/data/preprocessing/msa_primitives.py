import dataclasses
from typing import Optional, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file"""

    msa: np.array
    deletion_matrix: np.array
    headers: Optional[Sequence[str]]

    def __len__(self):
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        return Msa(
            msa=self.msa[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            headers=self.descriptions[:max_seqs],
        )


def parse_headers(msa: Msa):
    """_summary_

    Args:
        msa (Msa): _description_

    Returns:
        _type_: _description_
    """
    return msa.headers