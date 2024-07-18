from typing import Sequence, Optional
import dataclasses

DeletionMatrix = Sequence[Sequence[int]]

@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file"""

    sequences: Sequence[str]
    deletion_matrix: DeletionMatrix
    descriptions: Optional[Sequence[str]]

    def __post_init__(self):
        if not (
            len(self.sequences) == len(self.deletion_matrix) == len(self.descriptions)
        ):
            raise ValueError("All fields for an MSA must have the same length")

    def __len__(self):
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        return Msa(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
        )