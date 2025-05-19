"""This module contains SampleProcessingPipelines for MSA features."""

from pathlib import Path
from typing import Any

from biotite.structure import AtomArray

from openfold3.core.data.format.msa import (
    MsaSampleProcessorConfig,
    MsaSampleProcessorInput,
    MsaSampleProcessorInputInference,
    MsaSampleProcessorInputTrain,
)
from openfold3.core.data.io.sequence.msa import (
    MsaSampleParser,
    MsaSampleParserInference,
    MsaSampleParserTrain,
    parse_msas_sample,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.sequence.msa import (
    MainMsaProcessor,
    MsaArray,
    MsaArrayCollection,
    PairedMsaProcessor,
    QuerySeqProcessor,
    create_main,
    create_paired,
    create_query_seqs,
    expand_paired_msas,
    find_monomer_homomer,
)


# Functional MSA processing pipeline for training - TMP, to be replaced by the
# MSASampleProcessorTrain
@log_runtime_memory(runtime_dict_key="runtime-msa-proc")
def process_msas_af3(
    atom_array: AtomArray,
    assembly_data: dict[str, dict[str, Any]],
    alignments_directory: Path | None,
    alignment_db_directory: Path | None,
    alignment_index: dict | None,
    alignment_array_directory: Path | None,
    max_seq_counts: dict[str, int | float],
    aln_order: list[str],
    max_rows_paired: int,
    min_chains_paired_partial: int,
    pairing_mask_keys: list[str],
    moltypes: list[str],
) -> MsaArrayCollection:
    """Prepares the arrays needed to create MSA feature tensors.

    Follows the logic of the AF3 SI in sections 2.2 and 2.3.
    1. Query sequence
    2. Paired sequences from UniProt
        - only if n unique protein chains > 1
        - exclude block-diagonal unpaired sequences if min_chains_paired_partial = 2
        - only protein-protein chains are paired
    3. Main MSAs for each chain with unpaired sequences from non-UniProt databases

    Note: The returned MsaProcessedCollection contains None for the query_sequences
    if there are no protein or RNA chains in the crop.

    Args:
        atom_array (AtomArray):
            The cropped (training) or full (inference) atom array.
        assembly_data (dict[str, dict[str, Any]]):
            Dict containing the alignment representatives and molecule types for each
            chain.
        alignments_directory (Path | None):
            The path to the directory containing directories containing the alignment
            files per chain. Only used if alignment_db_directory is None.
        alignment_db_directory (Path | None):
            The path to the directory containing the alignment database or its shards
            AND the alignment database superindex file. If provided, it is used over
            alignments_directory.
        alignment_index (dict | None):
            Dictionary containing the alignment index for each chain ID. Only used if
            alignment_db_directory is provided.
        alignment_array_directory (Path | None):
            The path to the directory containing the preprocessed alignment arrays.
        max_seq_counts (dict):
            Dict of max number of sequences to keep from each parsed MSA. Also used to
            determine which MSAs to parse from each chain directory.
        aln_order (list[str]):
            A list of strings matching the alignment file names, indicating the order in
            which they should be concatenated to form the main MSA.
        max_rows_paired (int):
            The maximum number of rows to keep in the paired MSA.
        min_chains_paired_partial (int):
            The minimum number of chains to keep in the paired MSA.
        pairing_mask_keys (list[str]):
            List of keys indicating which types of masks to apply during pairing.
        moltypes (list[str]):
            List of molecule types to consider in the MSA.

    Returns:
        MsaArrayCollection:
            The collection of MsaArrays in the processed state.
    """

    if (alignment_db_directory is not None) and (alignment_index is None):
        raise ValueError(
            "Alignment index must be provided if alignment_db_directory is not None."
        )

    # Parse MSAs, deletion matrices into numpy arrays and metadata into dataframes
    msa_array_collection = parse_msas_sample(
        atom_array=atom_array,
        assembly_data=assembly_data,
        moltypes=moltypes,
        alignments_directory=alignments_directory,
        alignment_db_directory=alignment_db_directory,
        alignment_index=alignment_index,
        alignment_array_directory=alignment_array_directory,
        max_seq_counts=max_seq_counts,
    )

    # Create dicts with the processed query, paired and main MSA data per chain
    if len(msa_array_collection.rep_id_to_query_seq) > 0:
        # Create query
        chain_id_to_query_seq = create_query_seqs(msa_array_collection)

        # Determine whether to do pairing
        if not find_monomer_homomer(msa_array_collection):
            # Create paired UniProt MSA arrays
            chain_id_to_paired_msa = create_paired(
                msa_array_collection,
                max_rows_paired=max_rows_paired,
                min_chains_paired_partial=min_chains_paired_partial,
                pairing_mask_keys=pairing_mask_keys,
            )
        else:
            chain_id_to_paired_msa = {}

        # Create main MSA arrays
        chain_id_to_main_msa = create_main(
            msa_array_collection=msa_array_collection,
            chain_id_to_paired_msa=chain_id_to_paired_msa,
            aln_order=aln_order,
        )

    # Skip MSA processing if there are no protein or RNA chains
    else:
        chain_id_to_query_seq, chain_id_to_paired_msa, chain_id_to_main_msa = {}, {}, {}

    # Update MsaArrayCollection with processed MSA data
    msa_array_collection.set_state_processed(
        chain_id_to_query_seq=chain_id_to_query_seq,
        chain_id_to_paired_msa=chain_id_to_paired_msa,
        chain_id_to_main_msa=chain_id_to_main_msa,
    )

    return msa_array_collection


class MsaSampleProcessor:
    """Base class for MSA sample processing."""

    def __init__(self, config: MsaSampleProcessorConfig):
        self.config = config
        self.msa_sample_parser = MsaSampleParser(config=config.sample_parser)
        self.query_seq_processor = QuerySeqProcessor()
        self.paired_msa_processor = PairedMsaProcessor(
            config=config.paired_msa_processor
        )
        self.main_msa_processor = MainMsaProcessor(config=config.main_msa_processor)

    def create_query_seq(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def create_main_msa(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
        chain_id_to_paired_msa: dict[str, MsaArray] | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def forward(self, input: MsaSampleProcessorInput) -> MsaArrayCollection:
        # Parse MSAs
        msa_array_collection = self.msa_sample_parser(input=input)

        # Create dicts with the processed query, paired and main MSA data per chain
        chain_id_to_query_seq = self.create_query_seq(
            input=input, msa_array_collection=msa_array_collection
        )
        chain_id_to_paired_msa = self.create_paired_msa(
            input=input, msa_array_collection=msa_array_collection
        )
        chain_id_to_main_msa = self.create_main_msa(
            input=input,
            msa_array_collection=msa_array_collection,
            chain_id_to_paired_msa=chain_id_to_paired_msa,
        )

        # Update MsaArrayCollection with processed MSA data
        msa_array_collection.set_state_processed(
            chain_id_to_query_seq=chain_id_to_query_seq,
            chain_id_to_paired_msa=chain_id_to_paired_msa,
            chain_id_to_main_msa=chain_id_to_main_msa,
        )

        return msa_array_collection

    def __call__(self, input: MsaSampleProcessorInput) -> MsaArrayCollection:
        return self.forward(input=input)


# TODO: test
class MsaSampleProcessorTrain(MsaSampleProcessor):
    """Pipeline for MSA sample processing for training."""

    def __init__(self, config: MsaSampleProcessorConfig):
        super().__init__(config=config)
        self.msa_sample_parser = MsaSampleParserTrain(config=config.sample_parser)

    def create_query_seq(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create query sequences from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            chain_id_to_query_seq = self.query_seq_processor(
                msa_array_collection=msa_array_collection
            )
        else:
            chain_id_to_query_seq = {}
        return chain_id_to_query_seq

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create paired MSAs from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            # Determine whether to do pairing
            if not find_monomer_homomer(msa_array_collection):
                # Create paired UniProt MSA arrays
                chain_id_to_paired_msa = self.paired_msa_processor(
                    msa_array_collection=msa_array_collection,
                )
            else:
                chain_id_to_paired_msa = {}
        else:
            chain_id_to_paired_msa = {}
        return chain_id_to_paired_msa

    def create_main_msa(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
        chain_id_to_paired_msa: dict[str, MsaArray],
    ) -> dict[str, MsaArray]:
        """Create main MSAs from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            # Create main MSA arrays
            chain_id_to_main_msa = self.main_msa_processor(
                msa_array_collection=msa_array_collection,
                chain_id_to_paired_msa=chain_id_to_paired_msa,
            )
        else:
            chain_id_to_main_msa = {}
        return chain_id_to_main_msa


class MsaSampleProcessorInference(MsaSampleProcessor):
    """Pipeline for MSA sample processing for inference."""

    def __init__(self, config: MsaSampleProcessorConfig):
        super().__init__(config=config)
        self.msa_sample_parser = MsaSampleParserInference(config=config.sample_parser)

    def create_query_seq(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create query sequences from MSA arrays."""
        if (len(msa_array_collection.rep_id_to_query_seq) > 0) & input.use_msas:
            chain_id_to_query_seq = self.query_seq_processor(
                msa_array_collection=msa_array_collection
            )
        else:
            chain_id_to_query_seq = {}
        return chain_id_to_query_seq

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create paired MSAs from MSA arrays."""
        if (
            (len(msa_array_collection.rep_id_to_query_seq) > 0)
            & input.use_msas
            & input.use_paired_msas
        ):
            # Use precomputed paired MSAs
            if len(msa_array_collection.rep_id_to_paired_msa) > 0:
                chain_id_to_paired_msa = expand_paired_msas(
                    msa_array_collection=msa_array_collection
                )
            # Pair online from main MSAs
            elif not find_monomer_homomer(msa_array_collection):
                # Create paired UniProt MSA arrays
                chain_id_to_paired_msa = self.paired_msa_processor(
                    msa_array_collection=msa_array_collection,
                )
            else:
                chain_id_to_paired_msa = {}
        else:
            chain_id_to_paired_msa = {}
        return chain_id_to_paired_msa

    def create_main_msa(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
        chain_id_to_paired_msa: dict[str, MsaArray],
    ) -> dict[str, MsaArray]:
        """Create main MSAs from MSA arrays."""
        if (
            (len(msa_array_collection.rep_id_to_query_seq) > 0)
            & input.use_msas
            & input.use_main_msas
        ):
            # Create main MSA arrays
            chain_id_to_main_msa = self.main_msa_processor(
                msa_array_collection=msa_array_collection,
                chain_id_to_paired_msa=chain_id_to_paired_msa,
            )
        else:
            chain_id_to_main_msa = {}
        return chain_id_to_main_msa
