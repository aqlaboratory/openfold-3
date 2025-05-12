from pathlib import Path

from pydantic import BaseModel

from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.af3_all_atom.config.inference_query_format import Query


class MsaSampleParserConfig(BaseModel):
    max_seq_counts: dict[str, int | float]
    moltypes: list[str]


# class QuerySeqConstructorConfig(BaseModel):
#     pass


class PairedMsaConstructorConfig(BaseModel):
    max_rows_paired: int
    min_chains_paired_partial: int
    pairing_mask_keys: list[str]


class MainMsaConstructorConfig(BaseModel):
    aln_order: list[str]


class MsaSampleProcessorConfig(BaseModel):
    sample_parser: MsaSampleParserConfig
    # query_seq_constructor: QuerySeqConstructorConfig
    paired_msa_constructor: PairedMsaConstructorConfig
    main_msa_constructor: MainMsaConstructorConfig


class MsaChainData(BaseModel):
    molecule_type: MoleculeType
    paired_msa_file_paths: list[Path]
    main_msa_file_paths: list[Path]
    use_msas: bool
    use_paired_msas: bool
    use_main_msas: bool


class MsaSampleProcessorInput(BaseModel):
    """Dict-based expanded view of inference_query_format.query containing MSA data."""

    msa_chain_data: dict[str, MsaChainData]

    @classmethod
    def from_inference_query(cls, inference_query: Query):
        msa_chain_data = {}
        for chain in inference_query.chains:
            for chain_id in chain.chain_ids:
                msa_chain_data[chain_id] = MsaChainData(
                    molecule_type=chain.molecule_type,
                    paired_msa_file_paths=chain.paired_msa_file_paths,
                    main_msa_file_paths=chain.main_msa_file_paths,
                    use_msas=chain.use_msas,
                    use_paired_msas=chain.use_paired_msas,
                    use_main_msas=chain.use_main_msas,
                )
        return cls(msa_chain_data=msa_chain_data)
