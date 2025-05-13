from pathlib import Path

from pydantic import BaseModel

from openfold3.core.data.primitives.caches.format import DatasetChainData
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.af3_all_atom.config.inference_query_format import Query


# MSA sample parser configs
class MsaSampleParserConfig(BaseModel):
    max_seq_counts: dict[str, int | float]
    moltypes: list[str]


class MsaSampleParserConfigTrain(MsaSampleParserConfig):
    alignment_array_directory: Path | None
    alignment_db_directory: Path | None
    alignment_index: dict | None
    alignments_directory: Path | None


MsaSampleParserConfigInference = MsaSampleParserConfig

# MSA sample processor configs
# class QuerySeqProcessorConfig(BaseModel):
#     pass


class PairedMsaProcessorConfig(BaseModel):
    max_rows_paired: int
    min_chains_paired_partial: int
    pairing_mask_keys: list[str]


class MainMsaProcessorConfig(BaseModel):
    aln_order: list[str]


class MsaSampleProcessorConfig(BaseModel):
    sample_parser: MsaSampleParserConfig
    # query_seq_processor: QuerySeqProcessorConfig
    paired_msa_processor: PairedMsaProcessorConfig
    main_msa_processor: MainMsaProcessorConfig


# MSA sample processor input configs
class MsaChainDataTrain(BaseModel):
    molecule_type: MoleculeType
    alignment_representative_id: str


class MsaChainDataInference(BaseModel):
    molecule_type: MoleculeType
    paired_msa_file_paths: list[Path]
    main_msa_file_paths: list[Path]
    use_msas: bool
    use_paired_msas: bool
    use_main_msas: bool


class MsaSampleProcessorInputTrain(BaseModel):
    """Dict-based expanded view of inference_query_format.query containing MSA data."""

    msa_chain_data: dict[str, MsaChainDataTrain]

    @classmethod
    def create(
        cls,
        dataset_cache_entry: DatasetChainData,
        default_moltype: MoleculeType | None = None,
        default_alignment_representative_id: str | None = None,
    ):
        msa_chain_data = {}
        for chain_id, chain_data in dataset_cache_entry.chains.items():
            if hasattr(chain_data, "molecule_type"):
                molecule_type = chain_data.molecule_type
            else:
                molecule_type = default_moltype
            if hasattr(chain_data, "alignment_representative_id"):
                alignment_representative_id = chain_data.alignment_representative_id
            else:
                alignment_representative_id = default_alignment_representative_id

            msa_chain_data[chain_id] = MsaChainDataTrain(
                molecule_type=molecule_type,
                alignment_representative_id=alignment_representative_id,
            )
        return cls(msa_chain_data=msa_chain_data)


class MsaSampleProcessorInputInference(BaseModel):
    """Dict-based expanded view of inference_query_format.query containing MSA data."""

    msa_chain_data: dict[str, MsaChainDataInference]

    @classmethod
    def create(cls, inference_query: Query):
        msa_chain_data = {}
        for chain in inference_query.chains:
            for chain_id in chain.chain_ids:
                msa_chain_data[chain_id] = MsaChainDataInference(
                    molecule_type=chain.molecule_type,
                    paired_msa_file_paths=chain.paired_msa_file_paths,
                    main_msa_file_paths=chain.main_msa_file_paths,
                    use_msas=chain.use_msas,
                    use_paired_msas=chain.use_paired_msas,
                    use_main_msas=chain.use_main_msas,
                )
        return cls(msa_chain_data=msa_chain_data)
