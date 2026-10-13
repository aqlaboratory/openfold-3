# Copyright 2026 AlQuraishi Laboratory
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

"""Helpers for writing user-provided MSAs to disk and running them through the
inference MSA pipeline."""

import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openfold3.core.config.msa_pipeline_configs import (
    MsaKind,
    MsaSampleProcessorInputInference,
)
from openfold3.core.data.io.sequence.msa import (
    MSA_PARSER_REGISTRY,
    MsaSampleParserInference,
)
from openfold3.core.data.pipelines.sample_processing.msa import (
    MsaSampleProcessorInference,
)
from openfold3.core.data.primitives.featurization.msa import (
    MsaFeaturePrecursorOF3,
    create_msa_feature_precursor_of3,
)
from openfold3.core.data.primitives.sequence.msa import MsaArrayCollection
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    get_token_count,
    tokenize_atom_array,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import Query

# Distinct protein sequences of distinct lengths, so that a chain given another
# chain's MSA is detectable
LONG = "ACDEFGHIKLMNPQRSTVWY"
MEDIUM = "MKTAYIAKQRQISFVK"
SHORT = "GSHMLEDP"


def mutate(sequence: str, position: int, residue: str) -> str:
    """A one-residue variant of `sequence`, standing in for an MSA hit."""
    return sequence[:position] + residue + sequence[position + 1 :]


def sto(rows: list[tuple[str, str]]) -> str:
    """A minimal Stockholm alignment; the first row is the query."""
    lines = ["# STOCKHOLM 1.0", ""]
    lines += [f"{name}    {sequence}" for name, sequence in rows]
    lines += ["//", ""]
    return "\n".join(lines)


def pairable_alignment(sequence: str) -> str:
    """A gap-free alignment whose hits carry species information.

    The headers follow the format the pairing metadata parser expects
    (`tr|<id>|<id>_<species>/<start>-<end>`). SPECIESA is shared by every
    sequence, so the alignments of a heteromer's chains have rows to pair.
    """
    end = len(sequence)
    return sto(
        [
            ("query", sequence),
            (f"tr|A0A001|A0A001_SPECIESA/1-{end}", mutate(sequence, 2, "K")),
            (f"tr|A0A002|A0A002_SPECIESB/1-{end}", mutate(sequence, 3, "E")),
        ]
    )


def write_sto_files(location: Path, alignments: dict[str, str]) -> list[Path]:
    """Writes one <name>.sto file per alignment into the `location` directory and
    returns the file paths."""
    location.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, alignment in alignments.items():
        path = location / f"{name}.sto"
        path.write_text(alignment)
        paths.append(path)
    return paths


def write_sto_directory(location: Path, alignments: dict[str, str]) -> list[Path]:
    """Like `write_sto_files`, but returns the directory instead of its files."""
    write_sto_files(location, alignments)
    return [location]


def write_npz_file(location: Path, alignments: dict[str, str]) -> list[Path]:
    """Preparses all alignments into a single <location>.npz file."""
    location.parent.mkdir(parents=True, exist_ok=True)
    path = location.with_name(f"{location.name}.npz")
    parse_sto = MSA_PARSER_REGISTRY[".sto"]
    # Same on-disk format as preparse_alignments_of3.py: one pickled
    # MsaArray.to_dict() per alignment name.
    preparsed: dict[str, Any] = {
        name: parse_sto(alignment, None).to_dict()
        for name, alignment in alignments.items()
    }
    np.savez_compressed(path, **preparsed)
    return [path]


# The ways a chain's MSAs can be given in a query: a list of raw files, the
# directory containing them, or a preparsed .npz file.
MSA_FORMATS: dict[str, Callable[[Path, dict[str, str]], list[Path]]] = {
    "sto": write_sto_files,
    "sto_dir": write_sto_directory,
    "npz": write_npz_file,
}


def write_msas(fmt: str, location: Path, alignments: dict[str, str]) -> list[str]:
    """Writes `alignments` in format `fmt` and returns the paths for a query."""
    return [str(p) for p in MSA_FORMATS[fmt](location, alignments)]


# Where a chain's MSA files of one kind are written, given the root directory of
# that kind and the chain's location (see `ChainSpec`); each format then writes
# its files into that directory, or an .npz file of that name.
#   unique: one distinctly named directory per location, as OF3's own tooling
#           writes them.
#   shared: every directory is named "target", under a directory per location,
#           as in the report of gh-340.
MSA_LAYOUTS: dict[str, Callable[[Path, str], Path]] = {
    "unique": lambda root, location: root / f"msas_{location}",
    "shared": lambda root, location: root / location / "target",
}


@dataclass(frozen=True)
class ChainSpec:
    """One chain entry of a query built by `build_query`, and its MSAs.

    Attributes:
        sequence (str):
            The chain's sequence.
        main (Mapping[str, str] | Sequence[str]):
            The alignments in the chain's main MSA files: either alignment names,
            each built for the chain's own sequence, or a map from alignment name
            to the sequence the alignment is built for, i.e. its first row. Every
            alignment is a `pairable_alignment`. Empty for no main MSA files.
        paired (Mapping[str, str] | Sequence[str]):
            The same, for the chain's precomputed paired MSA files.
        chain_ids (Sequence[str] | None):
            The entry's chain IDs, e.g. "AB". By default, the letter of the
            entry's position in the query; give all entries' or none.
        molecule_type (str):
            The chain's molecule type.
        location (str | None):
            Where the entry's MSA files are written. Entries with the same
            location share files, and must then list the same alignments. By
            default, the chain IDs, so each entry has files of its own.
    """

    sequence: str
    main: Mapping[str, str] | Sequence[str] = ()
    paired: Mapping[str, str] | Sequence[str] = ()
    chain_ids: Sequence[str] | None = None
    molecule_type: str = "protein"
    location: str | None = None

    def alignments(self, msa_kind: MsaKind) -> dict[str, str]:
        """The chain's alignments of `msa_kind`, as file contents by name."""
        given = self.main if msa_kind is MsaKind.MAIN else self.paired
        if isinstance(given, str):
            raise TypeError(f"Give alignment names as a tuple, not {given!r}")
        built_for = (
            dict(given)
            if isinstance(given, Mapping)
            else dict.fromkeys(given, self.sequence)
        )
        return {name: pairable_alignment(seq) for name, seq in built_for.items()}


def build_query(
    root: Path,
    fmt: str,
    chains: Sequence[ChainSpec],
    *,
    layout: str = "unique",
    paired_layout: str | None = None,
    **query_fields: Any,
) -> Query:
    """Builds a query named "q" with one entry per chain, writing their MSA files.

    Args:
        root (Path):
            The directory to write MSA files under.
        fmt (str):
            The format to write MSA files in, a key of `MSA_FORMATS`.
        chains (Sequence[ChainSpec]):
            The query's chain entries.
        layout (str):
            How main MSA files are laid out, a key of `MSA_LAYOUTS`.
        paired_layout (str | None):
            How paired MSA files are laid out, by default as main ones.
        **query_fields:
            Further `Query` fields, e.g. `use_main_msas=False`.

    Returns:
        Query:
            The validated query.
    """
    entries = []
    for position, chain in enumerate(chains):
        chain_ids = list(chain.chain_ids or string.ascii_uppercase[position])
        entry: dict[str, Any] = {
            "molecule_type": chain.molecule_type,
            "chain_ids": chain_ids,
            "sequence": chain.sequence,
        }
        location = chain.location or "".join(chain_ids)
        for msa_kind, kind_layout in [
            (MsaKind.MAIN, layout),
            (MsaKind.PAIRED, paired_layout or layout),
        ]:
            if alignments := chain.alignments(msa_kind):
                entry[f"{msa_kind}_msa_file_paths"] = write_msas(
                    fmt, MSA_LAYOUTS[kind_layout](root / msa_kind, location), alignments
                )
        entries.append(entry)
    return Query.model_validate({"query_name": "q", "chains": entries, **query_fields})


def create_maps(query: Query, settings: MSASettings):
    """Runs only the chain -> representative mapping step."""
    input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    return MsaSampleParserInference(config=settings).create_maps(input=input)


def parse(query: Query, settings: MSASettings) -> MsaArrayCollection:
    """Runs only the inference MSA parsing step, which keeps the parsed
    alignments of each representative by name."""
    input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    return MsaSampleParserInference(config=settings)(input=input)


def process(query: Query, settings: MSASettings) -> MsaArrayCollection:
    """Runs the inference MSA parse + sample-processing pipeline."""
    input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    return MsaSampleProcessorInference(config=settings)(input=input)


def features(query: Query, settings: MSASettings) -> MsaFeaturePrecursorOF3:
    """Runs the inference MSA pipeline and featurizes its output for the query's
    structure, as the inference dataset does."""
    msa_array_collection = process(query, settings)
    atom_array, _ = structure_with_ref_mols_from_query(query=query)
    tokenize_atom_array(atom_array)
    add_token_positions(atom_array)
    return create_msa_feature_precursor_of3(
        atom_array=atom_array,
        msa_array_collection=msa_array_collection,
        n_tokens=get_token_count(atom_array),
    )
