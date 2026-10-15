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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from openfold3.core.config.msa_pipeline_configs import MsaSampleProcessorInputInference
from openfold3.core.data.io.sequence.msa import (
    MSA_PARSER_REGISTRY,
    MsaSampleParserInference,
)
from openfold3.core.data.pipelines.sample_processing.msa import (
    MsaSampleProcessorInference,
)
from openfold3.core.data.primitives.sequence.msa import MsaArrayCollection
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


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


def main_alignments(sequence: str) -> dict[str, str]:
    """One pairable and one unpaired main MSA for a chain, both gap-free."""
    return {
        "uniprot_hits": pairable_alignment(sequence),
        "uniref90_hits": sto(
            [("query", sequence), ("hit_uniref", mutate(sequence, 1, "W"))]
        ),
    }


def paired_alignments(sequence: str) -> dict[str, str]:
    """A precomputed paired MSA; every chain gets the same depth, as required."""
    return {
        "colabfold_paired": sto(
            [
                ("query", sequence),
                ("paired_1", mutate(sequence, 4, "A")),
                ("paired_2", mutate(sequence, 5, "C")),
            ]
        )
    }


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
