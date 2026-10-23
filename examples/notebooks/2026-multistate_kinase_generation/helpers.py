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

"""Pure helper functions for the multistate-kinase OpenFold3 notebook.

These are the reusable pieces factored out of ``notebook.ipynb`` — input
sanitization, ffindex/CIF parsing, k-mer template ranking, and query-JSON /
runner-YAML / CLI construction. They depend only on the standard library plus
``numpy`` (used solely by :func:`collapse_pae`) so that ``test_helpers.py`` runs
locally without a GPU or an ``openfold3`` install.

Deliberately NOT reusing the heavier ``openfold3`` utilities
(``metadata.get_chain_to_canonical_seq_dict``, the ``Chain``/``Query`` pydantic
models): they require biotite + CCD + the full package, which would defeat the
lightweight, dependency-free testing this module is designed for.
"""

from __future__ import annotations

import pathlib
import re

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# PDB-style template id, e.g. "1BLX_A" -> (entry="1BLX", chain="A").
_TEMPLATE_ID_RE = re.compile(r"^([0-9A-Za-z]{4,5})_([A-Za-z0-9]+)$")

# One row of an HHsearch .hhr hit table: "  1 1BLX_A ... 99.9  1E-30 ...".
_HHR_HIT_RE = re.compile(r"\s*\d+\s+(\S+)\s+(\d+\.\d+)\s+(\S+)")


# --------------------------------------------------------------------------- #
# Input configuration
# --------------------------------------------------------------------------- #
def sanitize_sequence(raw: str) -> str:
    """Strip non-letters and uppercase, e.g. a pasted FASTA/whitespace blob."""
    return re.sub(r"[^A-Za-z]", "", raw).upper()


def sanitize_jobname(raw: str) -> str:
    """Coerce an arbitrary label into a filesystem-safe job name."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", raw)


# --------------------------------------------------------------------------- #
# Template selection
# --------------------------------------------------------------------------- #
def parse_ffindex_entries(text: str) -> list[str]:
    """Return the entry name (first tab field) of every valid ffindex line.

    An ffindex line is ``name<TAB>offset<TAB>length``; lines with fewer than
    three tab-separated fields are ignored.
    """
    entries = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            entries.append(parts[0])
    return entries


def normalize_tid(raw: str) -> str:
    """Drop any extension from an ffindex/HHsearch id: ``1BLX_A.a3m`` -> ``1BLX_A``."""
    return pathlib.Path(raw).stem


def parse_template_id(tid: str) -> tuple[str, str] | None:
    """Split a template id into ``(entry, chain)``, or ``None`` if malformed."""
    m = _TEMPLATE_ID_RE.match(tid)
    if not m:
        return None
    return m.group(1), m.group(2)


def extract_protein_sequence(cif_text: str, min_std_fraction: float = 0.8) -> str:
    """Return the longest protein sequence found in a CIF's multi-line blocks.

    CIF stores multi-line values (like ``entity_poly`` sequences) as
    ``\\n;<content>\\n;``. We scan every such block, keep the ones that are more
    than ``min_std_fraction`` standard amino acids, and return the longest — the
    PDB-deposited canonical sequence, which is what we rank templates against.
    """
    best = ""
    for m in re.finditer(r"\n;(.*?)\n;", cif_text, re.DOTALL):
        seq = re.sub(r"[^A-Za-z]", "", m.group(1)).upper()
        if not seq:
            continue
        std_count = sum(1 for c in seq if c in STANDARD_AA)
        if std_count / max(len(seq), 1) > min_std_fraction and len(seq) > len(best):
            best = seq
    return best


def kmer_set(seq: str, k: int = 4) -> set[str]:
    """All contiguous k-mers of ``seq`` (empty set if shorter than ``k``)."""
    if len(seq) < k:
        return set()
    return set(seq[i : i + k] for i in range(len(seq) - k + 1))


def kmer_overlap(query_kmers: set[str], target_seq: str, k: int = 4) -> float:
    """Fraction of the query's k-mers that also occur in ``target_seq``.

    Values are small in absolute terms (0.02-0.10 for close kinase homologs)
    but the *ranking* is what matters — closer homologs score higher.
    """
    if len(target_seq) < k:
        return 0.0
    target_kmers = set(target_seq[i : i + k] for i in range(len(target_seq) - k + 1))
    return len(query_kmers & target_kmers) / max(len(query_kmers), 1)


def rank_templates_by_kmer(
    query_sequence: str,
    id_seq_pairs,
    k: int = 4,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Rank ``(id, sequence)`` pairs by k-mer overlap to the query, descending.

    Pairs whose sequence is empty are skipped. Returns ``(id, score)`` tuples,
    truncated to ``top_n`` when given.
    """
    query_kmers = kmer_set(query_sequence, k)
    scored = [
        (tid, kmer_overlap(query_kmers, seq, k)) for tid, seq in id_seq_pairs if seq
    ]
    scored.sort(key=lambda x: -x[1])
    if top_n is not None:
        scored = scored[:top_n]
    return scored


def parse_hhr_hits(
    hhr_text: str, max_hits: int | None = None
) -> list[tuple[str, float, str]]:
    """Parse the hit table of an HHsearch ``.hhr`` report.

    Returns ``(template_id, probability, evalue)`` tuples for each row of the
    ``No Hit ...`` table, with the template id normalized (extension stripped)
    and truncated to ``max_hits`` when given.
    """
    hits: list[tuple[str, float, str]] = []
    in_table = False
    for line in hhr_text.splitlines():
        if line.startswith(" No Hit"):
            in_table = True
            continue
        if in_table:
            if not line.strip():
                break
            m = _HHR_HIT_RE.match(line)
            if m:
                hits.append((normalize_tid(m.group(1)), float(m.group(2)), m.group(3)))
                if max_hits is not None and len(hits) >= max_hits:
                    break
    return hits


# --------------------------------------------------------------------------- #
# Query JSON / runner YAML / inference command
# --------------------------------------------------------------------------- #
def build_query_dict(
    jobname: str,
    query_sequence: str,
    template_cif_paths: list[str],
    template_cif_chain_ids: list[str],
    chain_ids=("A",),
) -> dict:
    """Assemble an OpenFold3 query dict for CIF Direct Template Mode (PR #37)."""
    return {
        "queries": {
            jobname: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": list(chain_ids),
                        "sequence": query_sequence,
                        "template_cif_paths": template_cif_paths,
                        "template_cif_chain_ids": template_cif_chain_ids,
                    }
                ]
            }
        }
    }


def build_runner_yaml(
    use_low_mem: bool = False,
    cif_direct_min_score: float = 0.05,
    structure_format: str = "pdb",
) -> str:
    """Render the ``runner.yml`` text: PDB output, permissive CIF-direct score,
    Triton (not DeepSpeed) evoformer attention, and the ``low_mem`` preset when
    requested."""
    presets = ["predict"]
    if use_low_mem:
        presets.append("low_mem")

    text = (
        "output_writer_settings:\n"
        f"  structure_format: {structure_format}\n"
        "template_preprocessor_settings:\n"
        f"  cif_direct_min_score: {cif_direct_min_score}\n"
        "model_update:\n"
        "  presets:\n"
    )
    for p in presets:
        text += f"    - {p}\n"
    text += (
        "  custom:\n"
        "    settings:\n"
        "      memory:\n"
        "        eval:\n"
        "          use_triton_triangle_kernels: true\n"
        "          use_deepspeed_evo_attention: false\n"
    )
    return text


def build_inference_command(
    query_json_path,
    output_dir,
    num_diffusion_samples: int,
    num_model_seeds: int,
    runner_yaml,
    use_msa_server: bool,
) -> list[str]:
    """Build the ``run_openfold predict`` argv.

    ``--use-msa-server=False`` is appended only when MSA-server mode is off; the
    CLI default (True) applies when the flag is absent.
    """
    cmd = [
        "run_openfold",
        "predict",
        "--query-json",
        str(query_json_path),
        "--output-dir",
        str(output_dir),
        "--num-diffusion-samples",
        str(num_diffusion_samples),
        "--num-model-seeds",
        str(num_model_seeds),
        "--runner-yaml",
        str(runner_yaml),
    ]
    if not use_msa_server:
        cmd.append("--use-msa-server=False")
    return cmd


# --------------------------------------------------------------------------- #
# Confidence parsing
# --------------------------------------------------------------------------- #
def extract_plddt(conf: dict) -> list:
    """pLDDT list from an OF3 confidence dict, trying the known key aliases."""
    return (
        conf.get("plddt") or conf.get("residue_plddts") or conf.get("atom_plddts") or []
    )


def mean_plddt(conf: dict) -> float:
    """Mean pLDDT; ``0.0`` when no pLDDT is present (no ZeroDivisionError)."""
    plddt = extract_plddt(conf)
    return sum(plddt) / max(len(plddt), 1)


def extract_pae(conf: dict):
    """PAE (or aligned-confidence-probs) from a confidence dict, else ``None``."""
    return (
        conf.get("predicted_aligned_error")
        or conf.get("pae")
        or conf.get("aligned_confidence_probs")
    )


def collapse_pae(pae):
    """Return a 2-D PAE matrix.

    If ``pae`` is the 3-D ``aligned_confidence_probs`` tensor it is collapsed to
    the expected PAE via per-bin centers; a 2-D input is returned unchanged.
    """
    import numpy as np

    pae = np.asarray(pae)
    if pae.ndim == 3:
        bin_centers = np.linspace(0.5, 31.5, pae.shape[-1])
        pae = (pae * bin_centers).sum(-1)
    return pae


# --------------------------------------------------------------------------- #
# Thin file-I/O wrappers (convenience for the notebook; not unit-tested)
# --------------------------------------------------------------------------- #
def read_cif_sequence(path, min_std_fraction: float = 0.8) -> str:
    """Read a CIF file and return its longest canonical protein sequence."""
    try:
        text = pathlib.Path(path).read_text(errors="ignore")
    except OSError:
        return ""
    return extract_protein_sequence(text, min_std_fraction=min_std_fraction)
