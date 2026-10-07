"""Rebuild the 1fdl train-mode template preprocessing fixture.

Not collected by pytest. Run from the repository root:

    python openfold3/tests/test_data/template_preprocessing/1fdl/regenerate_fixture.py \
        --raw-m8-dir <dir with the three full ColabFold .m8 files>

Inputs: the full ColabFold pdb70 ``.m8`` hit tables for the three 1fdl chains, named
by atomworks' sequence hash (sha256[:11]). Writes, next to this script:

    representatives.fasta                          rep_id -> sequence
    raw_pdb70.m8                                   selected real rows, author chain
                                                   IDs, all three chains in one table
    template_alignments/<rep_id>/colabfold_template.m8   selected real rows only, with
                                                   author chain IDs remapped to label
                                                   chain IDs (as align-msa-server does)
    template_structures/<pdb_id>.cif.gz            hit structures (downloaded from RCSB)
    golden/<rep_id>.npz                            predict-mode cache entries, no
                                                   release-date filtering

The golden entries come from the predict-mode TemplatePreprocessor, which already
works on main: ``idx_map``/``index``/``release_date`` per template do not depend on
the preprocessing mode, so they pin what train mode must reproduce.
"""

import argparse
import gzip
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import numpy as np

from openfold3.core.data.io.structure.cif import _load_ciffile
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorSettings,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_asym_id_to_canonical_seq_dict,
    get_author_to_label_chain_ids,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Chain,
    InferenceQuerySet,
    Query,
)

FIXTURE_DIR = Path(__file__).parent

# rep_id (1fdl label_asym_id) -> 1-based rows of the full .m8 to keep. Chosen so the
# release-date assertions have templates on both sides of each cutoff:
#   A light chain:  1fdl_L (self, 1991), 1qbl_L (1998), 1jhk_L (2001) -- distinct
#                   e-values in an order that differs from alphabetical, so template
#                   ordering is actually tested
#   B heavy chain:  1fdl_H (self, 1991), 1qbl_H (1998), 3hfm_H (1989)
#   C lysozyme:     1ior_A (2001), 7ynv_A (2022), 5lyz_A (1977)
SELECTED_ROWS = {"1fdl_A": [1, 3, 5], "1fdl_B": [1, 5, 62], "1fdl_C": [1, 5, 7]}


def sequence_hash(seq: str) -> str:
    """atomworks' sequence hash, used to name the raw ColabFold .m8 files."""
    return hashlib.sha256(seq.encode()).hexdigest()[:11]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-m8-dir", type=Path, required=True)
    args = parser.parse_args()

    struct_dir = FIXTURE_DIR / "template_structures"
    struct_dir.mkdir(exist_ok=True)

    # Query sequences: canonical SEQRES of 1fdl, keyed by label_asym_id.
    _download(struct_dir, "1fdl")
    with tempfile.TemporaryDirectory() as tmp:
        query_cif = Path(tmp) / "1fdl.cif"
        query_cif.write_bytes(
            gzip.decompress((struct_dir / "1fdl.cif.gz").read_bytes())
        )
        seqs = get_asym_id_to_canonical_seq_dict(_load_ciffile(query_cif))
    rep_seqs = {f"1fdl_{chain}": seqs[chain] for chain in ("A", "B", "C")}

    with (FIXTURE_DIR / "representatives.fasta").open("w") as f:
        for rep_id, seq in rep_seqs.items():
            f.write(f">{rep_id}\n{seq}\n")

    # Cut each real .m8 down to the selected rows.
    kept_rows: dict[str, list[list[str]]] = {}
    for rep_id, row_numbers in SELECTED_ROWS.items():
        raw = (args.raw_m8_dir / f"{sequence_hash(rep_seqs[rep_id])}.m8").read_text()
        lines = raw.splitlines()
        kept_rows[rep_id] = [lines[i - 1].split("\t") for i in row_numbers]

    # The selected rows as ColabFold returns them (author chain IDs, one table for all
    # chains), the input to remap_colabfold_template_chain_ids.
    (FIXTURE_DIR / "raw_pdb70.m8").write_text(
        "".join("\t".join(row) + "\n" for rows in kept_rows.values() for row in rows)
    )

    hit_ids = {r[1].split("_")[0].lower() for rows in kept_rows.values() for r in rows}
    for pdb_id in sorted(hit_ids):
        _download(struct_dir, pdb_id)

    # ColabFold names hits by author chain ID; the template preprocessor expects label
    # chain IDs. Remap the way align-msa-server does (remap_colabfold_template_chain_ids),
    # reading the label -> author map from the structure instead of the RCSB API.
    author_to_label = {
        pdb_id: get_author_to_label_chain_ids(_label_to_author(struct_dir, pdb_id))
        for pdb_id in hit_ids
    }
    for rep_id, rows in kept_rows.items():
        for row in rows:
            entry_id, author_chain_id = row[1].split("_")
            row[1] = f"{entry_id}_{author_to_label[entry_id][author_chain_id][0]}"
        out = FIXTURE_DIR / "template_alignments" / rep_id / "colabfold_template.m8"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join("\t".join(row) + "\n" for row in rows))

    _write_golden(rep_seqs)


def _download(struct_dir: Path, pdb_id: str) -> None:
    out = struct_dir / f"{pdb_id}.cif.gz"
    if not out.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
        with urllib.request.urlopen(url, timeout=60) as r:
            out.write_bytes(r.read())


def _label_to_author(struct_dir: Path, pdb_id: str) -> dict[str, str]:
    """Polymer label -> author chain IDs, same content as RCSB's chain mapping."""
    with gzip.open(struct_dir / f"{pdb_id}.cif.gz", "rt") as f:
        scheme = pdbx.CIFFile.read(f).block["pdbx_poly_seq_scheme"]
    return {
        str(label): str(author)
        for label, author in zip(
            scheme["asym_id"].as_array(), scheme["pdb_strand_id"].as_array()
        )
    }


def _write_golden(rep_seqs: dict[str, str]) -> None:
    golden_dir = FIXTURE_DIR / "golden"
    golden_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        structures = tmp / "template_structures"
        structures.mkdir()
        for gz in (FIXTURE_DIR / "template_structures").glob("*.cif.gz"):
            (structures / gz.name.removesuffix(".gz")).write_bytes(
                gzip.decompress(gz.read_bytes())
            )
        iqs = InferenceQuerySet(
            queries={
                rep_id: Query(
                    chains=[
                        Chain(
                            molecule_type=MoleculeType.PROTEIN,
                            chain_ids=["A"],
                            sequence=seq,
                            template_alignment_file_path=FIXTURE_DIR
                            / "template_alignments"
                            / rep_id
                            / "colabfold_template.m8",
                        )
                    ]
                )
                for rep_id, seq in rep_seqs.items()
            }
        )
        settings = TemplatePreprocessorSettings(
            mode="predict",
            output_directory=tmp / "out",
            structure_directory=structures,
            fetch_missing_structures=False,
            preparse_structures=True,
            n_processes=1,
        )
        TemplatePreprocessor(input_set=iqs, config=settings)()
        array_dir = settings.structure_array_directory
        assert array_dir is not None
        arrays = sorted(
            p.relative_to(array_dir).as_posix() for p in array_dir.rglob("*.npz")
        )
        print("structure arrays:", arrays)
        for rep_id in rep_seqs:
            entry = iqs.queries[rep_id].chains[0].template_alignment_file_path
            assert entry is not None
            shutil.copyfile(entry, golden_dir / f"{rep_id}.npz")
            with np.load(entry, allow_pickle=True) as npz:
                summary = {k: npz[k].item()["release_date"] for k in npz}
            print(rep_id, summary)


if __name__ == "__main__":
    main()
