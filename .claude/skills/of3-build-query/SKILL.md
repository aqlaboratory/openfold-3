---
name: of3-build-query
description: "Build an OpenFold3 (OF3) query.json input file from protein/RNA/DNA sequences, ligand SMILES/CCD codes, and/or an existing CIF structure file (auto-extracts per-chain sequences and bound ligands). Use when asked to create/build/generate a query.json, set up an OF3 prediction input, add a ligand to a receptor, or turn a PDB/CIF structure into an OF3 query. For running the resulting query, use the of3-predict skill next."
---

# Building an OF3 `query.json`

## Read the format reference first

`docs/source/input_format_reference.md` is the authoritative schema — **read it before writing any `query.json`** rather than working from memory. It is the single source of truth for every field; this skill does not restate it.

| What you need | Where |
|---|---|
| Top-level `queries` structure, batching, output naming | §1–2 |
| Chain fields per molecule type (protein / RNA / DNA / ligand), incl. `non_canonical_residues`, MSA and template path fields | §3 (§3.1 protein, §3.2 RNA, §3.3 DNA, §3.4 ligand) |
| `pocket_constraint` schema + full worked example | §4 |
| Complete multi-chain example JSON + links to runnable examples | §5 |
| How the query feeds into a run, MSA/template modes | `docs/source/inference.md` §3.1–3.3 |
| Precomputed MSA / template path conventions | `docs/source/precomputed_msa_how_to.md`, `docs/source/template_how_to.md` |

Runnable starting points: `examples/example_inference_inputs/*.json` (monomer, homomer, multimer, protein-ligand, pocket-constrained, multi-query) — copying and editing the closest one is usually faster and safer than writing from scratch.

## Judgment calls the reference doesn't make for you

- `<query_key>` names the output directory/files — make it descriptive (PDB ID, target name).
- One `chain_ids` **list** with one shared `sequence` is a homomer (N identical copies); separate chain entries are for chemically different chains. Same rule for repeated identical ligand copies.
- `ccd_codes` vs `smiles`: prefer `ccd_codes` when the ligand is a known PDB component (exact, unambiguous); use `smiles` for anything not in the CCD, or when you specifically want a particular protonation/tautomer state.
- Modified residues: keep the primary `sequence` in standard 1-letter codes and add `non_canonical_residues` — do not encode the modification into `sequence`.
- Multi-residue/polymeric ligands (e.g. glycans) are experimental — a list of `ccd_codes` on one chain works in the current release, but treat it as such.
- Only add `pocket_constraint` if the user actually wants pocket-biased sampling; it changes the inference algorithm, not just metadata.
- Before writing: sequence length should equal the chain length you expect, and every character should be a valid residue letter for that molecule type.

## Chains from an existing CIF file

Not covered in the docs. Use this to re-predict / extend / add a ligand to a known structure without retyping sequences. `openfold3.core.data.primitives.structure.query_extraction.chains_from_cif` does the extraction (unit-tested against real fixtures in `openfold3/tests/core/data/primitives/structure/test_query_extraction.py`); call it inside an OF3 pixi env:

```bash
pixi run -e openfold3-base python3 -c "
import json
from openfold3.core.data.primitives.structure.query_extraction import chains_from_cif

result = chains_from_cif('/path/to/structure.cif')
for w in result.warnings:
    print('WARNING:', w)

query = {'queries': {'my_target': {'chains': [c.model_dump(mode=\"json\", exclude_none=True) for c in result.chains]}}}
json.dump(query, open('query.json', 'w'), indent=2)
"
```

What it does:
- Classifies each author chain as protein/dna/rna using biotite's CCD-based amino-acid/nucleotide detection (recognizes modified residues like MSE or 5-iodouridine as polymer residues, not ligands) and builds the 1-letter `sequence`, recording anything outside the canonical alphabet under `non_canonical_residues`.
- Splits out hetero groups (ligands) by their own structural identity — even when a ligand shares the polymer's author chain letter in the raw CIF (common in PDB files) — rather than merging them into the polymer sequence.
- **Drops common crystallization aids, buffer/cryoprotectant molecules, and monoatomic ions by default** (AlphaFold3 SI Tables 9–10 + the ion list — the same exclusion set OF3 uses for training data curation), since these are usually not the ligand of interest. Check `result.warnings` — **read this list**; if something biologically relevant got excluded (e.g. a catalytic Zn²⁺/Mg²⁺, or a cofactor), either pass `keep_excluded=True` or add that one ligand back into the JSON by hand.
- Also warns about any non-canonical residues found and any multi-residue ligand groups, so you can spot-check those before running a prediction.

**Sequence reflects only what's resolved in the file, not the full biological/construct sequence.** A chain's `sequence` is built strictly from residues with modeled coordinates — disordered termini or internal loop residues with no coordinates in the CIF are simply absent, not filled in from `entity_poly`'s canonical sequence or UniProt. A protein chain coming out a few residues shorter than expected (e.g. a missing flexible C-terminal tail) is very likely this, not a bug.

**Always read the warnings and skim the resulting chain list against what you expect** (right number of chains, right molecule types, ligand(s) present) before treating the output as final — automatic classification from a real CIF has more edge cases than hand-written queries (asymmetric units with multiple copies, unusual modified residues, ligands the exclusion list doesn't recognize).

Multiple CIFs / mixing with hand-written chains: call `chains_from_cif` per file, then merge the `chains` lists yourself (e.g. a CIF-derived receptor plus a hand-written SMILES ligand chain).

## Validate before running

Merging pieces from different sources by hand is the easiest place to introduce a schema error (duplicate `chain_ids`, wrong nesting, `pocket_constraint` placed inside a chain instead of at query level). Validate with OF3's own pydantic schema rather than guessing:

```bash
pixi run -e openfold3-base python3 -c "
from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
qs = InferenceQuerySet.from_json('query.json')
for name, q in qs.queries.items():
    print(name, [(c.chain_ids, c.molecule_type.name) for c in q.chains])
"
```
This raises a clear `ValidationError` on most structural mistakes. Note it does *not* currently enforce "protein/dna/rna needs a sequence" or "ligand needs smiles or ccd_codes" (open TODO in the schema) — double-check those two by eye.

Next step: run the query with the `of3-predict` skill.
