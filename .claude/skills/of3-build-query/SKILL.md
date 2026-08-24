---
name: of3-build-query
description: "Build an OpenFold3 (OF3) query.json input file from protein/RNA/DNA sequences, ligand SMILES/CCD codes, and/or an existing CIF structure file (auto-extracts per-chain sequences and bound ligands). Use when asked to create/build/generate a query.json, set up an OF3 prediction input, add a ligand to a receptor, or turn a PDB/CIF structure into an OF3 query. For running the resulting query, use the of3-predict skill next."
---

# Building an OF3 `query.json`

A `query.json` has one top-level `queries` dict; each entry is one bioassembly with a `chains` list. Full schema: `docs/source/input_format_reference.md`. Runnable examples: `examples/example_inference_inputs/*.json`.

```json
{
  "queries": {
    "<query_key>": {
      "chains": [ { ... }, { ... } ]
    }
  }
}
```

`<query_key>` names the output directory/files, so make it descriptive (e.g. a PDB ID, target name).

## 1. Chains from sequence (protein / RNA / DNA)

```json
{ "molecule_type": "protein", "chain_ids": "A", "sequence": "PVLSCGEWQCL" }
{ "molecule_type": "rna",     "chain_ids": "E", "sequence": "AGCU" }
{ "molecule_type": "dna",     "chain_ids": "C", "sequence": "GACCTCT" }
```
- `chain_ids` can be a single string or a list of strings for identical repeated chains (homomer) — e.g. `"chain_ids": ["A", "B", "C", "D"]` for a tetramer, one `sequence` shared by all copies.
- Protein sequences: standard 1-letter codes, `X` for unknown, `U` for selenocysteine.
- For a residue with a non-canonical/modified variant (e.g. phosphoserine), keep the primary `sequence` in standard 1-letter codes and add `"non_canonical_residues": {"5": "SEP"}` (1-based index → 3-letter CCD code) — do not try to encode it directly in `sequence`.
- Sanity check before writing: sequence length should equal chain length; every character should be a valid residue letter for that molecule type.

## 2. Ligand chains from SMILES or CCD code

```json
{ "molecule_type": "ligand", "chain_ids": "Z", "smiles": "CC(=O)OC1C[NH+]2CCC1CC2" }
{ "molecule_type": "ligand", "chain_ids": "I", "ccd_codes": "NAG" }
```
- Exactly one of `smiles` / `ccd_codes` — never both on the same chain.
- Prefer `ccd_codes` when the ligand is a known PDB component (exact, no ambiguity); use `smiles` for anything not in the CCD, or when you specifically want a particular protonation/tautomer state.
- Multiple distinct ligand copies: repeat the `chain_ids` list like a homomer (`"chain_ids": ["F","G","H"]` with one shared `ccd_codes`/`smiles`), or add separate chain entries for chemically different ligands.
- Polymeric ligands (e.g. glycans with several linked residues) are not fully supported yet — a list of `ccd_codes` on one chain works for the current release but treat multi-residue ligands as experimental.

### Pocket constraints (optional, guides ligand placement)

Add at the **query** level (sibling of `chains`), not inside the ligand chain:
```json
"pocket_constraint": {
  "ligand_chain_id": "L",
  "pocket_residues": [["A", 2], ["A", 5], ["A", 9]],
  "max_distance": 4.0
}
```
`ligand_chain_id` must match a ligand chain's `chain_ids`. `pocket_residues` are `[chain_id, residue_id]` pairs using 1-based query sequence positions. Only add this if the user actually wants pocket-biased sampling — leave it out otherwise (it changes the inference algorithm, not just metadata).

## 3. Chains from an existing CIF file

Use this to re-predict / extend / add a ligand to a known structure without retyping sequences by hand. `openfold3.core.data.primitives.structure.query_extraction.chains_from_cif` does the extraction (unit-tested against real fixtures in `openfold3/tests/core/data/primitives/structure/test_query_extraction.py`); call it inside an OF3 pixi env, e.g.:

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
- **Drops common crystallization aids, buffer/cryoprotectant molecules, and monoatomic ions by default** (AlphaFold3 SI Tables 9–10 + the ion list — the same exclusion set OF3 itself uses for training data curation), since these are usually not the ligand of interest. Check `result.warnings` — **read this list**; if something biologically relevant got excluded (e.g. a catalytic Zn²⁺/Mg²⁺, or a cofactor), either pass `keep_excluded=True` or add that one ligand back into the JSON by hand.
- Also warns about any non-canonical residues found and any multi-residue ligand groups, so you can spot-check those before running a prediction.

**Sequence reflects only what's resolved in the file, not the full biological/construct sequence.** A chain's `sequence` is built strictly from residues with modeled coordinates — disordered termini or internal loop residues with no coordinates in the CIF are simply absent, not filled in from `entity_poly`'s canonical sequence or UniProt. A protein chain coming out a few residues shorter than you expected (e.g. missing a flexible C-terminal tail) is very likely this, not a bug — check the CIF's resolution/completeness if it matters for your use case.

**Always read the warnings and skim the resulting chain list against what you expect** (right number of chains, right molecule types, ligand(s) present) before treating the output as final — automatic classification from a real CIF has more edge cases than hand-written queries (asymmetric units with multiple copies, unusual modified residues, ligands the exclusion list doesn't recognize).

Multiple CIFs / mixing with hand-written chains: call `chains_from_cif` per file, then merge the `chains` lists yourself (e.g. combine a CIF-derived receptor with a hand-written ligand chain from SMILES).

## 4. Validate before running

Merging pieces from different sources by hand is the easiest place to introduce a schema error (duplicate `chain_ids`, wrong nesting, mixed-up `pocket_constraint`). Validate with OF3's own pydantic schema rather than guessing:

```bash
pixi run -e openfold3-base python3 -c "
from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
qs = InferenceQuerySet.from_json('query.json')
for name, q in qs.queries.items():
    print(name, [(c.chain_ids, c.molecule_type.name) for c in q.chains])
"
```
This will raise a clear `ValidationError` on most structural mistakes. Note it does *not* currently enforce "protein/dna/rna needs a sequence" or "ligand needs smiles or ccd_codes" (open TODO in the schema) — double-check those two by eye.

Next step: run the query with the `of3-predict` skill.
