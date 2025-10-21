## Output files
These .pkl files represent manually verified instances of `StructureWithReferenceMolecules` for different input queries. They were generated at commit `84c59c6ac9b44be9323b00507afbcd1387fae913` with:

```python
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

# ---
non_canonical_peptide_query = Query(
    chains=[
        Chain(
            molecule_type=MoleculeType.PROTEIN,
            chain_ids=["A"],
            sequence="MACHINELEARNING",
            non_canonical_residues={1: "MHO", 3: "SEC"},
        )
    ]
)

out = structure_with_ref_mols_from_query(non_canonical_peptide_query)

Path("test_data/structure_from_query/structure-w-ref-mols_non-std-peptide.pkl").write_bytes(pickle.dumps(out))

# ---
standard_peptide_query = Query(
    chains=[
        Chain(
            molecule_type=MoleculeType.PROTEIN,
            chain_ids=["A"],
            sequence="MACHINELEARNING",
        )
    ]
)

out = structure_with_ref_mols_from_query(standard_peptide_query)

Path("test_data/structure_from_query/structure-w-ref-mols_std-peptide.pkl").write_bytes(pickle.dumps(out))
```
