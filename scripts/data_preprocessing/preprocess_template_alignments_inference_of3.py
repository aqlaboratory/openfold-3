"""
Overview:
template alignment preprocessing
1. checks
   - query sequence from iqs matches alignment sequence
   - template sequence from structure matches alignment sequence
2. disambiguates query-template residue correspondences and precomputes them into a
more accessible format

Main steps
1. compute precache if precache_mode = "precompute", otherwise skip
2. iterate over inference query set entries and for each entry:
    3. iterate over chains in entry:
        4. if not already preprocessed, iterate over alignment rows:
            5. parse into TemplateHit (separate parsers for sto, a3m and m8)
                - if ambiguous alignment: realign with kalign & reindex, if fail: skip
            6. if 1st row of sto/a3m: match aln sequence to input sequence
                - if mismatch: realign with kalign & reindex, if fail: skip
            7. compute precache from template structure if not available and
            precache_mode = "greedy"
            8. filter by date and sequence indentity if specified
            9. match sequence in alignment to sequence in template precache
                - if mismatch: realign with kalign & reindex, if fail: skip
            10. create residue index map and add to template cache entry
            11. break if max number of templates reached
        12. save template cache entry to file
        13. update template paths for chain

prerequisites:
- template alignments
- template structures
- inference query set containing query sequences

outputs:
- template cache npz files for each unique chain in each query containing row idx and
    q-t residue index maps
- inference query set with updated template paths pointing to the npz template cache
    files
- template structure arrays
- (template precache files)
"""
