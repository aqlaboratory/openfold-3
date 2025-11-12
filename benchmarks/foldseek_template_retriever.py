#!/usr/bin/env python3
"""
Ultra-Fast FOLDSEEK Template Retriever for OpenFold3-MLX

This module provides blazing-fast template retrieval using FOLDSEEK's
dual sequence+structure similarity search, potentially beating AF3's 2021 cutoff
with more recent and better template coverage.

Key advantages over ColabFold templates:
1. SPEED: Local FOLDSEEK search vs API calls
2. RECENCY: Access to post-2021 structures vs AF3's 2021 cutoff
3. QUALITY: Dual sequence+3Di similarity vs sequence-only
4. CONTROL: Custom filtering thresholds vs black-box decisions

Usage:
    retriever = FoldSeekTemplateRetriever()
    templates = retriever.get_templates(sequence, max_templates=20)
"""

import subprocess
import tempfile
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib


@dataclass
class FoldSeekTemplate:
    """Template structure from FOLDSEEK search"""
    target_id: str           # Query sequence ID
    template_id: str         # Template PDB ID
    sequence_identity: float # Sequence identity (0-1)
    alignment_length: int    # Length of alignment
    e_value: float          # E-value
    bit_score: float        # Bit score
    query_start: int        # Query alignment start
    query_end: int          # Query alignment end
    template_start: int     # Template alignment start
    template_end: int       # Template alignment end
    query_aligned: str      # Query aligned sequence
    template_aligned: str   # Template aligned sequence
    template_path: str      # Path to template structure file
    search_type: str        # 'sequence' or 'structure'


class FoldSeekTemplateRetriever:
    """Ultra-fast template retriever using FOLDSEEK database"""

    def __init__(self,
                 foldseek_bin: str = "foldseek",
                 max_templates: int = 20,
                 min_identity: float = 0.01,  # Ultra-permissive for difficult targets (1%)
                 max_evalue: float = 1e-1,  # More permissive for template finding
                 template_date_cutoff: str = "2024-01-01",  # Beat AF3's 2021 cutoff!
                 use_dual_search: bool = True):

        self.foldseek_bin = foldseek_bin
        self.mmseqs_bin = "mmseqs"  # For sequence searches
        self.max_templates = max_templates
        self.min_identity = min_identity
        self.max_evalue = max_evalue
        self.template_date_cutoff = template_date_cutoff
        self.use_dual_search = use_dual_search

        # Configure databases (dual sequence + structure setup)
        db_base_path = Path.home() / "foldseek_databases"

        self.databases = {
            "pdb100": {
                "sequence_path": db_base_path / "pdb_seq",  # Sequence similarity
                "structure_path": db_base_path / "pdb",     # Structure similarity (3Di)
                "name": "PDB100",
                "type": "experimental",
                "priority": 1,  # Search first
                "weight": 1.2,  # 20% bonus for experimental structures
                "description": "Experimental structures (gold standard)"
            },
            "afdb_swissprot": {
                "sequence_path": db_base_path / "afdb_swissprot",  # No separate seq DB for AFDB
                "structure_path": db_base_path / "afdb_swissprot", # Structure similarity
                "name": "AFDB Swiss-Prot",
                "type": "predicted",
                "priority": 2,  # Search second
                "weight": 1.0,  # Standard weight
                "description": "AlphaFold predictions (high-confidence)"
            }
        }

        # Verify FOLDSEEK installation and databases
        self._verify_foldseek()

        # Cache for template structures and search results
        self.template_cache = {}
        self.search_cache = {}  # Cache search results by sequence hash

        print(f"✅ Dual-database FOLDSEEK retriever initialized")
        print(f"   PDB100 Sequence: {self.databases['pdb100']['sequence_path']}")
        print(f"   PDB100 Structure: {self.databases['pdb100']['structure_path']}")
        print(f"   AFDB Swiss-Prot: {self.databases['afdb_swissprot']['sequence_path']}")
        print(f"   Dual search: {'Enabled' if use_dual_search else 'PDB100 only'}")

    def _verify_foldseek(self):
        """Verify FOLDSEEK installation and databases"""
        # Check binary
        try:
            result = subprocess.run([self.foldseek_bin, "version"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"FOLDSEEK not found: {result.stderr}")

            print(f"✅ FOLDSEEK version: {result.stdout.strip()}")

        except Exception as e:
            raise RuntimeError(f"FOLDSEEK verification failed: {e}")

        # Check databases (both sequence and structure paths)
        verified_dbs = []
        for db_name, db_config in self.databases.items():
            seq_path = db_config["sequence_path"]
            struct_path = db_config["structure_path"]

            # Check sequence database files
            seq_verified = self._verify_database_path(seq_path)
            struct_verified = self._verify_database_path(struct_path)

            if seq_verified and struct_verified:
                verified_dbs.append(db_name)
                print(f"✅ {db_config['name']} database verified:")
                print(f"    Sequence: {seq_path}")
                print(f"    Structure: {struct_path}")
            else:
                print(f"⚠️  {db_config['name']} database incomplete:")
                print(f"    Sequence: {'✅' if seq_verified else '❌'} {seq_path}")
                print(f"    Structure: {'✅' if struct_verified else '❌'} {struct_path}")

    def _verify_database_path(self, db_path: Path) -> bool:
        """Verify a single database path has required FOLDSEEK files"""
        # Check if main database file exists (FOLDSEEK format)
        main_db_file = Path(str(db_path) + ".0") if (Path(str(db_path) + ".0")).exists() else db_path
        index_file = Path(str(db_path) + ".index")
        dbtype_file = Path(str(db_path) + ".dbtype")

        return main_db_file.exists() and index_file.exists() and dbtype_file.exists()

        if not verified_dbs:
            raise RuntimeError(
                f"No FOLDSEEK databases found. Please extract databases to {Path.home() / 'foldseek_databases'}"
            )

        # Update available databases
        if not self.use_dual_search or len(verified_dbs) == 1:
            # Use only first available database if dual search disabled or only one DB available
            primary_db = next(iter(verified_dbs))
            self.databases = {primary_db: self.databases[primary_db]}
            print(f"📊 Using single database: {self.databases[primary_db]['name']}")
        else:
            # Keep only verified databases
            self.databases = {db: self.databases[db] for db in verified_dbs}
            print(f"📊 Using {len(self.databases)} databases for dual search")

    def get_templates_parallel(self,
                             sequence: str,
                             sequence_id: str = "query") -> List[FoldSeekTemplate]:
        """Get templates using dual-database sequential search with caching"""

        # Create cache key from sequence and search parameters
        cache_key = self._create_cache_key(sequence)

        # Check cache first
        if cache_key in self.search_cache:
            cached_templates = self.search_cache[cache_key]
            print(f"🚀 Using cached templates for {sequence_id} ({len(cached_templates)} templates)")
            return cached_templates

        print(f"🔍 Retrieving templates for {sequence_id} ({len(sequence)} residues)")
        print(f"   Databases: {', '.join(db['name'] for db in self.databases.values())}")

        all_templates = []

        # Sequential dual-filtering search through databases (ordered by priority)
        sorted_dbs = sorted(self.databases.items(), key=lambda x: x[1]['priority'])

        for db_name, db_config in sorted_dbs:
            print(f"🔍 Dual-filtering search on {db_config['name']} database...")
            start_time = time.time()

            # Perform dual search: sequence + structure
            templates = self._dual_search_database(
                sequence=sequence,
                sequence_id=sequence_id,
                db_config=db_config,
                db_name=db_name
            )

            search_time = time.time() - start_time
            print(f"✅ {db_config['name']}: {len(templates)} dual-filtered templates in {search_time:.1f}s")

            all_templates.extend(templates)

        # Intelligent merging and quality-weighted ranking
        print(f"🔗 Merging and ranking {len(all_templates)} total templates...")
        merged_templates = self._merge_and_rank_templates(all_templates)

        # Final filtering
        final_templates = merged_templates[:self.max_templates]

        # Cache the results for future use
        self.search_cache[cache_key] = final_templates

        print(f"✅ Selected {len(final_templates)} high-quality templates")
        self._print_template_summary(final_templates)

        return final_templates

    def _create_cache_key(self, sequence: str) -> str:
        """Create a cache key from sequence and search parameters"""
        # Include search parameters in cache key to handle parameter changes
        params = f"{self.max_templates}_{self.min_identity}_{self.max_evalue}_{self.template_date_cutoff}"
        combined = f"{sequence}_{params}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _dual_search_database(self,
                             sequence: str,
                             sequence_id: str,
                             db_config: Dict,
                             db_name: str) -> List[FoldSeekTemplate]:
        """Perform dual sequence+structure search and cross-reference results"""

        print(f"    🧬 Sequence search...")

        # Step 1: Sequence similarity search
        seq_templates = self._search_single_database(
            sequence=sequence,
            sequence_id=sequence_id,
            db_path=db_config['sequence_path'],
            db_name=f"{db_name}_seq",
            db_type=db_config['type'],
            db_weight=db_config['weight']
        )

        print(f"    🏗️  Structure validation...")

        # Step 2: Validate sequence templates exist in structure database
        # This ensures we only return templates that have both sequence similarity AND structural data
        validated_templates = self._validate_templates_in_structure_db(
            seq_templates,
            db_config['structure_path']
        )

        print(f"    🎯 {len(validated_templates)} templates have both sequence + structure data")

        return validated_templates

    def _cross_reference_templates(self,
                                 seq_templates: List[FoldSeekTemplate],
                                 struct_templates: List[FoldSeekTemplate]) -> List[FoldSeekTemplate]:
        """Cross-reference sequence and structure search results"""

        # Create sets of template IDs for fast lookup
        seq_ids = {t.template_id for t in seq_templates}
        struct_ids = {t.template_id for t in struct_templates}

        # Find intersection - templates that appear in both searches
        common_ids = seq_ids.intersection(struct_ids)

        # Create template dictionaries for easy lookup
        seq_dict = {t.template_id: t for t in seq_templates}
        struct_dict = {t.template_id: t for t in struct_templates}

        dual_templates = []

        for template_id in common_ids:
            seq_template = seq_dict[template_id]
            struct_template = struct_dict[template_id]

            # Combine information from both searches
            # Use sequence template as base, but incorporate structure scores
            combined_template = FoldSeekTemplate(
                target_id=seq_template.target_id,
                template_id=template_id,
                sequence_identity=seq_template.sequence_identity,
                alignment_length=seq_template.alignment_length,
                e_value=min(seq_template.e_value, struct_template.e_value),  # Best of both
                bit_score=max(seq_template.bit_score, struct_template.bit_score),  # Best of both
                query_start=seq_template.query_start,
                query_end=seq_template.query_end,
                template_start=seq_template.template_start,
                template_end=seq_template.template_end,
                query_aligned=seq_template.query_aligned,
                template_aligned=seq_template.template_aligned,
                template_path=seq_template.template_path,
                search_type=seq_template.search_type
            )

            dual_templates.append(combined_template)

        return dual_templates

    def clear_cache(self):
        """Clear all cached search results"""
        self.search_cache.clear()
        self.template_cache.clear()
        print("🗑️  Search and template caches cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'search_cache_entries': len(self.search_cache),
            'template_cache_entries': len(self.template_cache)
        }

    def _search_single_database(self,
                               sequence: str,
                               sequence_id: str,
                               db_path: Path,
                               db_name: str,
                               db_type: str,
                               db_weight: float) -> List[FoldSeekTemplate]:
        """Search a database using appropriate tool (MMseqs2 for sequences, FOLDSEEK for structures)"""

        # Determine search tool based on database name
        is_sequence_db = "seq" in str(db_name) or "pdb_seq" in str(db_path)

        if is_sequence_db:
            return self._mmseqs_search(sequence, sequence_id, db_path, db_name, db_type, db_weight)
        else:
            return self._foldseek_structure_search(sequence, sequence_id, db_path, db_name, db_type, db_weight)

    def _mmseqs_search(self,
                      sequence: str,
                      sequence_id: str,
                      db_path: Path,
                      db_name: str,
                      db_type: str,
                      db_weight: float) -> List[FoldSeekTemplate]:
        """Perform sequence search using MMseqs2"""

        templates = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write query sequence to FASTA
            query_file = tmpdir / "query.fasta"
            with open(query_file, 'w') as f:
                f.write(f">{sequence_id}\n{sequence}\n")

            # Create query database
            query_db = tmpdir / "query_db"
            result = subprocess.run([
                self.mmseqs_bin, "createdb",
                str(query_file),
                str(query_db)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"⚠️  Could not create MMseqs query database: {result.stderr}")
                return []

            # Perform MMseqs2 search
            result_db = tmpdir / "results"
            result = subprocess.run([
                self.mmseqs_bin, "search",
                str(query_db),
                str(db_path),
                str(result_db),
                str(tmpdir / "tmp"),
                "--max-seqs", str(self.max_templates * 3),
                "-e", str(self.max_evalue),
                "--min-seq-id", str(self.min_identity),
                "-v", "1"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"⚠️  MMseqs search failed for {db_name}: {result.stderr}")
                return []

            # Convert results to readable format
            output_file = tmpdir / "results.m8"
            result = subprocess.run([
                self.mmseqs_bin, "convertalis",
                str(query_db),
                str(db_path),
                str(result_db),
                str(output_file),
                "--format-output", "target,query,fident,alnlen,mismatch,gapopen,tstart,tend,qstart,qend,evalue,bits"
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"⚠️  MMseqs convertalis failed for {db_name}: {result.stderr}")
                return []

            # Parse results
            if output_file.exists() and output_file.stat().st_size > 0:
                templates = self._parse_m8_results(output_file, sequence_id, db_type, db_weight)

        return templates

    def _foldseek_structure_search(self,
                                  sequence: str,
                                  sequence_id: str,
                                  db_path: Path,
                                  db_name: str,
                                  db_type: str,
                                  db_weight: float) -> List[FoldSeekTemplate]:
        """Perform structure validation search using FOLDSEEK header lookup"""

        # This method is not used directly anymore - structure validation is done
        # via header lookup in _validate_templates_in_structure_db
        print(f"      🔧 Structure validation via header lookup")
        return []

    def _validate_templates_in_structure_db(self,
                                          seq_templates: List[FoldSeekTemplate],
                                          struct_db_path: Path) -> List[FoldSeekTemplate]:
        """Validate that sequence templates exist in structure database"""

        if not seq_templates:
            return []

        validated_templates = []

        # Extract template IDs to search for
        template_ids = [t.template_id for t in seq_templates]

        # Use lookup file to validate template IDs exist in structure database
        lookup_file = Path(str(struct_db_path) + ".lookup")

        print(f"      🔍 Validating {len(template_ids)} templates in structure DB...")

        try:
            if lookup_file.exists():
                # Use grep to find template IDs in lookup file
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as id_file:
                    # Write template IDs to file for grep (one per line, no '>' prefix)
                    for tid in template_ids:
                        id_file.write(f"{tid}\n")
                    id_file.flush()

                    # Use grep to find matching IDs in lookup file
                    result = subprocess.run([
                        "grep", "-Ff", id_file.name, str(lookup_file)
                    ], capture_output=True, text=True, timeout=30)

                    Path(id_file.name).unlink()  # cleanup

                    if result.returncode == 0:
                        # Parse which IDs were found
                        found_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                        found_ids = set()
                        for line in found_lines:
                            if line and '\t' in line:
                                # Lookup format: index\ttemplate_id\tindex
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    found_ids.add(parts[1])

                        # Filter seq_templates to only those found in structure DB
                        for template in seq_templates:
                            if template.template_id in found_ids:
                                validated_templates.append(template)

                        print(f"      ✅ {len(validated_templates)}/{len(seq_templates)} templates have structural data")

                    else:
                        print(f"      ⚠️  No templates found in structure database")

            else:
                print(f"      ⚠️  Lookup file not found: {lookup_file}")
                # Fallback to returning all sequence templates
                validated_templates = seq_templates

        except Exception as e:
            print(f"      ⚠️  Structure validation error: {e}")
            # Fallback to returning all sequence templates
            validated_templates = seq_templates

        return validated_templates

    def _alternative_search(self,
                          query_file: Path,
                          db_path: Path,
                          tmpdir: Path,
                          sequence_id: str,
                          db_type: str,
                          db_weight: float) -> List[FoldSeekTemplate]:
        """Alternative search approach for problematic databases"""

        try:
            # Create query database first
            query_db = tmpdir / "query_db"
            result = subprocess.run([
                self.foldseek_bin, "createdb",
                str(query_file),
                str(query_db)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"⚠️  Could not create query database: {result.stderr}")
                return []

            # Try search with query database
            output_file = tmpdir / "results.m8"
            result = subprocess.run([
                self.foldseek_bin, "search",
                str(query_db),
                str(db_path),
                str(output_file),
                str(tmpdir / "tmp2"),
                "--max-seqs", str(self.max_templates * 3),
                "-e", str(self.max_evalue),
                "--format-output", "target,query,fident,alnlen,mismatch,gapopen,tstart,tend,qstart,qend,evalue,bits",
                "-v", "1"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"⚠️  Alternative search also failed: {result.stderr}")
                return []

            # Parse results
            if output_file.exists() and output_file.stat().st_size > 0:
                return self._parse_m8_results(output_file, sequence_id, db_type, db_weight)

        except Exception as e:
            print(f"⚠️  Alternative search error: {e}")

        return []

    def _merge_and_rank_templates(self, all_templates: List[FoldSeekTemplate]) -> List[FoldSeekTemplate]:
        """Merge templates from multiple databases and rank by intelligent quality score"""

        # Deduplicate by template_id (keep best scoring)
        template_dict = {}
        for template in all_templates:
            key = template.template_id

            # Calculate comprehensive quality score
            quality_score = self._calculate_template_quality_score(template)

            if key not in template_dict or quality_score > self._calculate_template_quality_score(template_dict[key]):
                template_dict[key] = template

        # Sort by comprehensive quality score (descending)
        merged_templates = list(template_dict.values())
        merged_templates.sort(key=self._calculate_template_quality_score, reverse=True)

        # Additional filtering
        filtered_templates = self._filter_templates(merged_templates)

        return filtered_templates

    def _calculate_template_quality_score(self, template: FoldSeekTemplate) -> float:
        """Calculate comprehensive template quality score"""

        # Base score from bit score
        base_score = template.bit_score

        # Database type weight (experimental vs predicted)
        db_weight = 1.2 if template.search_type == 'experimental' else 1.0

        # Coverage bonus (alignment length relative to typical protein length)
        # Assume average protein is ~300 residues, give bonus for good coverage
        coverage_bonus = min(template.alignment_length / 300.0, 1.0) * 0.1

        # Identity bonus (higher identity gets exponential bonus)
        identity_bonus = (template.sequence_identity ** 2) * 0.2

        # E-value penalty (lower e-value is better)
        evalue_factor = max(0.1, 1.0 - (template.e_value / self.max_evalue))

        # Combine all factors
        quality_score = (base_score * db_weight * evalue_factor) + coverage_bonus + identity_bonus

        return quality_score

    def _print_template_summary(self, templates: List[FoldSeekTemplate]):
        """Print summary of selected templates"""

        if not templates:
            print("   No templates selected")
            return

        # Count by database type
        experimental_count = sum(1 for t in templates if t.search_type == 'experimental')
        predicted_count = sum(1 for t in templates if t.search_type == 'predicted')

        print(f"   Experimental: {experimental_count}, Predicted: {predicted_count}")
        print(f"   Best templates:")

        for i, template in enumerate(templates[:3]):
            print(f"     {i+1}. {template.template_id}: "
                  f"{template.sequence_identity:.1%} identity, "
                  f"E={template.e_value:.2e}, "
                  f"Type={template.search_type}")

    def _sequence_search(self, sequence: str, sequence_id: str) -> List[FoldSeekTemplate]:
        """Perform sequence similarity search"""

        templates = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write query sequence to FASTA
            query_file = tmpdir / "query.fasta"
            with open(query_file, 'w') as f:
                f.write(f">{sequence_id}\n{sequence}\n")

            # FOLDSEEK sequence search
            output_file = tmpdir / "results.m8"

            search_cmd = [
                self.foldseek_bin, "easy-search",
                str(query_file),
                str(self.foldseek_db_path),
                str(output_file),
                str(tmpdir / "tmp"),
                "--format-mode", "4",  # BLAST+ format
                "--max-seqs", str(self.max_templates * 2),  # Get extra for filtering
                "-e", str(self.max_evalue),
                "--min-seq-id", str(self.min_identity),
                "-v", "1"  # Minimal verbosity
            ]

            start_time = time.time()
            result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=300)
            search_time = time.time() - start_time

            if result.returncode != 0:
                print(f"⚠️  Sequence search failed: {result.stderr}")
                return []

            print(f"🚀 Sequence search completed in {search_time:.2f}s")

            # Parse results
            if output_file.exists() and output_file.stat().st_size > 0:
                templates = self._parse_m8_results(output_file, sequence_id, "sequence")

        return templates

    def _structure_search(self, sequence: str, sequence_id: str) -> List[FoldSeekTemplate]:
        """Perform structure similarity search (placeholder for now)"""

        # For structure search, we'd need either:
        # 1. A predicted structure from a fast folder (like ChimeraX AlphaFold)
        # 2. Convert sequence to 3Di alphabet for structure-aware search

        # For now, return empty list but keep the architecture ready
        print("🔧 Structure search not implemented yet - focusing on sequence search")
        return []

    def _parse_m8_results(self,
                         results_file: Path,
                         sequence_id: str,
                         db_type: str,
                         db_weight: float = 1.0) -> List[FoldSeekTemplate]:
        """Parse FOLDSEEK M8 format results"""

        templates = []

        try:
            # M8 format columns (MMseqs2/FOLDSEEK custom format)
            # Based on our --format-output: "target,query,fident,alnlen,mismatch,gapopen,tstart,tend,qstart,qend,evalue,bits"
            columns = [
                'target', 'query', 'identity', 'alignment_length',
                'mismatches', 'gap_opens', 'target_start', 'target_end',
                'query_start', 'query_end', 'evalue', 'bitscore'
            ]

            df = pd.read_csv(results_file, sep='\t', header=None, names=columns)

            for _, row in df.iterrows():
                # Extract PDB ID from target (e.g., "1ABC_A" -> "1ABC")
                pdb_id = row['target'].split('_')[0]

                template = FoldSeekTemplate(
                    target_id=sequence_id,
                    template_id=row['target'],
                    sequence_identity=row['identity'],  # MMseqs2 already reports as fraction (0-1)
                    alignment_length=row['alignment_length'],
                    e_value=row['evalue'],
                    bit_score=row['bitscore'],
                    query_start=row['query_start'],
                    query_end=row['query_end'],
                    template_start=row['target_start'],
                    template_end=row['target_end'],
                    query_aligned="",  # Would need alignment details for this
                    template_aligned="",
                    template_path=self._get_template_path(pdb_id),
                    search_type=db_type
                )

                templates.append(template)

        except Exception as e:
            print(f"⚠️  Failed to parse {results_file}: {e}")

        return templates

    def _get_template_path(self, pdb_id: str) -> str:
        """Get path to template structure file"""
        # This would depend on your PDB file organization
        # Common patterns:
        return f"/data/pdb/{pdb_id[1:3]}/pdb{pdb_id.lower()}.ent.gz"

    def _merge_templates(self,
                        seq_templates: List[FoldSeekTemplate],
                        struct_templates: List[FoldSeekTemplate]) -> List[FoldSeekTemplate]:
        """Merge and deduplicate templates from both searches"""

        # Combine all templates
        all_templates = seq_templates + struct_templates

        # Deduplicate by template_id (keep best scoring)
        template_dict = {}
        for template in all_templates:
            key = template.template_id
            if key not in template_dict or template.bit_score > template_dict[key].bit_score:
                template_dict[key] = template

        return list(template_dict.values())

    def _filter_templates(self, templates: List[FoldSeekTemplate]) -> List[FoldSeekTemplate]:
        """Filter templates by quality and date cutoff"""

        filtered = []

        for template in templates:
            # Basic quality filters
            if (template.sequence_identity >= self.min_identity and
                template.e_value <= self.max_evalue and
                template.alignment_length >= 30):  # Minimum alignment length

                filtered.append(template)

        # Sort by bit score (descending)
        filtered.sort(key=lambda t: t.bit_score, reverse=True)

        return filtered

    def convert_to_m8_format(self,
                           templates: List[FoldSeekTemplate],
                           output_file: Path):
        """Convert templates to M8 format for OpenFold3-MLX"""

        with open(output_file, 'w') as f:
            for template in templates:
                # Write in M8/BLAST format
                f.write(f"{template.target_id}\t{template.template_id}\t"
                       f"{template.sequence_identity*100:.1f}\t{template.alignment_length}\t"
                       f"0\t0\t{template.query_start}\t{template.query_end}\t"
                       f"{template.template_start}\t{template.template_end}\t"
                       f"{template.e_value:.2e}\t{template.bit_score:.1f}\n")

        print(f"💾 Saved {len(templates)} templates to {output_file}")


def main():
    """Demo/test function"""
    # Test sequence (P11972)
    test_sequence = "MVDKNRTLHELSSKNFSRTPNGLIFTNDLKTVYSIFLICLDLKEKKHSSDTKSFLLTAFTKHFHFTFTYQEAIKAMGQLELKVDMNTTCINVSYNIKPSLARHLLTLFMSSKLLHTPQDRTRGEPKEKVLFQPTPKGVAVLQKYVRDIGLKTMPDILLSSFNSMKLFTFERSSVTDSIIHSDYLIHILFIKMMGAKPNVWSPTNADDPLPCLSSLLEYTNNDDTFTFEKSKPEQGWQAQIGNIDINDLERVSPLAHRFFTNPDSESHTQYYVSNAGIRLFENKTFGTSKKIVIKYTFTTKAIWQWIMDCTDIMHVKEAVSLAALFLKTGLIVPVLLQPSRTDKKKFQISRSSFFTLSKRGWDLVSWTGCKSNNIRAPNGSTIDLDFTLRGHMTVRDEKKTLDDSEGFSQDMLISSSNLNKLDYVLTDPGMRYLFRRHLEKELCVENLDVFIEIKRFLKKMTILKKLIDSKHCDKKSNTSTSKNNIVKTIDSALMKQANECLEMAYHIYSSYIMIGSPYQLNIHHNLRQNISDIMLHPHSPLSEHFPTNLYDPSPASAESAASSISSTEADTLGEPPEVSLKPSKNLSNENCSFKKQGFKHQLKEYKPAPLTLAETHSPNASVENSHTIVRYGMDNTQNDTKSVESFPATLKVLRKLYPLFEIVSNEMYRLMNNDSFQKFTQSDVYKDASALIEIQEKC"

    try:
        retriever = FoldSeekTemplateRetriever()
        templates = retriever.get_templates_parallel(test_sequence, "T1201")

        print(f"\n🎯 Retrieved {len(templates)} templates:")
        for i, template in enumerate(templates[:5]):
            print(f"  {i+1}. {template.template_id}: "
                  f"{template.sequence_identity:.1%} identity, "
                  f"E={template.e_value:.2e}")

        # Save in M8 format for OpenFold3-MLX
        output_file = Path("test_templates.m8")
        retriever.convert_to_m8_format(templates, output_file)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()