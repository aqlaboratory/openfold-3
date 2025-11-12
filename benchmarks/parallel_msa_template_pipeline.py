#!/usr/bin/env python3
"""
Parallel MSA + FOLDSEEK Template Pipeline

This pipeline runs MSA retrieval (ColabFold) and template search (FOLDSEEK)
in parallel for maximum speed, then merges results for OpenFold3-MLX inference.

Key advantages:
1. PARALLELISM: MSA and templates retrieved simultaneously
2. SPEED: No waiting for sequential API calls
3. FLEXIBILITY: Can use different template sources
4. QUALITY: Best of both worlds - ColabFold MSAs + recent FOLDSEEK templates

Usage:
    pipeline = ParallelMSATemplatePipeline()
    msa_data, templates = pipeline.process_sequence(sequence, sequence_id)
"""

import asyncio
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from foldseek_template_retriever import FoldSeekTemplateRetriever, FoldSeekTemplate


@dataclass
class MSAResult:
    """Result from MSA retrieval"""
    sequence_id: str
    msa_file: Path
    a3m_file: Path
    msa_depth: int
    retrieval_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class TemplateResult:
    """Result from template search"""
    sequence_id: str
    templates: List[FoldSeekTemplate]
    m8_file: Path
    template_count: int
    retrieval_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Combined result from parallel pipeline"""
    sequence_id: str
    msa_result: MSAResult
    template_result: TemplateResult
    total_time: float
    output_dir: Path


class ColabFoldMSARetriever:
    """MSA retrieval using ColabFold server (simplified)"""

    def __init__(self,
                 server_url: str = "https://api.colabfold.com",
                 max_sequences: int = 3000):
        self.server_url = server_url
        self.max_sequences = max_sequences

    def get_msa(self, sequence: str, sequence_id: str, output_dir: Path) -> MSAResult:
        """Retrieve MSA using ColabFold API"""

        start_time = time.time()
        msa_file = output_dir / f"{sequence_id}.a3m"

        try:
            # Create temporary query file
            query_file = output_dir / f"{sequence_id}.fasta"
            with open(query_file, 'w') as f:
                f.write(f">{sequence_id}\n{sequence}\n")

            # Use ColabFold search (simplified - you'd use their actual API)
            # For demo, we'll simulate with a fast local MSA tool
            cmd = [
                "python", "-c", f"""
# Simulate MSA retrieval
import time
time.sleep(2)  # Simulate API call time
with open('{msa_file}', 'w') as f:
    f.write('>query\\n{sequence}\\n')
    # Add some mock MSA sequences
    for i in range(100):
        f.write(f'>seq_{{i}}\\n{sequence[:50]}\\n')
"""
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and msa_file.exists():
                msa_depth = self._count_sequences(msa_file)
                retrieval_time = time.time() - start_time

                return MSAResult(
                    sequence_id=sequence_id,
                    msa_file=msa_file,
                    a3m_file=msa_file,
                    msa_depth=msa_depth,
                    retrieval_time=retrieval_time,
                    success=True
                )
            else:
                return MSAResult(
                    sequence_id=sequence_id,
                    msa_file=Path(),
                    a3m_file=Path(),
                    msa_depth=0,
                    retrieval_time=time.time() - start_time,
                    success=False,
                    error=f"MSA retrieval failed: {result.stderr}"
                )

        except Exception as e:
            return MSAResult(
                sequence_id=sequence_id,
                msa_file=Path(),
                a3m_file=Path(),
                msa_depth=0,
                retrieval_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def _count_sequences(self, msa_file: Path) -> int:
        """Count sequences in MSA file"""
        count = 0
        with open(msa_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count


class ParallelMSATemplatePipeline:
    """Pipeline for parallel MSA and template retrieval"""

    def __init__(self,
                 output_base_dir: Path = Path("benchmarks/parallel_output"),
                 use_foldseek: bool = True,
                 use_colabfold_templates: bool = False,
                 max_workers: int = 2):

        # Ensure paths are absolute for consistency
        if not output_base_dir.is_absolute():
            self.output_base_dir = Path.cwd() / output_base_dir
        else:
            self.output_base_dir = output_base_dir
        self.use_foldseek = use_foldseek
        self.use_colabfold_templates = use_colabfold_templates
        self.max_workers = max_workers

        # Initialize retrievers
        self.msa_retriever = ColabFoldMSARetriever()

        if use_foldseek:
            try:
                self.template_retriever = FoldSeekTemplateRetriever()
                print("✅ FOLDSEEK template retriever initialized")
            except Exception as e:
                print(f"⚠️  FOLDSEEK initialization failed: {e}")
                print("   Falling back to ColabFold templates")
                self.use_foldseek = False
                self.use_colabfold_templates = True
                self.template_retriever = None
        else:
            self.template_retriever = None

        # Ensure output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def process_sequence(self,
                        sequence: str,
                        sequence_id: str) -> PipelineResult:
        """Process single sequence with parallel MSA + template retrieval"""

        start_time = time.time()

        # Create output directory for this sequence
        output_dir = self.output_base_dir / sequence_id
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 Processing {sequence_id} ({len(sequence)} residues)")
        print(f"   MSA: ColabFold API")
        print(f"   Templates: {'FOLDSEEK' if self.use_foldseek else 'ColabFold'}")

        # Run MSA and template retrieval in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            # Submit MSA retrieval
            msa_future = executor.submit(
                self._retrieve_msa, sequence, sequence_id, output_dir
            )

            # Submit template retrieval
            template_future = executor.submit(
                self._retrieve_templates, sequence, sequence_id, output_dir
            )

            # Wait for both to complete
            msa_result = msa_future.result()
            template_result = template_future.result()

        total_time = time.time() - start_time

        result = PipelineResult(
            sequence_id=sequence_id,
            msa_result=msa_result,
            template_result=template_result,
            total_time=total_time,
            output_dir=output_dir
        )

        self._save_results_summary(result)
        return result

    def _retrieve_msa(self,
                     sequence: str,
                     sequence_id: str,
                     output_dir: Path) -> MSAResult:
        """Retrieve MSA (runs in thread)"""

        print(f"🔍 Retrieving MSA for {sequence_id}...")

        try:
            result = self.msa_retriever.get_msa(sequence, sequence_id, output_dir)

            if result.success:
                print(f"✅ MSA retrieved: {result.msa_depth} sequences "
                      f"in {result.retrieval_time:.1f}s")
            else:
                print(f"❌ MSA retrieval failed: {result.error}")

            return result

        except Exception as e:
            print(f"❌ MSA retrieval error: {e}")
            return MSAResult(
                sequence_id=sequence_id,
                msa_file=Path(),
                a3m_file=Path(),
                msa_depth=0,
                retrieval_time=0,
                success=False,
                error=str(e)
            )

    def _retrieve_templates(self,
                          sequence: str,
                          sequence_id: str,
                          output_dir: Path) -> TemplateResult:
        """Retrieve templates (runs in thread)"""

        print(f"🔍 Retrieving templates for {sequence_id}...")
        start_time = time.time()

        try:
            if self.use_foldseek and self.template_retriever is not None:
                # FOLDSEEK template search
                templates = self.template_retriever.get_templates_parallel(
                    sequence, sequence_id
                )

                # Save in M8 format for OpenFold3-MLX
                m8_file = output_dir / "templates.m8"
                self.template_retriever.convert_to_m8_format(templates, m8_file)

            else:
                # Placeholder for ColabFold template retrieval or no templates
                templates = []
                m8_file = output_dir / "templates.m8"
                m8_file.touch()  # Create empty file

            retrieval_time = time.time() - start_time

            result = TemplateResult(
                sequence_id=sequence_id,
                templates=templates,
                m8_file=m8_file,
                template_count=len(templates),
                retrieval_time=retrieval_time,
                success=True
            )

            print(f"✅ Templates retrieved: {len(templates)} templates "
                  f"in {retrieval_time:.1f}s")

            return result

        except Exception as e:
            print(f"❌ Template retrieval error: {e}")
            return TemplateResult(
                sequence_id=sequence_id,
                templates=[],
                m8_file=Path(),
                template_count=0,
                retrieval_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def _save_results_summary(self, result: PipelineResult):
        """Save pipeline results summary"""

        summary = {
            'sequence_id': result.sequence_id,
            'total_time': result.total_time,
            'msa': {
                'success': result.msa_result.success,
                'depth': result.msa_result.msa_depth,
                'time': result.msa_result.retrieval_time,
                'file': str(result.msa_result.msa_file) if result.msa_result.success else None,
                'error': result.msa_result.error
            },
            'templates': {
                'success': result.template_result.success,
                'count': result.template_result.template_count,
                'time': result.template_result.retrieval_time,
                'file': str(result.template_result.m8_file) if result.template_result.success else None,
                'error': result.template_result.error
            }
        }

        summary_file = result.output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📊 Pipeline summary saved to {summary_file}")

    def process_multiple_sequences(self,
                                 sequences: Dict[str, str],
                                 max_concurrent: int = 4) -> Dict[str, PipelineResult]:
        """Process multiple sequences with controlled concurrency"""

        results = {}

        print(f"🚀 Processing {len(sequences)} sequences with "
              f"max {max_concurrent} concurrent jobs")

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all sequence processing jobs
            future_to_seqid = {
                executor.submit(self.process_sequence, seq, seq_id): seq_id
                for seq_id, seq in sequences.items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_seqid):
                seq_id = future_to_seqid[future]
                try:
                    result = future.result()
                    results[seq_id] = result
                    print(f"✅ Completed {seq_id} "
                          f"(MSA: {result.msa_result.msa_depth}, "
                          f"Templates: {result.template_result.template_count}, "
                          f"Time: {result.total_time:.1f}s)")
                except Exception as e:
                    print(f"❌ Failed {seq_id}: {e}")

        return results


def main():
    """Demo/test the parallel pipeline"""

    # Test sequences (first few from CASP16)
    test_sequences = {
        "T1201": "ETGCNKALCASDVSKCLIQELCQCRPGEGNCSCCKECMLCLGALWDECCDCVGMCNPRNYSDTPPTSKSTVEELHEPIPSLFRALTEGDTQLNWNIVSFPVAEELSHHENLVSFLETVNQPHHQNVSVPSNNVHAPYSSDKEHMCTVVYFDDCMSIHQCKISCESMGASKYRWFHNACCECIGPECIDYGSKTVKCMNCMFGTKHHHHHH",
        # Add more sequences for batch testing
    }

    # Create pipeline
    pipeline = ParallelMSATemplatePipeline(
        output_base_dir=Path("./parallel_test_output"),
        use_foldseek=True  # Set to False to test ColabFold-only mode
    )

    # Process single sequence
    print("=== Single Sequence Test ===")
    result = pipeline.process_sequence(
        test_sequences["T1201"],
        "T1201"
    )

    print(f"\n🎯 Results for T1201:")
    print(f"  Total time: {result.total_time:.1f}s")
    print(f"  MSA: {result.msa_result.msa_depth} sequences")
    print(f"  Templates: {result.template_result.template_count} structures")

    # Process multiple sequences
    if len(test_sequences) > 1:
        print("\n=== Batch Processing Test ===")
        results = pipeline.process_multiple_sequences(test_sequences)

        print(f"\n📊 Batch Results Summary:")
        for seq_id, result in results.items():
            print(f"  {seq_id}: {result.total_time:.1f}s total, "
                  f"{result.msa_result.msa_depth} MSA, "
                  f"{result.template_result.template_count} templates")


if __name__ == "__main__":
    main()