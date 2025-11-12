#!/usr/bin/env python3
"""
Enhanced Benchmark Runner with FOLDSEEK Template Integration

This enhanced benchmarker integrates FOLDSEEK template retrieval with OpenFold3-MLX
for potentially superior performance vs Google AF3's 2021 template cutoff.

Key enhancements:
1. Parallel MSA + FOLDSEEK template retrieval
2. Automatic integration with OpenFold3-MLX inference
3. Comparison benchmarks: ColabFold vs FOLDSEEK vs AF3
4. Advanced performance and accuracy metrics

Usage:
    python benchmark_runner_with_foldseek.py
"""

import argparse
import json
import subprocess
import time
import pandas as pd
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import sys

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from parallel_msa_template_pipeline import ParallelMSATemplatePipeline, PipelineResult
from benchmark_runner import (
    FASTAParser, SystemMonitor, AccuracyCalculator,
    BenchmarkResult
)


@dataclass
class FoldSeekBenchmarkResult(BenchmarkResult):
    """Extended benchmark result with FOLDSEEK template info"""

    # Template information (all with defaults to fix dataclass inheritance)
    template_source: str = "none"            # 'foldseek', 'colabfold', or 'none'
    template_count: int = 0                  # Number of templates used
    template_retrieval_time: float = 0.0    # Time to get templates
    msa_depth: int = 0                       # MSA depth
    msa_retrieval_time: float = 0.0         # Time to get MSA

    # Parallel pipeline timing
    parallel_retrieval_time: float = 0.0    # Total parallel MSA+template time
    preprocessing_time: float = 0.0         # OpenFold3-MLX preprocessing time
    inference_time: float = 0.0             # Pure inference time

    # Resource usage details
    resource_usage: Dict = None             # Detailed resource stats


class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with FOLDSEEK integration"""

    def __init__(self,
                 output_dir: Path = Path("benchmarks/enhanced_benchmark_results"),
                 use_foldseek: bool = True,
                 max_concurrent_jobs: int = 2,
                 compare_template_sources: bool = True):

        # Ensure all paths are relative to project root
        if not output_dir.is_absolute():
            self.output_dir = Path.cwd() / output_dir
        else:
            self.output_dir = output_dir
        self.use_foldseek = use_foldseek
        self.max_concurrent_jobs = max_concurrent_jobs
        self.compare_template_sources = compare_template_sources

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_output_dir = self.output_dir / "pipeline_data"
        self.inference_output_dir = self.output_dir / "inference_results"

        for dir_path in [self.pipeline_output_dir, self.inference_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.fasta_parser = FASTAParser()
        self.system_monitor = SystemMonitor()
        self.accuracy_calculator = AccuracyCalculator()

        # Initialize parallel pipeline (paths relative to current working directory)
        pipeline_dir = self.pipeline_output_dir
        if not pipeline_dir.is_absolute():
            pipeline_dir = Path.cwd() / pipeline_dir

        self.msa_template_pipeline = ParallelMSATemplatePipeline(
            output_base_dir=pipeline_dir,
            use_foldseek=use_foldseek
        )

        print(f"🚀 Enhanced Benchmark Runner initialized")
        print(f"   Template source: {'FOLDSEEK' if use_foldseek else 'ColabFold'}")
        print(f"   Comparison mode: {'ON' if compare_template_sources else 'OFF'}")
        print(f"   Output directory: {output_dir}")

    def run_single_benchmark(self,
                           sequence: str,
                           sequence_id: str,
                           template_source: str = "auto") -> FoldSeekBenchmarkResult:
        """Run benchmark for single sequence with specified template source"""

        print(f"\n{'='*60}")
        print(f"🧬 Benchmarking {sequence_id} ({len(sequence)} residues)")
        print(f"   Template source: {template_source}")
        print(f"{'='*60}")

        start_time = time.time()

        # Step 1: Parallel MSA + Template Retrieval
        print(f"\n📡 Step 1: Retrieving MSA + Templates")

        # Configure pipeline for template source
        if template_source == "foldseek":
            self.msa_template_pipeline.use_foldseek = True
        elif template_source == "colabfold":
            self.msa_template_pipeline.use_foldseek = False
            self.msa_template_pipeline.use_colabfold_templates = True
        elif template_source == "none":
            self.msa_template_pipeline.use_foldseek = False
            self.msa_template_pipeline.use_colabfold_templates = False

        pipeline_result = self.msa_template_pipeline.process_sequence(sequence, sequence_id)

        # Step 2: Create OpenFold3-MLX query configuration
        print(f"\n⚙️  Step 2: Configuring OpenFold3-MLX")
        query_config = self._create_query_config(sequence_id, pipeline_result)

        # Step 3: Run OpenFold3-MLX inference
        print(f"\n🔮 Step 3: Running OpenFold3-MLX inference")
        inference_start = time.time()

        # Start monitoring system resources
        self.system_monitor.start_monitoring()
        try:
            inference_result = self._run_openfold3_inference(query_config, sequence_id)
        finally:
            # Stop monitoring regardless of success/failure
            resource_stats = self.system_monitor.stop_monitoring()

        inference_time = time.time() - inference_start

        # Step 4: Calculate accuracy metrics
        print(f"\n📊 Step 4: Calculating accuracy metrics")
        accuracy_result = self._calculate_accuracy_metrics(sequence_id, inference_result)

        # Compile final result
        total_time = time.time() - start_time

        result = FoldSeekBenchmarkResult(
            # Base benchmark result fields (required by parent class)
            sequence_id=sequence_id,
            config_name=f"foldseek_{template_source}",
            success=inference_result.get('success', False),
            wall_time_seconds=total_time,
            peak_memory_gb=resource_stats.get('peak_memory_gb', 0.0),
            avg_cpu_percent=resource_stats.get('avg_cpu_percent', 0.0),
            avg_gpu_percent=resource_stats.get('avg_gpu_percent', None),
            error_message=inference_result.get('error') if not inference_result.get('success', False) else None,
            mean_plddt=accuracy_result.get('mean_plddt', 0.0),

            # Enhanced fields
            template_source=template_source,
            template_count=pipeline_result.template_result.template_count,
            template_retrieval_time=pipeline_result.template_result.retrieval_time,
            msa_depth=pipeline_result.msa_result.msa_depth,
            msa_retrieval_time=pipeline_result.msa_result.retrieval_time,
            parallel_retrieval_time=pipeline_result.total_time,
            preprocessing_time=inference_result.get('preprocessing_time', 0.0),
            inference_time=inference_time,

            # Resource usage
            resource_usage=resource_stats
        )

        print(f"\n✅ Benchmark completed for {sequence_id}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   MSA depth: {pipeline_result.msa_result.msa_depth}")
        print(f"   Templates: {pipeline_result.template_result.template_count}")
        print(f"   Mean pLDDT: {result.mean_plddt:.1f}")

        return result

    def _create_query_config(self,
                           sequence_id: str,
                           pipeline_result: PipelineResult) -> Dict:
        """Create OpenFold3-MLX query configuration with custom templates"""

        # Build chain configuration
        chain_config = {
            "molecule_type": "protein",
            "chain_ids": ["A"],
            "sequence": "",  # Will be read from MSA file
        }

        # Add MSA if available
        if pipeline_result.msa_result.success and pipeline_result.msa_result.msa_file.exists():
            chain_config["main_msa_file_paths"] = [
                str(pipeline_result.msa_result.msa_file)
            ]

        # Add templates if available
        if (pipeline_result.template_result.success and
            pipeline_result.template_result.m8_file.exists()):

            chain_config["template_alignment_file_path"] = str(
                pipeline_result.template_result.m8_file
            )

            print(f"✅ Template configuration added: {pipeline_result.template_result.template_count} templates")

        # Create query in correct format expected by OpenFold3
        config = {
            "queries": {
                sequence_id: {
                    "chains": [chain_config]
                }
            }
        }

        return config

    def _run_openfold3_inference(self,
                               query_config: Dict,
                               sequence_id: str) -> Dict:
        """Run OpenFold3-MLX inference with custom configuration"""

        # Create query JSON file
        query_file = self.inference_output_dir / f"{sequence_id}_query.json"
        with open(query_file, 'w') as f:
            json.dump(query_config, f, indent=2)

        # Create output directory for this inference
        output_dir = self.inference_output_dir / sequence_id
        output_dir.mkdir(parents=True, exist_ok=True)

        preprocessing_start = time.time()

        # Run OpenFold3-MLX prediction (paths relative to project root since cwd=project_root)
        cmd = [
            "python", "openfold3/run_openfold.py", "predict",
            "--query_json", str(query_file),
            "--output_dir", str(output_dir),
            "--use_msa_server", "True",
            "--use_templates", "False",
        ]

        print(f"🔮 Running: {' '.join(cmd)}")

        try:
            # Determine project root directory
            project_root = Path.cwd()
            if project_root.name == "benchmarks":
                project_root = project_root.parent

            result = subprocess.run(
                cmd,
                cwd=project_root,  # Always run from project root
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            preprocessing_time = time.time() - preprocessing_start

            if result.returncode == 0:
                print(f"✅ Inference completed successfully")
                return {
                    "success": True,
                    "output_dir": output_dir,
                    "preprocessing_time": preprocessing_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"❌ Inference failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "preprocessing_time": preprocessing_time
                }

        except subprocess.TimeoutExpired:
            print(f"⏰ Inference timed out after 30 minutes")
            return {
                "success": False,
                "error": "Timeout after 30 minutes",
                "preprocessing_time": preprocessing_time
            }
        except Exception as e:
            print(f"❌ Inference error: {e}")
            return {
                "success": False,
                "error": str(e),
                "preprocessing_time": preprocessing_time
            }

    def _calculate_accuracy_metrics(self,
                                  sequence_id: str,
                                  inference_result: Dict) -> Dict:
        """Calculate accuracy metrics from inference results"""

        if not inference_result.get("success", False):
            return {"mean_plddt": 0.0, "confidence_files": 0}

        output_dir = inference_result["output_dir"]

        # Use existing accuracy calculator
        mean_plddt, conf_files, plddt_list = self.accuracy_calculator.calculate_plddt_from_output(
            output_dir, sequence_id
        )

        return {
            "mean_plddt": mean_plddt or 0.0,
            "confidence_files": conf_files,
            "plddt_distribution": plddt_list
        }

    def run_comparison_benchmark(self,
                               sequences: Dict[str, str]) -> Dict[str, List[FoldSeekBenchmarkResult]]:
        """Run comparison benchmark across different template sources"""

        if not self.compare_template_sources:
            print("⚠️  Comparison mode disabled - running with default configuration only")
            return self.run_standard_benchmark(sequences)

        template_sources = ["foldseek", "colabfold", "none"]
        all_results = {}

        print(f"\n🏁 Running comparison benchmark")
        print(f"   Sequences: {len(sequences)}")
        print(f"   Template sources: {template_sources}")
        print(f"   Total jobs: {len(sequences) * len(template_sources)}")

        for template_source in template_sources:
            print(f"\n{'='*80}")
            print(f"📊 TEMPLATE SOURCE: {template_source.upper()}")
            print(f"{'='*80}")

            source_results = []

            for seq_id, sequence in sequences.items():
                try:
                    result = self.run_single_benchmark(sequence, seq_id, template_source)
                    source_results.append(result)

                    # Save individual result
                    result_file = self.output_dir / f"{seq_id}_{template_source}_result.json"
                    with open(result_file, 'w') as f:
                        json.dump(asdict(result), f, indent=2)

                except Exception as e:
                    print(f"❌ Failed {seq_id} with {template_source}: {e}")

            all_results[template_source] = source_results

        # Generate comparison report
        self._generate_comparison_report(all_results)

        return all_results

    def run_standard_benchmark(self,
                             sequences: Dict[str, str]) -> Dict[str, List[FoldSeekBenchmarkResult]]:
        """Run standard benchmark with default configuration"""

        template_source = "foldseek" if self.use_foldseek else "colabfold"
        results = []

        print(f"\n🚀 Running standard benchmark with {template_source} templates")

        for seq_id, sequence in sequences.items():
            try:
                result = self.run_single_benchmark(sequence, seq_id, template_source)
                results.append(result)

                # Save result
                result_file = self.output_dir / f"{seq_id}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(asdict(result), f, indent=2)

            except Exception as e:
                print(f"❌ Failed {seq_id}: {e}")

        return {template_source: results}

    def _generate_comparison_report(self, all_results: Dict[str, List[FoldSeekBenchmarkResult]]):
        """Generate comprehensive comparison report"""

        print(f"\n{'='*80}")
        print(f"📊 COMPARISON REPORT: FOLDSEEK vs COLABFOLD vs NO TEMPLATES")
        print(f"{'='*80}")

        comparison_data = []

        for template_source, results in all_results.items():
            if not results:
                continue

            # Calculate summary statistics
            mean_plddt = sum(r.mean_plddt for r in results) / len(results)
            mean_time = sum(r.wall_time_seconds for r in results) / len(results)
            mean_templates = sum(r.template_count for r in results) / len(results)
            mean_msa_depth = sum(r.msa_depth for r in results) / len(results)

            comparison_data.append({
                'template_source': template_source,
                'sequences': len(results),
                'mean_plddt': mean_plddt,
                'mean_total_time': mean_time,
                'mean_templates': mean_templates,
                'mean_msa_depth': mean_msa_depth
            })

            print(f"\n🎯 {template_source.upper()} Results:")
            print(f"   Sequences: {len(results)}")
            print(f"   Mean pLDDT: {mean_plddt:.1f}")
            print(f"   Mean time: {mean_time:.1f}s")
            print(f"   Mean templates: {mean_templates:.1f}")
            print(f"   Mean MSA depth: {mean_msa_depth:.0f}")

        # Save comparison data
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = self.output_dir / "template_source_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)

        print(f"\n💾 Comparison data saved to {comparison_file}")

        # Determine winner
        if len(comparison_data) > 1:
            best_accuracy = max(comparison_data, key=lambda x: x['mean_plddt'])
            print(f"\n🏆 Best Accuracy: {best_accuracy['template_source'].upper()} "
                  f"(pLDDT: {best_accuracy['mean_plddt']:.1f})")


def main():
    """Main benchmark execution"""

    parser = argparse.ArgumentParser(description="Enhanced OpenFold3-MLX Benchmark with FOLDSEEK")
    parser.add_argument("--fasta", type=Path, default="benchmarks/casp16_monomers.fasta",
                       help="FASTA file with sequences to benchmark")
    parser.add_argument("--output", type=Path, default="benchmarks/enhanced_benchmark_results",
                       help="Output directory for benchmark results")
    parser.add_argument("--use-foldseek", action="store_true", default=True,
                       help="Use FOLDSEEK for template retrieval")
    parser.add_argument("--compare-sources", action="store_true", default=False,
                       help="Compare different template sources")
    parser.add_argument("--max-sequences", type=int, default=5,
                       help="Maximum sequences to benchmark (for testing)")
    parser.add_argument("--max-concurrent", type=int, default=2,
                       help="Maximum concurrent jobs")

    args = parser.parse_args()

    # Load sequences
    fasta_parser = FASTAParser()
    sequences = fasta_parser.parse_fasta(args.fasta)

    # Limit sequences for testing
    if args.max_sequences:
        sequences = dict(list(sequences.items())[:args.max_sequences])

    print(f"🧬 Loaded {len(sequences)} sequences from {args.fasta}")

    # Create benchmark runner
    runner = EnhancedBenchmarkRunner(
        output_dir=args.output,
        use_foldseek=args.use_foldseek,
        max_concurrent_jobs=args.max_concurrent,
        compare_template_sources=args.compare_sources
    )

    # Run benchmark
    if args.compare_sources:
        results = runner.run_comparison_benchmark(sequences)
    else:
        results = runner.run_standard_benchmark(sequences)

    print(f"\n✅ Enhanced benchmark completed!")
    print(f"   Results saved to: {args.output}")


if __name__ == "__main__":
    main()