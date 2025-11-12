#!/usr/bin/env python3
"""
FOLDSEEK vs ColabFold Speed & Quality Benchmark

This script runs head-to-head comparisons between FOLDSEEK and ColabFold
template retrieval systems with OpenFold3-MLX inference to measure:

1. Template retrieval speed
2. Final model quality (pLDDT scores)
3. Template coverage and relevance
4. Overall performance metrics

Usage:
    python foldseek_vs_colabfold_benchmark.py --sequences 3 --output results/
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Tuple
import time
import sys

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from benchmark_runner_with_foldseek import EnhancedBenchmarkRunner, FoldSeekBenchmarkResult
from benchmark_runner import FASTAParser


class FoldSeekVsColabFoldComparator:
    """Comprehensive speed and quality comparison between FOLDSEEK and ColabFold"""

    def __init__(self, output_dir: Path = Path("benchmarks/comparison_results")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize benchmark runners for both systems
        self.foldseek_runner = EnhancedBenchmarkRunner(
            output_dir=self.output_dir / "foldseek_results",
            use_foldseek=True
        )

        self.colabfold_runner = EnhancedBenchmarkRunner(
            output_dir=self.output_dir / "colabfold_results",
            use_foldseek=False
        )

        # Results storage
        self.results = {
            'foldseek': [],
            'colabfold': []
        }

        print(f"🥊 FOLDSEEK vs ColabFold Benchmark initialized")
        print(f"   Output directory: {self.output_dir}")

    def run_comparison(self, sequences: Dict[str, str], max_sequences: int = None) -> Dict[str, List[FoldSeekBenchmarkResult]]:
        """Run complete comparison between FOLDSEEK and ColabFold"""

        if max_sequences:
            sequences = dict(list(sequences.items())[:max_sequences])

        # Store sequence information for later analysis
        self.sequence_data = sequences

        print(f"\n🚀 Starting comparison with {len(sequences)} sequences")
        print(f"   Template sources: FOLDSEEK vs ColabFold")
        print(f"   Metrics: Speed, Quality (pLDDT), Template Coverage")

        for i, (seq_id, sequence) in enumerate(sequences.items(), 1):
            print(f"\n{'='*80}")
            print(f"🧬 Sequence {i}/{len(sequences)}: {seq_id} ({len(sequence)} residues)")
            print(f"{'='*80}")

            # Run FOLDSEEK benchmark
            print(f"\n🔍 Running FOLDSEEK benchmark...")
            try:
                foldseek_start = time.time()
                foldseek_result = self.foldseek_runner.run_single_benchmark(
                    sequence, seq_id, template_source="foldseek"
                )
                foldseek_result.total_benchmark_time = time.time() - foldseek_start
                self.results['foldseek'].append(foldseek_result)

                print(f"✅ FOLDSEEK completed in {foldseek_result.total_benchmark_time:.1f}s")
                print(f"   Templates: {foldseek_result.template_count}")
                print(f"   Template time: {foldseek_result.template_retrieval_time:.1f}s")

            except Exception as e:
                print(f"❌ FOLDSEEK benchmark failed: {e}")
                continue

            # Run ColabFold benchmark
            print(f"\n🌐 Running ColabFold benchmark...")
            try:
                colabfold_start = time.time()
                colabfold_result = self.colabfold_runner.run_single_benchmark(
                    sequence, seq_id, template_source="colabfold"
                )
                colabfold_result.total_benchmark_time = time.time() - colabfold_start
                self.results['colabfold'].append(colabfold_result)

                print(f"✅ ColabFold completed in {colabfold_result.total_benchmark_time:.1f}s")
                print(f"   Templates: {colabfold_result.template_count}")
                print(f"   Template time: {colabfold_result.template_retrieval_time:.1f}s")

            except Exception as e:
                print(f"❌ ColabFold benchmark failed: {e}")
                continue

            # Quick comparison for this sequence
            if len(self.results['foldseek']) > 0 and len(self.results['colabfold']) > 0:
                fs_result = self.results['foldseek'][-1]
                cf_result = self.results['colabfold'][-1]

                speedup = cf_result.template_retrieval_time / max(fs_result.template_retrieval_time, 0.1)

                print(f"\n📊 Quick comparison for {seq_id}:")
                print(f"   Speed: FOLDSEEK {speedup:.1f}x faster than ColabFold")
                print(f"   Templates: FOLDSEEK {fs_result.template_count} vs ColabFold {cf_result.template_count}")

                if fs_result.mean_plddt and cf_result.mean_plddt:
                    quality_diff = fs_result.mean_plddt - cf_result.mean_plddt
                    winner = "FOLDSEEK" if quality_diff > 0 else "ColabFold"
                    print(f"   Quality: {winner} +{abs(quality_diff):.1f} pLDDT points")

        # Save results
        self._save_results()

        return self.results

    def generate_comparison_report(self) -> Path:
        """Generate comprehensive comparison report with visualizations"""

        print(f"\n📊 Generating comparison report...")

        # Convert results to DataFrame for analysis
        df_results = []

        for system, results in self.results.items():
            for result in results:
                df_results.append({
                    'system': system,
                    'sequence_id': result.sequence_id,
                    'sequence_length': len(self.sequence_data.get(result.sequence_id, "")),
                    'template_count': result.template_count,
                    'template_retrieval_time': result.template_retrieval_time,
                    'msa_depth': result.msa_depth,
                    'msa_retrieval_time': result.msa_retrieval_time,
                    'inference_time': result.inference_time,
                    'total_time': result.total_benchmark_time,
                    'mean_plddt': result.mean_plddt,
                    'peak_memory_gb': result.peak_memory_gb,
                    'success': result.inference_success
                })

        df = pd.DataFrame(df_results)

        if df.empty:
            print("❌ No results to analyze")
            return None

        # Generate analysis report
        report_file = self.output_dir / "comparison_report.md"

        with open(report_file, 'w') as f:
            f.write("# FOLDSEEK vs ColabFold Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            if len(df) > 0:
                foldseek_data = df[df['system'] == 'foldseek']
                colabfold_data = df[df['system'] == 'colabfold']

                if len(foldseek_data) > 0 and len(colabfold_data) > 0:
                    # Speed comparison
                    avg_fs_template_time = foldseek_data['template_retrieval_time'].mean()
                    avg_cf_template_time = colabfold_data['template_retrieval_time'].mean()
                    speedup = avg_cf_template_time / max(avg_fs_template_time, 0.1)

                    f.write(f"### Speed Performance\n")
                    f.write(f"- **FOLDSEEK**: {avg_fs_template_time:.1f}s average template retrieval\n")
                    f.write(f"- **ColabFold**: {avg_cf_template_time:.1f}s average template retrieval\n")
                    f.write(f"- **Speedup**: {speedup:.1f}x faster with FOLDSEEK\n\n")

                    # Quality comparison
                    avg_fs_plddt = foldseek_data['mean_plddt'].mean()
                    avg_cf_plddt = colabfold_data['mean_plddt'].mean()
                    quality_diff = avg_fs_plddt - avg_cf_plddt

                    f.write(f"### Quality Performance\n")
                    f.write(f"- **FOLDSEEK**: {avg_fs_plddt:.1f} average pLDDT\n")
                    f.write(f"- **ColabFold**: {avg_cf_plddt:.1f} average pLDDT\n")
                    f.write(f"- **Difference**: {quality_diff:+.1f} pLDDT points\n\n")

                    # Template coverage
                    avg_fs_templates = foldseek_data['template_count'].mean()
                    avg_cf_templates = colabfold_data['template_count'].mean()

                    f.write(f"### Template Coverage\n")
                    f.write(f"- **FOLDSEEK**: {avg_fs_templates:.1f} average templates\n")
                    f.write(f"- **ColabFold**: {avg_cf_templates:.1f} average templates\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

            # Methodology
            f.write("## Methodology\n\n")
            f.write("This benchmark compared FOLDSEEK vs ColabFold template retrieval using:\n")
            f.write("- Same CASP16 monomer sequences\n")
            f.write("- OpenFold3-MLX for inference\n")
            f.write("- Local FOLDSEEK databases (PDB100 + AFDB Swiss-Prot)\n")
            f.write("- Standard evaluation metrics (pLDDT, timing)\n\n")

        # Generate visualizations
        self._create_visualizations(df)

        print(f"✅ Report generated: {report_file}")
        return report_file

    def _create_visualizations(self, df: pd.DataFrame):
        """Create comparison visualizations"""

        if df.empty or len(df) < 2:
            return

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FOLDSEEK vs ColabFold Benchmark Comparison', fontsize=16)

        # 1. Template retrieval time comparison
        if 'template_retrieval_time' in df.columns:
            sns.boxplot(data=df, x='system', y='template_retrieval_time', ax=axes[0,0])
            axes[0,0].set_title('Template Retrieval Time')
            axes[0,0].set_ylabel('Time (seconds)')

        # 2. Quality (pLDDT) comparison
        if 'mean_plddt' in df.columns:
            sns.boxplot(data=df, x='system', y='mean_plddt', ax=axes[0,1])
            axes[0,1].set_title('Model Quality (pLDDT)')
            axes[0,1].set_ylabel('Mean pLDDT')

        # 3. Template count comparison
        if 'template_count' in df.columns:
            sns.boxplot(data=df, x='system', y='template_count', ax=axes[1,0])
            axes[1,0].set_title('Template Coverage')
            axes[1,0].set_ylabel('Number of Templates')

        # 4. Total time comparison
        if 'total_time' in df.columns:
            sns.boxplot(data=df, x='system', y='total_time', ax=axes[1,1])
            axes[1,1].set_title('Total Benchmark Time')
            axes[1,1].set_ylabel('Time (seconds)')

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualizations saved: {plot_file}")

    def _save_results(self):
        """Save raw results to JSON"""

        results_file = self.output_dir / "raw_results.json"

        # Convert results to JSON-serializable format
        json_results = {}
        for system, results in self.results.items():
            json_results[system] = [asdict(result) for result in results]

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"💾 Results saved: {results_file}")


def main():
    """Main execution"""

    parser = argparse.ArgumentParser(description="FOLDSEEK vs ColabFold Speed & Quality Benchmark")
    parser.add_argument("--fasta", type=Path, default="benchmarks/casp16_monomers.fasta",
                       help="FASTA file with sequences to benchmark")
    parser.add_argument("--output", type=Path, default="benchmarks/foldseek_colabfold_comparison",
                       help="Output directory for comparison results")
    parser.add_argument("--sequences", type=int, default=3,
                       help="Maximum sequences to benchmark (for speed)")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Skip generating visualization plots")

    args = parser.parse_args()

    # Load sequences
    print(f"📂 Loading sequences from {args.fasta}")
    fasta_parser = FASTAParser()
    sequence_infos = fasta_parser.parse_fasta(args.fasta)

    # Convert to dictionary format expected by comparator
    sequences = {seq_info.id: seq_info.sequence for seq_info in sequence_infos}
    print(f"✅ Loaded {len(sequences)} sequences")

    # Initialize comparator
    comparator = FoldSeekVsColabFoldComparator(output_dir=args.output)

    # Run comparison
    try:
        results = comparator.run_comparison(sequences, max_sequences=args.sequences)

        # Generate report
        report_file = comparator.generate_comparison_report()

        print(f"\n🎉 Benchmark comparison completed!")
        print(f"   Results: {len(results['foldseek'])} FOLDSEEK vs {len(results['colabfold'])} ColabFold")
        print(f"   Report: {report_file}")
        print(f"   Output: {args.output}")

    except KeyboardInterrupt:
        print(f"\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()