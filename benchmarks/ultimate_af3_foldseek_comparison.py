#!/usr/bin/env python3
"""
Ultimate OpenFold3-MLX vs AlphaFold3 vs FOLDSEEK Template Comparison

This script orchestrates the complete comparison workflow:
1. FOLDSEEK templates + OpenFold3-MLX
2. ColabFold templates + OpenFold3-MLX
3. Google AlphaFold3 Server results
4. No-template baseline (for reference)

The ultimate test: Can OpenFold3-MLX + FOLDSEEK beat Google's AF3?

Usage:
    # Phase 1: Run all local benchmarks
    python ultimate_af3_foldseek_comparison.py --phase local

    # Phase 2: Compare with downloaded AF3 results
    python ultimate_af3_foldseek_comparison.py --phase compare --af3-results path/to/af3/results/

    # Complete workflow
    python ultimate_af3_foldseek_comparison.py --phase all
"""

import argparse
import json
import pandas as pd
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional
import sys

# Add project paths
sys.path.append(str(Path(__file__).parent))

from benchmark_runner_with_foldseek import EnhancedBenchmarkRunner, FoldSeekBenchmarkResult
from compare_af3_results import parse_af3_results
from create_af3server_submission import parse_fasta


@dataclass
class UltimateComparisonResult:
    """Master comparison result across all methods"""

    sequence_id: str
    sequence_length: int

    # OpenFold3-MLX results (multiple template sources)
    foldseek_result: Optional[FoldSeekBenchmarkResult]
    colabfold_result: Optional[FoldSeekBenchmarkResult]
    no_template_result: Optional[FoldSeekBenchmarkResult]

    # Google AF3 results
    af3_plddt: Optional[float]
    af3_available: bool

    # Best local method
    best_local_method: str
    best_local_plddt: float

    # AF3 comparison (if available)
    beats_af3: Optional[bool]
    plddt_advantage: Optional[float]


class UltimateComparisonSuite:
    """Master benchmark suite for the ultimate AF3 comparison"""

    def __init__(self,
                 output_dir: Path = Path("benchmarks/ultimate_comparison_results"),
                 max_sequences: Optional[int] = None):

        # Ensure all paths are relative to project root
        if not output_dir.is_absolute():
            self.output_dir = Path.cwd() / output_dir
        else:
            self.output_dir = output_dir
        self.max_sequences = max_sequences

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_results_dir = self.output_dir / "local_benchmarks"
        self.comparison_dir = self.output_dir / "af3_comparison"

        for dir_path in [self.local_results_dir, self.comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize enhanced benchmark runner for local tests
        # Use absolute paths since we're managing path consistency at this level
        self.local_runner = EnhancedBenchmarkRunner(
            output_dir=self.local_results_dir,
            compare_template_sources=True
        )

        print(f"🎯 Ultimate Comparison Suite initialized")
        print(f"   Output directory: {output_dir}")

    def run_local_benchmarks(self, sequences: Dict[str, str]) -> Dict[str, Dict[str, FoldSeekBenchmarkResult]]:
        """Run all local benchmark variations"""

        print(f"\n{'='*80}")
        print(f"🚀 PHASE 1: LOCAL BENCHMARKS")
        print(f"{'='*80}")
        print(f"Testing: FOLDSEEK, ColabFold, No-template baselines")
        print(f"Sequences: {len(sequences)}")

        # Run comparison benchmark (tests all template sources)
        all_local_results = self.local_runner.run_comparison_benchmark(sequences)

        # Reorganize results by sequence for easier analysis
        by_sequence = {}
        for template_source, results in all_local_results.items():
            for result in results:
                seq_id = result.sequence_id
                if seq_id not in by_sequence:
                    by_sequence[seq_id] = {}
                by_sequence[seq_id][template_source] = result

        # Save organized results
        organized_file = self.local_results_dir / "organized_local_results.json"
        serializable_results = {}
        for seq_id, methods in by_sequence.items():
            serializable_results[seq_id] = {
                method: asdict(result) for method, result in methods.items()
            }

        with open(organized_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n📊 Local benchmark summary:")
        for template_source, results in all_local_results.items():
            if results:
                avg_plddt = sum(r.mean_plddt for r in results) / len(results)
                avg_time = sum(r.wall_time_seconds for r in results) / len(results)
                print(f"   {template_source:12}: {len(results):2} sequences, "
                      f"avg pLDDT: {avg_plddt:5.1f}, avg time: {avg_time:5.1f}s")

        return by_sequence

    def compare_with_af3(self,
                        local_results: Dict[str, Dict[str, FoldSeekBenchmarkResult]],
                        af3_results_dir: Path) -> Dict[str, UltimateComparisonResult]:
        """Compare local results with Google AF3 server results"""

        print(f"\n{'='*80}")
        print(f"⚔️  PHASE 2: AF3 COMPARISON")
        print(f"{'='*80}")

        if not af3_results_dir.exists():
            print(f"❌ AF3 results directory not found: {af3_results_dir}")
            print("   Run with --phase local for local benchmarks only")
            return {}

        # Parse AF3 results
        af3_df = parse_af3_results(af3_results_dir)
        print(f"📊 Loaded {len(af3_df)} AF3 results")

        # Create master comparison results
        ultimate_results = {}

        for seq_id, method_results in local_results.items():

            # Find AF3 result for this sequence
            af3_row = af3_df[af3_df['sequence_id'] == seq_id]
            af3_plddt = None
            af3_available = False

            if not af3_row.empty and af3_row.iloc[0]['af3_available']:
                af3_plddt = af3_row.iloc[0]['af3_plddt']
                af3_available = True

            # Determine best local method
            best_method = "none"
            best_plddt = 0.0

            for method, result in method_results.items():
                if result.mean_plddt > best_plddt:
                    best_plddt = result.mean_plddt
                    best_method = method

            # Calculate AF3 comparison metrics
            beats_af3 = None
            plddt_advantage = None

            if af3_available and af3_plddt is not None:
                plddt_advantage = best_plddt - af3_plddt
                beats_af3 = plddt_advantage > 0

            # Create ultimate comparison result
            ultimate_results[seq_id] = UltimateComparisonResult(
                sequence_id=seq_id,
                sequence_length=method_results[best_method].sequence_length,

                foldseek_result=method_results.get('foldseek'),
                colabfold_result=method_results.get('colabfold'),
                no_template_result=method_results.get('none'),

                af3_plddt=af3_plddt,
                af3_available=af3_available,

                best_local_method=best_method,
                best_local_plddt=best_plddt,

                beats_af3=beats_af3,
                plddt_advantage=plddt_advantage
            )

        # Generate comprehensive comparison report
        self._generate_ultimate_report(ultimate_results)

        return ultimate_results

    def _generate_ultimate_report(self, ultimate_results: Dict[str, UltimateComparisonResult]):
        """Generate the ultimate comparison report"""

        print(f"\n{'='*100}")
        print(f"🏆 ULTIMATE COMPARISON REPORT")
        print(f"{'='*100}")

        results_list = list(ultimate_results.values())
        af3_available_results = [r for r in results_list if r.af3_available]

        print(f"📊 Dataset Overview:")
        print(f"   Total sequences: {len(results_list)}")
        print(f"   AF3 results available: {len(af3_available_results)}")
        print(f"   Coverage: {len(af3_available_results)/len(results_list):.1%}")

        # Method comparison
        print(f"\n🎯 Method Performance (all sequences):")

        for method in ['foldseek', 'colabfold', 'none']:
            method_results = [getattr(r, f"{method}_result") for r in results_list
                            if getattr(r, f"{method}_result") is not None]

            if method_results:
                avg_plddt = sum(r.mean_plddt for r in method_results) / len(method_results)
                avg_time = sum(r.wall_time_seconds for r in method_results) / len(method_results)
                avg_templates = sum(r.template_count for r in method_results) / len(method_results)

                print(f"   {method.upper():12}: {len(method_results):2} sequences, "
                      f"avg pLDDT: {avg_plddt:5.1f}, "
                      f"avg time: {avg_time:5.1f}s, "
                      f"avg templates: {avg_templates:4.1f}")

        # AF3 head-to-head comparison
        if af3_available_results:
            print(f"\n⚔️  Head-to-Head vs Google AF3:")

            # Overall statistics
            local_wins = sum(1 for r in af3_available_results if r.beats_af3)
            af3_wins = sum(1 for r in af3_available_results if r.beats_af3 == False)
            ties = len(af3_available_results) - local_wins - af3_wins

            avg_advantage = sum(r.plddt_advantage for r in af3_available_results
                              if r.plddt_advantage is not None) / len(af3_available_results)

            best_local_avg = sum(r.best_local_plddt for r in af3_available_results) / len(af3_available_results)
            af3_avg = sum(r.af3_plddt for r in af3_available_results) / len(af3_available_results)

            print(f"   OpenFold3-MLX (best): {best_local_avg:.1f} mean pLDDT")
            print(f"   Google AF3:           {af3_avg:.1f} mean pLDDT")
            print(f"   Average advantage:    {avg_advantage:+.1f} pLDDT")
            print(f"\n   Win/Loss Record:")
            print(f"     Local wins:  {local_wins:2}")
            print(f"     AF3 wins:    {af3_wins:2}")
            print(f"     Ties:        {ties:2}")

            # Method breakdown for wins
            method_wins = {'foldseek': 0, 'colabfold': 0, 'none': 0}
            for r in af3_available_results:
                if r.beats_af3:
                    method_wins[r.best_local_method] += 1

            print(f"\n   Wins by method:")
            for method, wins in method_wins.items():
                print(f"     {method.upper():12}: {wins:2} wins")

            # Overall verdict
            print(f"\n🎯 VERDICT:")
            if local_wins > af3_wins:
                print(f"   🎉 OpenFold3-MLX WINS! ({local_wins} vs {af3_wins})")
                print(f"      Average {avg_advantage:.1f} pLDDT advantage")
                if method_wins['foldseek'] > method_wins['colabfold']:
                    print(f"      🚀 FOLDSEEK templates are the key to success!")
            elif local_wins == af3_wins:
                print(f"   ⚖️  TIE GAME! Both systems perform equally well")
            else:
                print(f"   📊 AF3 edges ahead ({af3_wins} vs {local_wins})")
                print(f"      But OpenFold3-MLX is competitive!")

        # Save detailed comparison data
        self._save_ultimate_results(ultimate_results)

    def _save_ultimate_results(self, ultimate_results: Dict[str, UltimateComparisonResult]):
        """Save comprehensive comparison data"""

        # Convert to DataFrame for analysis
        comparison_data = []

        for seq_id, result in ultimate_results.items():
            base_data = {
                'sequence_id': seq_id,
                'sequence_length': result.sequence_length,
                'af3_plddt': result.af3_plddt,
                'af3_available': result.af3_available,
                'best_local_method': result.best_local_method,
                'best_local_plddt': result.best_local_plddt,
                'beats_af3': result.beats_af3,
                'plddt_advantage': result.plddt_advantage
            }

            # Add method-specific data
            for method in ['foldseek', 'colabfold', 'none']:
                method_result = getattr(result, f"{method}_result")
                if method_result:
                    base_data.update({
                        f'{method}_plddt': method_result.mean_plddt,
                        f'{method}_time': method_result.wall_time_seconds,
                        f'{method}_templates': method_result.template_count,
                        f'{method}_msa_depth': method_result.msa_depth
                    })
                else:
                    base_data.update({
                        f'{method}_plddt': None,
                        f'{method}_time': None,
                        f'{method}_templates': None,
                        f'{method}_msa_depth': None
                    })

            comparison_data.append(base_data)

        # Save as CSV for easy analysis
        df = pd.DataFrame(comparison_data)
        csv_file = self.comparison_dir / "ultimate_comparison_results.csv"
        df.to_csv(csv_file, index=False)

        # Save as JSON for complete data
        json_file = self.comparison_dir / "ultimate_comparison_results.json"
        serializable_results = {
            seq_id: asdict(result) for seq_id, result in ultimate_results.items()
        }
        with open(json_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Ultimate comparison results saved:")
        print(f"   CSV: {csv_file}")
        print(f"   JSON: {json_file}")

    def prepare_af3_submission(self, sequences: Dict[str, str]):
        """Prepare Google AF3 server submission"""

        print(f"\n📤 Preparing Google AF3 submission...")

        # Use existing AF3 submission creator
        submission_file = self.output_dir / "af3_submission.json"

        # Create temporary FASTA for submission script
        temp_fasta = self.output_dir / "sequences.fasta"
        with open(temp_fasta, 'w') as f:
            for seq_id, sequence in sequences.items():
                f.write(f">{seq_id}\n{sequence}\n")

        # Run submission creator
        cmd = [
            "python", "create_af3server_submission.py"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0:
            # Move generated file to our output directory
            import shutil
            shutil.move("af3server_submission_casp16.json", submission_file)

            print(f"✅ AF3 submission ready: {submission_file}")
            print(f"   Upload to: https://alphafoldserver.com/")
        else:
            print(f"❌ Failed to create AF3 submission: {result.stderr}")


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description="Ultimate OpenFold3-MLX vs AF3 vs FOLDSEEK Comparison")

    parser.add_argument("--phase", choices=["local", "compare", "all"], default="all",
                       help="Benchmark phase to run")
    parser.add_argument("--fasta", type=Path, default="benchmarks/casp16_monomers.fasta",
                       help="FASTA file with benchmark sequences")
    parser.add_argument("--af3-results", type=Path,
                       help="Directory with downloaded AF3 results")
    parser.add_argument("--output", type=Path, default="benchmarks/ultimate_comparison_results",
                       help="Output directory")
    parser.add_argument("--max-sequences", type=int, default=10,
                       help="Maximum sequences for testing (None for all)")
    parser.add_argument("--prepare-af3-submission", action="store_true",
                       help="Prepare AF3 server submission file")

    args = parser.parse_args()

    # Load sequences
    fasta_tuples = parse_fasta(args.fasta)

    # Convert to dict {id: sequence} and limit if requested
    if args.max_sequences:
        fasta_tuples = fasta_tuples[:args.max_sequences]

    sequences = {seq_id: sequence for seq_id, _, sequence in fasta_tuples}

    print(f"🧬 Loaded {len(sequences)} sequences from {args.fasta}")

    # Initialize comparison suite
    suite = UltimateComparisonSuite(
        output_dir=args.output,
        max_sequences=args.max_sequences
    )

    # Execute requested phases
    if args.phase in ["local", "all"]:
        print(f"\n🚀 Running local benchmarks...")
        local_results = suite.run_local_benchmarks(sequences)

        if args.prepare_af3_submission:
            suite.prepare_af3_submission(sequences)

    if args.phase in ["compare", "all"]:
        if args.phase == "compare":
            # Load existing local results
            local_file = suite.local_results_dir / "organized_local_results.json"
            if not local_file.exists():
                print(f"❌ No local results found. Run --phase local first.")
                return

            with open(local_file, 'r') as f:
                loaded_data = json.load(f)

            # Reconstruct results from JSON
            local_results = {}
            for seq_id, methods in loaded_data.items():
                local_results[seq_id] = {}
                for method, data in methods.items():
                    local_results[seq_id][method] = FoldSeekBenchmarkResult(**data)

        if not args.af3_results:
            print(f"❌ --af3-results required for comparison phase")
            return

        print(f"\n⚔️  Comparing with AF3 results...")
        ultimate_results = suite.compare_with_af3(local_results, args.af3_results)

    print(f"\n✅ Ultimate comparison complete!")
    print(f"   Results in: {args.output}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Review results in ultimate_comparison_results.csv")
    print(f"   2. Use data for your whitepaper")
    print(f"   3. Submit to Google AF3 if you prepared submission")


if __name__ == "__main__":
    main()