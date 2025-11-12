#!/usr/bin/env python3
"""
Demo script showing the benchmarker working and detecting MSA issues

This script runs a minimal test to demonstrate the benchmarker functionality
and the MSA fallback detection.

Usage:
    python demo_benchmarker.py
"""

import json
import tempfile
from pathlib import Path
import sys
import subprocess

def create_test_fasta():
    """Create a small test FASTA with a few short sequences"""
    test_fasta = """>test_short Easy test sequence (20 residues)
ACDEFGHIKLMNPQRSTVWY
>test_medium Medium test sequence (40 residues)
ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY
"""
    return test_fasta

def run_benchmark_demo():
    """Run a demo benchmark to show functionality"""
    print("OpenFold3-MLX Benchmarker Demo")
    print("=" * 50)
    print("This demo will:")
    print("1. Create test sequences")
    print("2. Run benchmark with both MSA and no-MSA configs")
    print("3. Show MSA fallback detection")
    print("4. Generate example reports")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test FASTA
        fasta_file = tmpdir / "test_sequences.fasta"
        with open(fasta_file, 'w') as f:
            f.write(create_test_fasta())

        print(f"✅ Created test FASTA: {fasta_file}")
        print(f"   - 2 test sequences (20 and 40 residues)")
        print()

        # Run benchmark
        cmd = [
            sys.executable, "benchmark_runner.py",
            "--fasta", str(fasta_file),
            "--output_dir", str(tmpdir / "demo_results"),
            "--timeout", "10",  # Short timeout for demo
            "--verbose"
        ]

        print("🚀 Running benchmark...")
        print(f"Command: {' '.join(cmd[-6:])}")
        print()

        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute max
            )

            print("📊 Benchmark Results:")
            print("-" * 30)
            if result.returncode == 0:
                print("✅ Benchmark completed successfully!")
                print()
                # Show last part of stdout (the summary)
                stdout_lines = result.stdout.split('\n')
                summary_start = -1
                for i, line in enumerate(stdout_lines):
                    if "BENCHMARK SUMMARY" in line:
                        summary_start = i
                        break

                if summary_start >= 0:
                    for line in stdout_lines[summary_start:]:
                        if line.strip():
                            print(line)
                else:
                    print("Summary not found in output")

            else:
                print(f"❌ Benchmark failed with return code {result.returncode}")
                print("STDERR:")
                print(result.stderr[-1000:])  # Last 1000 chars

            # Show generated files
            results_dir = tmpdir / "demo_results"
            if results_dir.exists():
                print(f"\n📁 Generated Files:")
                for file_path in sorted(results_dir.glob("**/*")):
                    if file_path.is_file():
                        size_kb = file_path.stat().st_size / 1024
                        print(f"  - {file_path.name}: {size_kb:.1f} KB")

                # Show sample of results JSON if it exists
                results_json = results_dir / "benchmark_results.json"
                if results_json.exists():
                    print(f"\n📄 Sample Results Data:")
                    with open(results_json) as f:
                        data = json.load(f)

                    if data:
                        sample_result = data[0]
                        print(f"  Sequence: {sample_result.get('sequence_id')}")
                        print(f"  Config: {sample_result.get('config_name')}")
                        print(f"  Success: {sample_result.get('success')}")
                        print(f"  Time: {sample_result.get('wall_time_seconds', 0):.1f}s")
                        print(f"  MSA Fallback: {sample_result.get('msa_fallback_detected', False)}")
                        if sample_result.get('warnings'):
                            print(f"  Warnings: {len(sample_result['warnings'])}")

        except subprocess.TimeoutExpired:
            print("⏰ Demo timed out (benchmark taking too long)")
        except Exception as e:
            print(f"❌ Demo failed: {e}")

def main():
    """Main entry point"""
    print("Note: This demo requires OpenFold3 to be properly installed.")
    print("Run 'python quick_test.py' first to verify setup.")
    print()

    response = input("Continue with demo? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Demo cancelled.")
        return

    print()
    run_benchmark_demo()

if __name__ == "__main__":
    main()