#!/usr/bin/env python3
"""
Test script to verify path consistency from project root
"""

from pathlib import Path
import sys

def test_paths():
    """Test that all expected paths exist from project root"""

    print("🧪 Testing path consistency from project root")
    print(f"Current working directory: {Path.cwd()}")

    # Expected files/directories from project root
    expected_paths = [
        "benchmarks/casp16_monomers.fasta",
        "benchmarks/benchmark_runner.py",
        "benchmarks/ultimate_af3_foldseek_comparison.py",
        "openfold3/run_openfold.py",
        "openfold3/__init__.py"
    ]

    all_good = True

    for path_str in expected_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ {path_str}")
        else:
            print(f"❌ {path_str} - NOT FOUND")
            all_good = False

    print(f"\n📁 Test output directories (will be created):")
    test_dirs = [
        "benchmarks/ultimate_comparison_results",
        "benchmarks/enhanced_benchmark_results",
        "benchmarks/parallel_output"
    ]

    for dir_str in test_dirs:
        dir_path = Path(dir_str)
        print(f"📂 {dir_str} -> {dir_path.absolute()}")

    if all_good:
        print(f"\n✅ All paths are accessible from project root!")
        print(f"🚀 Scripts should work correctly when run from: {Path.cwd()}")
    else:
        print(f"\n❌ Some paths are missing. Check your project structure.")

    return all_good

if __name__ == "__main__":
    success = test_paths()
    sys.exit(0 if success else 1)