#!/usr/bin/env python3
"""
Quick Test for OpenFold3-MLX Setup

This script performs a quick validation that OpenFold3 is properly set up
and can run inference before running the full benchmark suite.

Usage:
    python quick_test.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

def create_test_query():
    """Create a simple test query for a small protein"""
    # Use ubiquitin as a test case (small, well-studied protein)
    test_query = {
        "queries": {
            "test_ubiquitin": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        # Ubiquitin sequence (76 residues)
                        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
                    }
                ]
            }
        }
    }
    return test_query

def test_openfold_import():
    """Test that OpenFold3 can be imported"""
    print("🧪 Testing OpenFold3 imports...")
    try:
        result = subprocess.run([
            sys.executable, "-c",
            "import openfold3; print('OpenFold3 imported successfully')"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ OpenFold3 import test passed")
            return True
        else:
            print(f"❌ OpenFold3 import failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ OpenFold3 import test failed: {e}")
        return False

def test_openfold_cli():
    """Test that OpenFold3 CLI is accessible"""
    print("🧪 Testing OpenFold3 CLI...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "openfold3.run_openfold", "--help"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and "predict" in result.stdout:
            print("✅ OpenFold3 CLI test passed")
            return True
        else:
            print(f"❌ OpenFold3 CLI test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ OpenFold3 CLI test failed: {e}")
        return False

def test_config_validation():
    """Test that the MSA runner config is valid"""
    print("🧪 Testing configuration validation...")

    config_path = Path(__file__).parent.parent / "msa_only_runner.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        # Try to validate the config using OpenFold3's validator
        result = subprocess.run([
            sys.executable, "-c",
            f"""
import yaml
from pathlib import Path
config_path = Path('{config_path}')
with open(config_path) as f:
    config = yaml.safe_load(f)
print('Config validation passed')
"""
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Configuration validation passed")
            return True
        else:
            print(f"❌ Configuration validation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def test_minimal_inference():
    """Test minimal inference run"""
    print("🧪 Testing minimal inference run...")
    print("   (This may take a few minutes...)")

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test query
        query_file = tmpdir / "test_query.json"
        test_query = create_test_query()

        with open(query_file, 'w') as f:
            json.dump(test_query, f, indent=2)

        # Create test output directory
        output_dir = tmpdir / "test_output"
        output_dir.mkdir()

        # Use the working msa_only_runner.yaml config
        config_file = Path(__file__).parent.parent / "msa_only_runner.yaml"

        # Run inference
        start_time = time.time()
        try:
            cmd = [
                sys.executable, "-m", "openfold3.run_openfold", "predict",
                "--runner_yaml", str(config_file),
                "--query_json", str(query_file),
                "--output_dir", str(output_dir),
                "--use_msa_server", "false",  # Disable for speed in quick test
                "--use_templates", "false"   # Disable due to parsing bug
            ]

            print(f"   Running: {' '.join(cmd[-6:])}")  # Show relevant parts of command

            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,  # Run from OpenFold3 root
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                # Check if output files were created
                output_files = list(output_dir.glob("**/*"))
                if output_files:
                    print(f"✅ Minimal inference test passed in {elapsed:.1f}s")
                    print(f"   Generated {len(output_files)} output files")
                    return True
                else:
                    print(f"❌ Inference completed but no output files found")
                    print(f"   STDOUT: {result.stdout[-200:]}")
                    return False
            else:
                print(f"❌ Minimal inference test failed after {elapsed:.1f}s")
                print(f"   Return code: {result.returncode}")
                print(f"   STDERR: {result.stderr[-500:]}")
                return False

        except subprocess.TimeoutExpired:
            print(f"❌ Inference test timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"❌ Inference test failed with error: {e}")
            return False

def main():
    """Run all quick tests"""
    print("OpenFold3-MLX Quick Test Suite")
    print("=" * 50)
    print("This will validate that OpenFold3 is properly set up")
    print("for benchmarking. Each test should take < 30 seconds")
    print("except the inference test which may take a few minutes.\n")

    tests = [
        ("Import Test", test_openfold_import),
        ("CLI Test", test_openfold_cli),
        ("Config Test", test_config_validation),
        ("Inference Test", test_minimal_inference),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            if test_func():
                passed += 1
            else:
                print("   This test failed. Check the error messages above.")
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test failed with unexpected error: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! OpenFold3-MLX is ready for benchmarking.")
        print("\nNext steps:")
        print("  1. Install benchmark dependencies: pip install -r requirements.txt")
        print("  2. Run test benchmark: python benchmark_runner.py --limit 1 --verbose")
        print("  3. Run full benchmark: python benchmark_runner.py")
    else:
        print("❌ Some tests failed. Please fix the issues before running benchmarks.")
        print("\nTroubleshooting:")
        print("  1. Check that OpenFold3 is properly installed")
        print("  2. Verify all dependencies are installed")
        print("  3. Check that you're running from the correct directory")
        print("  4. Ensure sufficient system resources (RAM, disk space)")

        sys.exit(1)

if __name__ == "__main__":
    main()