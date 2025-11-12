#!/usr/bin/env python3
"""
Test script for OpenFold3-MLX Benchmarker

This script validates that the benchmarker components work correctly
before running the full benchmark suite.

Usage:
    python test_benchmarker.py
"""

import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the benchmarks directory to the path so we can import benchmark_runner
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_runner import (
    FASTAParser, QueryGenerator, SequenceInfo, BenchmarkConfig,
    BenchmarkRunner, SystemMonitor
)

class TestFASTAParser(unittest.TestCase):
    """Test FASTA file parsing"""

    def setUp(self):
        # Create a temporary FASTA file for testing
        self.test_fasta_content = """>test_seq_1 Description 1
MKLLVILAAALLHAPAGGPVSRRKKAEPAKRFLERAGGFEDVGIGAAWVSSLLLALFFGGFGGGGGGLLLLLLLL
>test_seq_2 Description 2|with pipe
ACDEFGHIKLMNPQRSTVWY
>test_seq_3 Short sequence
MKLLVIL
"""

    def test_parse_fasta(self):
        """Test basic FASTA parsing functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(self.test_fasta_content)
            f.flush()

            sequences = FASTAParser.parse_fasta(Path(f.name))

            # Clean up
            Path(f.name).unlink()

        # Verify results
        self.assertEqual(len(sequences), 3)

        # Check first sequence
        self.assertEqual(sequences[0].id, "test_seq_1")
        self.assertEqual(sequences[0].description, "Description 1")
        self.assertEqual(len(sequences[0].sequence), 70)

        # Check second sequence
        self.assertEqual(sequences[1].id, "test_seq_2")
        self.assertEqual(sequences[1].description, "Description 2|with pipe")
        self.assertEqual(sequences[1].sequence, "ACDEFGHIKLMNPQRSTVWY")

        # Check third sequence
        self.assertEqual(sequences[2].id, "test_seq_3")
        self.assertEqual(sequences[2].length, 7)

    def test_sequence_info_clean_id(self):
        """Test filesystem-safe ID generation"""
        seq = SequenceInfo("test|seq:with/bad*chars", "desc", "ACGT", 4)
        self.assertEqual(seq.clean_id, "test_seq_with_bad_chars")

class TestQueryGenerator(unittest.TestCase):
    """Test query JSON generation"""

    def test_create_query_json(self):
        """Test JSON query file generation"""
        seq_info = SequenceInfo("test_seq", "Test sequence", "ACDEFGHIKLMNPQRSTVWY", 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_query.json"
            QueryGenerator.create_query_json(seq_info, output_path)

            # Verify file was created and has correct structure
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                data = json.load(f)

            # Verify structure
            self.assertIn("queries", data)
            self.assertIn("test_seq", data["queries"])

            query = data["queries"]["test_seq"]
            self.assertIn("chains", query)
            self.assertEqual(len(query["chains"]), 1)

            chain = query["chains"][0]
            self.assertEqual(chain["molecule_type"], "protein")
            self.assertEqual(chain["chain_ids"], ["A"])
            self.assertEqual(chain["sequence"], "ACDEFGHIKLMNPQRSTVWY")

class TestBenchmarkConfig(unittest.TestCase):
    """Test benchmark configuration"""

    def test_benchmark_config_validation(self):
        """Test configuration validation"""
        # Valid config should work
        config = BenchmarkConfig(
            name="test",
            description="Test config",
            runner_yaml="test.yaml",
            use_msa_server=True,
            use_templates=False,
            use_mlx_attention=True
        )
        self.assertEqual(config.num_seeds, 1)  # Default value

        # Invalid num_seeds should raise error
        with self.assertRaises(ValueError):
            BenchmarkConfig(
                name="test",
                description="Test config",
                runner_yaml="test.yaml",
                use_msa_server=True,
                use_templates=False,
                use_mlx_attention=True,
                num_seeds=0  # Invalid
            )

class TestSystemMonitor(unittest.TestCase):
    """Test system monitoring functionality"""

    def test_monitoring_lifecycle(self):
        """Test start/stop monitoring cycle"""
        monitor = SystemMonitor()

        # Initially not monitoring
        self.assertFalse(monitor.monitoring)

        # Start monitoring
        monitor.start_monitoring()
        self.assertTrue(monitor.monitoring)

        # Collect some stats
        monitor.collect_stats()
        monitor.collect_stats()

        # Stop and get summary
        summary = monitor.stop_monitoring()
        self.assertFalse(monitor.monitoring)

        # Verify summary structure
        self.assertIn('avg_cpu_percent', summary)
        self.assertIn('peak_memory_gb', summary)
        self.assertIn('avg_gpu_percent', summary)

class TestBenchmarkRunner(unittest.TestCase):
    """Test benchmark runner functionality"""

    def test_config_creation(self):
        """Test benchmark configuration creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "benchmark_output"
            openfold_root = Path(__file__).parent.parent  # Parent directory

            runner = BenchmarkRunner(output_dir, openfold_root)
            configs = runner.create_benchmark_configs()

            # Should create multiple configs
            self.assertGreater(len(configs), 0)

            # Check config names
            config_names = [c.name for c in configs]
            self.assertIn("mlx_with_msa", config_names)
            self.assertIn("mlx_no_msa", config_names)

            # Verify directories were created
            self.assertTrue((output_dir / "configs").exists())
            self.assertTrue((output_dir / "queries").exists())
            self.assertTrue((output_dir / "results").exists())

def run_integration_test():
    """Run a basic integration test to verify the benchmarker can parse CASP16 data"""
    print("Running integration test...")

    # Check if CASP16 FASTA exists
    casp16_fasta = Path(__file__).parent / "casp16_monomers.fasta"
    if not casp16_fasta.exists():
        print("❌ CASP16 FASTA file not found. Please ensure casp16_monomers.fasta is in the benchmarks directory.")
        return False

    try:
        # Parse CASP16 sequences
        sequences = FASTAParser.parse_fasta(casp16_fasta)
        print(f"✅ Parsed {len(sequences)} sequences from CASP16 FASTA")

        # Show some statistics
        lengths = [seq.length for seq in sequences]
        print(f"  Sequence lengths: {min(lengths)} - {max(lengths)} residues")
        print(f"  Average length: {sum(lengths) / len(lengths):.1f} residues")

        # Test query generation for first sequence
        with tempfile.TemporaryDirectory() as tmpdir:
            query_path = Path(tmpdir) / "test_query.json"
            QueryGenerator.create_query_json(sequences[0], query_path)

            with open(query_path) as f:
                query_data = json.load(f)

            print(f"✅ Generated query JSON for {sequences[0].id}")

            # Test config creation
            output_dir = Path(tmpdir) / "test_output"
            runner = BenchmarkRunner(output_dir, Path(__file__).parent.parent)
            configs = runner.create_benchmark_configs()

            print(f"✅ Created {len(configs)} benchmark configurations")
            for config in configs:
                print(f"  - {config.name}")

        print("✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("OpenFold3-MLX Benchmarker Test Suite")
    print("=" * 50)

    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    print("\n" + "=" * 50)

    # Run integration test
    success = run_integration_test()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The benchmarker is ready to use.")
        print("\nTo run the benchmark:")
        print("  python benchmark_runner.py --limit 1 --verbose  # Test run")
        print("  python benchmark_runner.py --limit 5           # Small benchmark")
        print("  python benchmark_runner.py                     # Full benchmark")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()