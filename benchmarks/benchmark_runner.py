#!/usr/bin/env python3
"""
OpenFold3-MLX Benchmark Runner

This script benchmarks OpenFold3-MLX performance on the CASP16 monomer dataset,
comparing different inference modes (with/without MSA, templates, etc.).

Usage:
    python benchmark_runner.py --fasta casp16_monomers.fasta --output_dir results/
    python benchmark_runner.py --config custom_benchmark.yaml
    python benchmark_runner.py --help

Features:
- Parses CASP16 FASTA sequences and generates query JSON files
- Runs inference with multiple configurations (MSA on/off, templates, etc.)
- Monitors system resources (CPU, memory, GPU utilization)
- Tracks detailed timing for each inference stage
- Generates comprehensive benchmark reports
- Supports cross-platform comparison (MLX vs CUDA)
"""

import argparse
import json
import logging
import os
import platform
import psutil
import shutil
import subprocess
import sys
import time
import traceback
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    name: str
    description: str
    runner_yaml: str
    use_msa_server: bool
    use_templates: bool
    use_mlx_attention: bool
    batch_size: int = 1
    num_seeds: int = 1
    timeout_minutes: int = 30

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be at least 1")
        if self.timeout_minutes < 1:
            raise ValueError("timeout_minutes must be at least 1")

@dataclass
class SequenceInfo:
    """Information about a protein sequence from FASTA"""
    id: str
    description: str
    sequence: str
    length: int

    @property
    def clean_id(self) -> str:
        """Return a filesystem-safe version of the sequence ID"""
        # Remove problematic characters and keep only alphanumeric and safe symbols
        import re
        return re.sub(r'[^\w\-_.]', '_', self.id)

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    sequence_id: str
    config_name: str
    success: bool
    wall_time_seconds: float
    peak_memory_gb: float
    avg_cpu_percent: float
    avg_gpu_percent: Optional[float]
    error_message: Optional[str] = None
    output_files: List[str] = None
    msa_fallback_detected: bool = False
    warnings: List[str] = None
    mean_plddt: Optional[float] = None
    num_samples: int = 0
    structure_files: List[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.warnings is None:
            self.warnings = []
        if self.structure_files is None:
            self.structure_files = []

class FASTAParser:
    """Parser for FASTA files"""

    @staticmethod
    def parse_fasta(fasta_file: Path) -> List[SequenceInfo]:
        """Parse a FASTA file and return sequence information"""
        sequences = []
        current_id = None
        current_desc = None
        current_seq = []

        with open(fasta_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    # Save previous sequence if it exists
                    if current_id is not None:
                        seq_str = ''.join(current_seq)
                        sequences.append(SequenceInfo(
                            id=current_id,
                            description=current_desc,
                            sequence=seq_str,
                            length=len(seq_str)
                        ))

                    # Parse header line
                    header_parts = line[1:].split(' ', 1)
                    current_id = header_parts[0]
                    current_desc = header_parts[1] if len(header_parts) > 1 else ""
                    current_seq = []
                else:
                    # Sequence line
                    current_seq.append(line)

        # Don't forget the last sequence
        if current_id is not None:
            seq_str = ''.join(current_seq)
            sequences.append(SequenceInfo(
                id=current_id,
                description=current_desc,
                sequence=seq_str,
                length=len(seq_str)
            ))

        logger.info(f"Parsed {len(sequences)} sequences from {fasta_file}")
        return sequences

class QueryGenerator:
    """Generates OpenFold3 query JSON files from sequences"""

    @staticmethod
    def create_query_json(seq_info: SequenceInfo, output_path: Path) -> None:
        """Create a query JSON file for a sequence"""
        query_data = {
            "queries": {
                seq_info.clean_id: {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": seq_info.sequence
                        }
                    ]
                }
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(query_data, f, indent=2)

class AccuracyCalculator:
    """Calculates accuracy metrics from OpenFold3 output files"""

    @staticmethod
    def calculate_plddt_from_output(output_dir: Path, sequence_id: str) -> Tuple[Optional[float], int, List[str]]:
        """
        Calculate mean pLDDT from OpenFold3 output directory

        Returns:
            tuple: (mean_plddt, num_samples, structure_files)
        """
        try:
            # Find all confidence files in the output directory
            confidence_files = list(output_dir.glob("**/*confidences.json"))
            structure_files = list(output_dir.glob("**/*model.cif")) + list(output_dir.glob("**/*model.pdb"))

            if not confidence_files:
                logger.warning(f"No confidence files found in {output_dir}")
                return None, 0, [str(f) for f in structure_files]

            all_plddts = []

            for conf_file in confidence_files:
                try:
                    with open(conf_file, 'r') as f:
                        conf_data = json.load(f)

                    if 'plddt' in conf_data:
                        plddt_scores = conf_data['plddt']
                        # Calculate mean pLDDT for this sample
                        sample_mean_plddt = sum(plddt_scores) / len(plddt_scores)
                        all_plddts.append(sample_mean_plddt)

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not parse confidence file {conf_file}: {e}")
                    continue

            if all_plddts:
                # Return the mean pLDDT across all samples
                mean_plddt = sum(all_plddts) / len(all_plddts)
                return mean_plddt, len(all_plddts), [str(f) for f in structure_files]
            else:
                logger.warning(f"No valid pLDDT scores found for {sequence_id}")
                return None, 0, [str(f) for f in structure_files]

        except Exception as e:
            logger.error(f"Error calculating pLDDT for {sequence_id}: {e}")
            return None, 0, []

class SystemMonitor:
    """Monitors system resources during benchmark runs"""

    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_percent': [],
            'memory_gb': [],
            'gpu_percent': [],
            'timestamps': []
        }

    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.stats = {
            'cpu_percent': [],
            'memory_gb': [],
            'gpu_percent': [],
            'timestamps': []
        }

    def stop_monitoring(self):
        """Stop monitoring and return summary stats"""
        self.monitoring = False

        if not self.stats['cpu_percent']:
            return {
                'avg_cpu_percent': 0.0,
                'peak_memory_gb': 0.0,
                'avg_gpu_percent': None
            }

        return {
            'avg_cpu_percent': sum(self.stats['cpu_percent']) / len(self.stats['cpu_percent']),
            'peak_memory_gb': max(self.stats['memory_gb']) if self.stats['memory_gb'] else 0.0,
            'avg_gpu_percent': (
                sum(self.stats['gpu_percent']) / len(self.stats['gpu_percent'])
                if self.stats['gpu_percent'] else None
            )
        }

    def collect_stats(self):
        """Collect current system stats (called periodically)"""
        if not self.monitoring:
            return

        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.used / (1024**3)

        self.stats['cpu_percent'].append(cpu_percent)
        self.stats['memory_gb'].append(memory_gb)
        self.stats['timestamps'].append(time.time())

        # GPU stats (if available)
        gpu_percent = self._get_gpu_utilization()
        if gpu_percent is not None:
            self.stats['gpu_percent'].append(gpu_percent)

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available"""
        try:
            if platform.system() == "Darwin":  # macOS
                # For Apple Silicon, we can use powermetrics or Activity Monitor
                # For now, return None as it's complex to implement
                return None
            else:
                # For NVIDIA GPUs
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].load * 100
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get GPU utilization: {e}")

        return None

class BenchmarkRunner:
    """Main benchmark runner class"""

    def __init__(self, output_dir: Path, openfold_root: Path):
        self.output_dir = Path(output_dir)
        self.openfold_root = Path(openfold_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.queries_dir = self.output_dir / "queries"
        self.configs_dir = self.output_dir / "configs"
        self.results_dir = self.output_dir / "results"

        for dir_path in [self.queries_dir, self.configs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.monitor = SystemMonitor()
        self.results: List[BenchmarkResult] = []

    def create_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Create benchmark configurations for different inference modes"""
        base_config_path = self.openfold_root / "msa_only_runner.yaml"

        configs = [
            # Primary MLX configuration with MSA
            BenchmarkConfig(
                name="mlx_with_msa",
                description="MLX attention with MSA server (high accuracy)",
                runner_yaml=str(base_config_path),
                use_msa_server=True,
                use_templates=False,
                use_mlx_attention=True,
                timeout_minutes=60,  # Generous timeout for MSA processing
            ),
        ]

        # Save configs to files
        for config in configs:
            config_file = self.configs_dir / f"{config.name}.yaml"
            self._create_config_file(config, config_file)

        return configs

    def _create_config_file(self, config: BenchmarkConfig, output_path: Path):
        """Create a YAML config file for a benchmark configuration"""
        config_data = {
            'experiment_settings': {
                'mode': 'predict',
                'use_msa_server': config.use_msa_server,
                'use_templates': config.use_templates,
            },
            'data_module_args': {
                'batch_size': config.batch_size,
                'num_workers': 0,
                'num_workers_validation': 0,
            },
            'pl_trainer_args': {
                'devices': 1,
                'accelerator': 'gpu' if platform.system() != "Darwin" else 'mps',
            },
            'model_update': {
                'presets': ["predict", "pae_enabled"],
                'custom': {
                    'settings': {
                        'memory': {
                            'eval': {
                                'use_deepspeed_evo_attention': not config.use_mlx_attention,
                                'use_lma': False,
                                'use_mlx_attention': config.use_mlx_attention,
                            }
                        }
                    }
                }
            }
        }

        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def run_single_inference(
        self,
        seq_info: SequenceInfo,
        config: BenchmarkConfig,
        timeout_seconds: int = 1800
    ) -> BenchmarkResult:
        """Run inference for a single sequence with a specific configuration"""

        logger.info(f"Running {config.name} for {seq_info.id} (length: {seq_info.length})")

        # Create query file
        query_file = self.queries_dir / f"{seq_info.clean_id}.json"
        QueryGenerator.create_query_json(seq_info, query_file)

        # Output directory for this run
        run_output_dir = self.results_dir / config.name / seq_info.clean_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Config file path
        config_file = self.configs_dir / f"{config.name}.yaml"

        # Prepare command
        cmd = [
            sys.executable, "-m", "openfold3.run_openfold", "predict",
            "--runner_yaml", str(config_file),
            "--query_json", str(query_file),
            "--output_dir", str(run_output_dir),
            "--use_msa_server", str(config.use_msa_server).lower(),
            "--use_templates", str(config.use_templates).lower()
        ]

        start_time = time.time()
        self.monitor.start_monitoring()

        try:
            # Start monitoring thread
            import threading
            stop_monitoring = threading.Event()

            def monitor_loop():
                while not stop_monitoring.wait(1.0):  # Monitor every second
                    self.monitor.collect_stats()

            monitor_thread = threading.Thread(target=monitor_loop)
            monitor_thread.start()

            # Run inference
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.openfold_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()

            wall_time = time.time() - start_time
            stats = self.monitor.stop_monitoring()

            if result.returncode == 0:
                # Success - check for MSA fallback issues
                output_files = list(run_output_dir.glob("**/*"))
                warnings = []
                msa_fallback_detected = False

                # Check stderr for MSA parsing issues with detailed debugging
                stderr_text = result.stderr.lower()
                stdout_text = result.stdout.lower()

                if any(keyword in stderr_text for keyword in [
                    "emptydataerror", "no columns to parse",
                    "pdb70.m8", "pandas", "fallback"
                ]):
                    msa_fallback_detected = True
                    warnings.append("MSA parsing errors detected - may have fallen back to no-MSA mode")

                    # Add detailed debugging for m8 file issues
                    if "pdb70.m8" in stderr_text and "emptydataerror" in stderr_text:
                        warnings.append("DEBUG: Empty pdb70.m8 file detected - ColabFold MSA server returned no templates")
                    if "no columns to parse" in stderr_text:
                        warnings.append("DEBUG: pandas.read_csv failed on .m8 file - file likely empty or malformed")

                # Check for specific m8-related errors in output
                if config.use_msa_server:
                    # Look for ColabFold MSA temp directories mentioned in stderr
                    temp_dir_mentions = []
                    for line in result.stderr.split('\n'):
                        if 'of3_colabfold_msas' in line.lower():
                            temp_dir_mentions.append(line.strip())

                    if temp_dir_mentions:
                        warnings.append(f"DEBUG: ColabFold MSA temp dir: {temp_dir_mentions[0].split()[-1] if temp_dir_mentions else 'unknown'}")

                # Check if MSA was requested but no MSA files were generated
                if config.use_msa_server and not any("msa" in str(f).lower() for f in output_files):
                    msa_fallback_detected = True
                    warnings.append("MSA requested but no MSA output files detected")

                # Add debug info about what files were actually generated
                if config.use_msa_server:
                    warnings.append(f"DEBUG: {len(output_files)} output files generated")

                # Calculate accuracy metrics
                mean_plddt, num_samples, structure_files = AccuracyCalculator.calculate_plddt_from_output(
                    run_output_dir, seq_info.id
                )

                if mean_plddt is not None:
                    logger.info(f"   Mean pLDDT: {mean_plddt:.1f} ({num_samples} samples)")
                else:
                    warnings.append("Could not calculate pLDDT - no confidence files found")

                return BenchmarkResult(
                    sequence_id=seq_info.id,
                    config_name=config.name,
                    success=True,
                    wall_time_seconds=wall_time,
                    peak_memory_gb=stats['peak_memory_gb'],
                    avg_cpu_percent=stats['avg_cpu_percent'],
                    avg_gpu_percent=stats['avg_gpu_percent'],
                    output_files=[str(f) for f in output_files],
                    msa_fallback_detected=msa_fallback_detected,
                    warnings=warnings,
                    mean_plddt=mean_plddt,
                    num_samples=num_samples,
                    structure_files=structure_files
                )
            else:
                # Failure
                error_msg = f"Command failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"

                return BenchmarkResult(
                    sequence_id=seq_info.id,
                    config_name=config.name,
                    success=False,
                    wall_time_seconds=wall_time,
                    peak_memory_gb=stats['peak_memory_gb'],
                    avg_cpu_percent=stats['avg_cpu_percent'],
                    avg_gpu_percent=stats['avg_gpu_percent'],
                    error_message=error_msg
                )

        except subprocess.TimeoutExpired:
            self.monitor.stop_monitoring()
            wall_time = time.time() - start_time

            return BenchmarkResult(
                sequence_id=seq_info.id,
                config_name=config.name,
                success=False,
                wall_time_seconds=wall_time,
                peak_memory_gb=0.0,
                avg_cpu_percent=0.0,
                avg_gpu_percent=None,
                error_message=f"Inference timed out after {timeout_seconds} seconds"
            )

        except Exception as e:
            self.monitor.stop_monitoring()
            wall_time = time.time() - start_time

            return BenchmarkResult(
                sequence_id=seq_info.id,
                config_name=config.name,
                success=False,
                wall_time_seconds=wall_time,
                peak_memory_gb=0.0,
                avg_cpu_percent=0.0,
                avg_gpu_percent=None,
                error_message=f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            )

    def run_benchmark_suite(
        self,
        sequences: List[SequenceInfo],
        configs: List[BenchmarkConfig],
        max_workers: int = 1,
        sequence_limit: Optional[int] = None
    ) -> List[BenchmarkResult]:
        """Run the full benchmark suite"""

        if sequence_limit:
            sequences = sequences[:sequence_limit]
            logger.info(f"Limited to first {sequence_limit} sequences")

        logger.info(f"Running benchmark on {len(sequences)} sequences with {len(configs)} configurations")

        results = []
        total_runs = len(sequences) * len(configs)
        completed_runs = 0

        # For now, run sequentially to avoid resource conflicts
        for seq_info in sequences:
            for config in configs:
                logger.info(f"Progress: {completed_runs + 1}/{total_runs}")

                try:
                    result = self.run_single_inference(
                        seq_info,
                        config,
                        timeout_seconds=config.timeout_minutes * 60
                    )
                    results.append(result)

                    if result.success:
                        logger.info(f"✅ {seq_info.id} with {config.name}: {result.wall_time_seconds:.1f}s")
                    else:
                        logger.warning(f"❌ {seq_info.id} with {config.name}: {result.error_message}")

                except Exception as e:
                    logger.error(f"Unexpected error running {seq_info.id} with {config.name}: {e}")
                    results.append(BenchmarkResult(
                        sequence_id=seq_info.id,
                        config_name=config.name,
                        success=False,
                        wall_time_seconds=0.0,
                        peak_memory_gb=0.0,
                        avg_cpu_percent=0.0,
                        avg_gpu_percent=None,
                        error_message=str(e)
                    ))

                completed_runs += 1

        self.results = results
        return results

    def generate_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report"""

        # Convert results to DataFrame for analysis
        results_data = [asdict(result) for result in results]
        df = pd.DataFrame(results_data)

        # Basic stats
        total_runs = len(results)
        successful_runs = len([r for r in results if r.success])
        failed_runs = total_runs - successful_runs
        msa_fallback_runs = len([r for r in results if r.success and r.msa_fallback_detected])

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'platform': platform.platform(),
                'python_version': sys.version,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'msa_fallback_runs': msa_fallback_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0.0
            },
            'summary_stats': {},
            'per_config_stats': {},
            'per_sequence_stats': {},
            'failures': []
        }

        if successful_runs > 0:
            successful_df = df[df['success'] == True]

            # Calculate accuracy statistics
            successful_df_with_plddt = successful_df[successful_df['mean_plddt'].notna()]
            accuracy_stats = {}

            if len(successful_df_with_plddt) > 0:
                accuracy_stats = {
                    'avg_mean_plddt': successful_df_with_plddt['mean_plddt'].mean(),
                    'median_mean_plddt': successful_df_with_plddt['mean_plddt'].median(),
                    'min_mean_plddt': successful_df_with_plddt['mean_plddt'].min(),
                    'max_mean_plddt': successful_df_with_plddt['mean_plddt'].max(),
                    'plddt_available_runs': len(successful_df_with_plddt),
                    'avg_num_samples': successful_df_with_plddt['num_samples'].mean(),
                }

            report['summary_stats'] = {
                'total_wall_time_hours': successful_df['wall_time_seconds'].sum() / 3600,
                'avg_wall_time_seconds': successful_df['wall_time_seconds'].mean(),
                'median_wall_time_seconds': successful_df['wall_time_seconds'].median(),
                'min_wall_time_seconds': successful_df['wall_time_seconds'].min(),
                'max_wall_time_seconds': successful_df['wall_time_seconds'].max(),
                'avg_peak_memory_gb': successful_df['peak_memory_gb'].mean(),
                'max_peak_memory_gb': successful_df['peak_memory_gb'].max(),
                **accuracy_stats
            }

            # Per-config stats
            for config_name in successful_df['config_name'].unique():
                config_df = successful_df[successful_df['config_name'] == config_name]
                config_df_with_plddt = config_df[config_df['mean_plddt'].notna()]

                config_stats = {
                    'count': len(config_df),
                    'avg_wall_time_seconds': config_df['wall_time_seconds'].mean(),
                    'median_wall_time_seconds': config_df['wall_time_seconds'].median(),
                    'avg_peak_memory_gb': config_df['peak_memory_gb'].mean(),
                }

                if len(config_df_with_plddt) > 0:
                    config_stats.update({
                        'avg_mean_plddt': config_df_with_plddt['mean_plddt'].mean(),
                        'median_mean_plddt': config_df_with_plddt['mean_plddt'].median(),
                        'plddt_count': len(config_df_with_plddt),
                    })

                report['per_config_stats'][config_name] = config_stats

        # Failures
        failed_df = df[df['success'] == False]
        for _, row in failed_df.iterrows():
            report['failures'].append({
                'sequence_id': row['sequence_id'],
                'config_name': row['config_name'],
                'error_message': row['error_message']
            })

        # Warnings and MSA fallbacks
        report['warnings'] = []
        for result in results:
            if result.success and result.warnings:
                for warning in result.warnings:
                    report['warnings'].append({
                        'sequence_id': result.sequence_id,
                        'config_name': result.config_name,
                        'warning': warning
                    })

        return report

    def save_results(self, results: List[BenchmarkResult], report: Dict[str, Any]):
        """Save benchmark results and report"""

        # Save detailed results as JSON
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Save summary report
        report_file = self.output_dir / "benchmark_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Save CSV for easy analysis
        csv_file = self.output_dir / "benchmark_results.csv"
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(csv_file, index=False)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Summary report: {report_file}")
        logger.info(f"Detailed results: {results_file}")
        logger.info(f"CSV data: {csv_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Benchmark OpenFold3-MLX on CASP16 monomer dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--fasta",
        type=Path,
        default="casp16_monomers.fasta",
        help="Path to CASP16 monomers FASTA file"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default="results",
        help="Output directory for benchmark results"
    )

    parser.add_argument(
        "--openfold_root",
        type=Path,
        default=".",
        help="Root directory of OpenFold3 repository"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of sequences to benchmark (for testing)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout per inference in minutes (default: 30)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.fasta.exists():
        logger.error(f"FASTA file not found: {args.fasta}")
        sys.exit(1)

    if not args.openfold_root.exists():
        logger.error(f"OpenFold3 root directory not found: {args.openfold_root}")
        sys.exit(1)

    try:
        # Parse sequences
        logger.info("Parsing FASTA sequences...")
        sequences = FASTAParser.parse_fasta(args.fasta)

        if not sequences:
            logger.error("No sequences found in FASTA file")
            sys.exit(1)

        # Initialize benchmark runner
        logger.info("Initializing benchmark runner...")
        runner = BenchmarkRunner(args.output_dir, args.openfold_root)

        # Create benchmark configurations
        logger.info("Creating benchmark configurations...")
        configs = runner.create_benchmark_configs()

        # Update timeout in configs
        for config in configs:
            config.timeout_minutes = args.timeout

        logger.info(f"Created {len(configs)} benchmark configurations:")
        for config in configs:
            logger.info(f"  - {config.name}: {config.description}")

        # Run benchmark suite
        logger.info("Starting benchmark suite...")
        start_time = time.time()

        results = runner.run_benchmark_suite(
            sequences=sequences,
            configs=configs,
            sequence_limit=args.limit
        )

        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.1f} seconds")

        # Generate and save report
        logger.info("Generating benchmark report...")
        report = runner.generate_report(results)
        runner.save_results(results, report)

        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Total runs: {report['metadata']['total_runs']}")
        print(f"Successful: {report['metadata']['successful_runs']}")
        print(f"Failed: {report['metadata']['failed_runs']}")
        print(f"MSA fallbacks detected: {report['metadata']['msa_fallback_runs']}")
        print(f"Success rate: {report['metadata']['success_rate']:.1%}")

        if report['metadata']['successful_runs'] > 0:
            print(f"\nTiming Summary:")
            print(f"  Average: {report['summary_stats']['avg_wall_time_seconds']:.1f}s")
            print(f"  Median: {report['summary_stats']['median_wall_time_seconds']:.1f}s")
            print(f"  Range: {report['summary_stats']['min_wall_time_seconds']:.1f}s - {report['summary_stats']['max_wall_time_seconds']:.1f}s")

            print(f"\nMemory Summary:")
            print(f"  Average peak: {report['summary_stats']['avg_peak_memory_gb']:.1f} GB")
            print(f"  Maximum peak: {report['summary_stats']['max_peak_memory_gb']:.1f} GB")

            if 'avg_mean_plddt' in report['summary_stats']:
                print(f"\nAccuracy Summary (pLDDT):")
                print(f"  Average: {report['summary_stats']['avg_mean_plddt']:.1f}")
                print(f"  Median: {report['summary_stats']['median_mean_plddt']:.1f}")
                print(f"  Range: {report['summary_stats']['min_mean_plddt']:.1f} - {report['summary_stats']['max_mean_plddt']:.1f}")
                print(f"  Available for: {report['summary_stats']['plddt_available_runs']}/{report['metadata']['successful_runs']} runs")
                print(f"  Average samples per run: {report['summary_stats']['avg_num_samples']:.1f}")

        if report['metadata']['failed_runs'] > 0:
            print(f"\nFailures:")
            for failure in report['failures'][:5]:  # Show first 5 failures
                print(f"  - {failure['sequence_id']} ({failure['config_name']}): {failure['error_message'][:100]}...")
            if len(report['failures']) > 5:
                print(f"  ... and {len(report['failures']) - 5} more")

        if report['metadata']['msa_fallback_runs'] > 0:
            print(f"\nMSA Issues Detected:")
            warnings_shown = 0
            for warning in report['warnings'][:10]:  # Show first 10 warnings
                if 'DEBUG' in warning['warning']:
                    print(f"  - {warning['sequence_id']} ({warning['config_name']}): {warning['warning']}")
                    warnings_shown += 1
            if len(report['warnings']) > warnings_shown:
                print(f"  ... and {len(report['warnings']) - warnings_shown} more debug messages")

            print(f"\n💡 MSA Troubleshooting:")
            print(f"   - Check internet connection for ColabFold MSA server")
            print(f"   - Verify ColabFold server is accessible")
            print(f"   - Consider using 'mlx_no_msa' config for reliable benchmarking")

        print(f"\nDetailed results saved to: {args.output_dir}")
        print("="*80)

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()