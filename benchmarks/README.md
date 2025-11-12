# OpenFold3-MLX Benchmarking Suite

This directory contains tools for benchmarking OpenFold3-MLX performance on the CASP16 monomer dataset and comparing different inference configurations.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Default Benchmark

```bash
# Run benchmark on first 5 sequences (for testing)
python benchmark_runner.py --limit 5

# Run full benchmark on all CASP16 sequences
python benchmark_runner.py

# Run with custom timeout (default is 30 minutes per sequence)
python benchmark_runner.py --timeout 60 --limit 10
```

### 3. View Results

Results are saved to `results/` directory by default:
- `benchmark_results.json` - Detailed results for each run
- `benchmark_report.json` - Summary statistics and analysis
- `benchmark_results.csv` - Tabular data for easy analysis
- `configs/` - Generated configuration files
- `queries/` - Generated query JSON files
- `results/` - OpenFold3 output files organized by config and sequence

## Features

### 🔬 **Comprehensive Benchmarking**
- Automatically parses CASP16 FASTA sequences
- Tests multiple inference configurations:
  - MLX with MSA server
  - MLX without MSA (faster)
  - MLX with templates
  - CPU fallback modes
- Measures wall-clock time, memory usage, and CPU/GPU utilization

### 📊 **Detailed Monitoring**
- Real-time system resource monitoring
- Peak memory tracking
- GPU utilization (when available)
- Cross-platform compatibility (macOS M1/M2, Linux CUDA, CPU)

### 📈 **Rich Reporting**
- Summary statistics across all runs
- Per-configuration performance analysis
- Failure analysis with error details
- CSV export for further analysis
- Hardware and platform information

## Usage Examples

### Basic Usage

```bash
# Test on a few sequences
python benchmark_runner.py --limit 3 --verbose

# Full benchmark with custom output directory
python benchmark_runner.py --output_dir my_benchmark_results

# Specify custom FASTA file
python benchmark_runner.py --fasta my_sequences.fasta
```

### Advanced Configuration

```bash
# Use custom OpenFold3 installation
python benchmark_runner.py --openfold_root /path/to/openfold3

# Extended timeout for large sequences
python benchmark_runner.py --timeout 120

# Quick test run
python benchmark_runner.py --limit 1 --timeout 5 --verbose
```

## Benchmark Configurations

The benchmarker runs with this optimized configuration:

| Configuration | MSA Server | Templates | MLX Attention | Description |
|--------------|------------|-----------|---------------|-------------|
| `mlx_with_msa` | ✅ | ❌ | ✅ | MLX with MSA server (high accuracy) |

**Key Features:**
- **✅ MSA Fixed**: Resolved pandas `.m8` file parsing errors
- **✅ pLDDT Calculation**: Automatic accuracy assessment for all predictions
- **✅ Streamlined**: Single high-quality configuration (no low-accuracy no-MSA mode)
- **❌ Template Support**: Temporarily disabled (parsing bugs in template alignment code)

## Understanding Results

### Performance Metrics
- **Wall Time**: Total time from start to finish for inference
- **Memory Usage**: Peak memory consumption during inference
- **CPU/GPU Utilization**: System resource usage patterns

### Accuracy Metrics
- **pLDDT**: Predicted Local Distance Difference Test scores (0-100, higher is better)
  - **Average pLDDT**: Mean confidence across all residues and samples
  - **Range**: Distribution of pLDDT scores across sequences
  - **Sample Count**: Number of diffusion samples generated per sequence

### Success Metrics
- **Success Rate**: Percentage of sequences that completed successfully
- **MSA Integration**: Verification that MSA processing worked correctly
- **Structure Generation**: Number of 3D structure files created per sequence

### Benchmarking Focus
With the optimized configuration, the benchmarker measures:
- **MLX Performance**: Apple Silicon-specific optimizations and speed
- **MSA Quality**: Impact of multiple sequence alignments on accuracy
- **Memory Efficiency**: Resource usage patterns for different sequence lengths
- **Accuracy Consistency**: pLDDT score distribution across the CASP16 dataset

## Files Description

- `benchmark_runner.py` - Main benchmarking script
- `casp16_monomers.fasta` - CASP16 monomer sequences for benchmarking
- `requirements.txt` - Python dependencies
- `config_samples/` - Example custom configuration files
- `README.md` - This file

## CASP16 Dataset

The `casp16_monomers.fasta` file contains protein sequences from the CASP16 competition:

- **57 sequences** ranging from 32 to 818 residues
- **Diverse protein types**: enzymes, membrane proteins, viral proteins, etc.
- **Challenging targets** that test folding accuracy and speed

### Example Sequences
- `T1219` - Human HD6 (32 residues) - Small antimicrobial peptide
- `T1214` - E. coli YncD (677 residues) - Large multi-domain protein
- `T1295` - Nanoparticle antigen (469 residues) - Complex synthetic protein

## Hardware Optimization

### Apple Silicon (M1/M2/M3)
The benchmarker automatically enables MLX optimizations:
- Native MLX attention kernels
- Unified memory optimization
- Apple Silicon-specific settings

### NVIDIA GPUs
Fallback to PyTorch with CUDA:
- DeepSpeed attention kernels
- Mixed precision training
- Tensor core utilization

### CPU-Only
Conservative settings for CPU inference:
- Lower batch sizes
- Chunked attention
- Memory-efficient operations

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**
   - Reduce `--limit` to test fewer sequences
   - Increase `--timeout` for large proteins
   - Check system RAM availability

3. **MSA Server Timeout**
   - Ensure internet connection for ColabFold MSA server
   - Consider using `mlx_no_msa` configuration for offline testing

4. **GPU Not Detected**
   - Verify GPU drivers and CUDA installation
   - Check PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`

### Getting Help

1. **Run with verbose logging**:
   ```bash
   python benchmark_runner.py --verbose --limit 1
   ```

2. **Check OpenFold3 installation**:
   ```bash
   python -m openfold3.run_openfold --help
   ```

3. **Test individual sequence**:
   ```bash
   python -m openfold3.run_openfold predict --runner_yaml msa_only_runner.yaml --query_json examples/example_inference_inputs/query_ubiquitin.json --output_dir test_output
   ```

## Contributing

To extend the benchmarker:

1. **Add new configurations** in `BenchmarkRunner.create_benchmark_configs()`
2. **Enhance monitoring** in `SystemMonitor` class
3. **Improve reporting** in `BenchmarkRunner.generate_report()`
4. **Add visualization** with matplotlib/seaborn

## License

This benchmarking suite is part of OpenFold3-MLX and follows the same license terms.