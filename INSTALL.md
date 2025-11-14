# Installation Guide - OpenFold3-MLX

This guide provides step-by-step instructions for installing OpenFold3-MLX, optimized for Apple Silicon.

## Quick Install (Recommended)

### One-Command Installation

If you've already cloned the repository:

```bash
cd openfold-3-mlx
chmod +x install.sh && ./install.sh
```

### Install from GitHub (Coming Soon)

Once this repository is public, you can install directly:

```bash
wget https://raw.githubusercontent.com/YOUR_ORG/openfold-3-mlx/main/install.sh
chmod +x install.sh && ./install.sh
```

Or with curl:

```bash
curl -O https://raw.githubusercontent.com/YOUR_ORG/openfold-3-mlx/main/install.sh
chmod +x install.sh && ./install.sh
```

## What the Installer Does

The `install.sh` script automatically:

1. ✓ Checks system requirements (Python 3.10+, git)
2. ✓ Detects Apple Silicon for MLX optimizations
3. ✓ Creates a Python virtual environment (`venv/`)
4. ✓ Upgrades pip, setuptools, and wheel
5. ✓ Installs PyTorch
6. ✓ Installs MLX (on Apple Silicon)
7. ✓ Installs OpenFold3-MLX in development mode
8. ✓ Verifies the installation
9. ✓ Provides next steps and usage instructions

## System Requirements

### Minimum Requirements

- **Operating System**: macOS 11+ (for MLX), or Linux (CPU/CUDA only)
- **Python**: 3.10 or higher
- **Memory**: 16 GB RAM minimum, 32 GB recommended
- **Storage**: ~10 GB for dependencies and models

### Recommended for Optimal Performance

- **Hardware**: Apple Silicon (M1/M2/M3/M4)
- **Memory**: 32 GB unified memory or more
- **Storage**: 50 GB free space (for databases and models)

### For CUDA/GPU (Non-Apple Silicon)

- NVIDIA GPU with 16+ GB VRAM
- CUDA 11.8 or higher
- cuDNN compatible with your CUDA version

## Manual Installation

If you prefer to install manually or need to customize the installation:

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_ORG/openfold-3-mlx.git
cd openfold-3-mlx
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade Build Tools

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch

```bash
pip install torch torchvision torchaudio
```

### 5. Install MLX (Apple Silicon Only)

```bash
pip install mlx mlx-lm
```

### 6. Install OpenFold3-MLX

```bash
pip install -e .
```

### 7. Install Optional Dependencies

For MSA alignment with kalign2:

```bash
# Using mamba (faster)
mamba install kalign2 -c bioconda

# Or using conda
conda install kalign2 -c bioconda
```

## Post-Installation Setup

### 1. Download Model Parameters

After installation, download the trained model weights:

```bash
setup_openfold
```

This downloads the AlphaFold3-compatible model parameters (~1.5 GB).

### 2. Verify Installation

Test that everything is working:

```bash
python3 -c "import openfold3; print(f'OpenFold3 version: {openfold3.__version__}')"
```

For Apple Silicon, verify MLX integration:

```bash
python3 -c "from openfold3.core.model.primitives.attention_mlx import is_mlx_available; print(f'MLX available: {is_mlx_available()}')"
```

### 3. Run Test Prediction

Try a quick prediction on the example input:

```bash
run_openfold predict --query_json=examples/example_inference_inputs/query_ubiquitin.json
```

### 4. Enable MLX Optimizations (Apple Silicon)

For 2.1x speedup on Apple Silicon:

```bash
run_openfold predict \
  --query_json=your_query.json \
  --config=openfold3/config/apple_silicon.yaml
```

## Troubleshooting

### Installation Fails with "Python version too old"

Ensure Python 3.10 or higher:

```bash
python3 --version
```

If needed, install a newer Python version:

```bash
# macOS (using Homebrew)
brew install python@3.11

# Or download from python.org
```

### MLX Not Available on Apple Silicon

Try reinstalling MLX:

```bash
pip uninstall mlx mlx-lm
pip install mlx mlx-lm
```

### PyTorch Installation Fails

Specify PyTorch version explicitly:

```bash
pip install torch==2.5.1 torchvision torchaudio
```

### Out of Memory During Prediction

Use chunked attention for long sequences:

```bash
run_openfold predict \
  --query_json=your_query.json \
  --config=openfold3/config/apple_silicon.yaml \
  --globals.model.globals.attention.use_chunked_attention_threshold=512
```

### Virtual Environment Not Activating

Make sure to source (not execute) the activate script:

```bash
# Correct
source venv/bin/activate

# Incorrect (won't work)
./venv/bin/activate
```

### kalign2 Not Found

Install via conda/mamba:

```bash
conda install kalign2 -c bioconda
```

Or use the built-in Python alignment (slower):

```bash
run_openfold predict --query_json=your_query.json --use_python_alignment
```

## Updating OpenFold3-MLX

Since the package is installed in development mode (`-e`), you can update by pulling the latest changes:

```bash
cd openfold-3-mlx
git pull origin main

# Reinstall if dependencies changed
pip install -e .
```

## Uninstallation

To completely remove OpenFold3-MLX:

```bash
# Deactivate virtual environment
deactivate

# Remove the virtual environment
rm -rf venv/

# Remove the repository (optional)
cd ..
rm -rf openfold-3-mlx/
```

## Development Installation

For contributors who want to run tests and development tools:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run MLX-specific tests
pytest tests/test_mlx*.py -v
```

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3/M4)

- MLX provides 2.1x speedup over CPU
- Unified memory allows larger models
- Use `apple_silicon.yaml` config for optimal settings
- Supports mixed precision (float16) efficiently

### Linux with NVIDIA GPU

- Requires CUDA 11.8+ and compatible cuDNN
- Use DeepSpeed4Science kernels for attention
- cuEquivariance kernels for triangle attention
- Configure with `--use_deepspeed_evo_attention=true`

### Intel Mac or Linux CPU

- Falls back to pure PyTorch implementation
- Slower than GPU/MLX but functional
- Use smaller batch sizes and chunk sizes
- Consider cloud GPU instances for production use

## Additional Resources

- **Documentation**: https://openfold-3.readthedocs.io/
- **MLX Integration Details**: See `WOW.md` in this repository
- **Examples**: `examples/example_inference_inputs/`
- **Issues**: Report bugs on GitHub Issues
- **AlphaFold3 Paper**: https://www.nature.com/articles/s41586-024-07487-w

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with:
   - Your system information (OS, Python version, hardware)
   - Full error message
   - Steps to reproduce

## License

OpenFold3-MLX is licensed under the Apache 2.0 License. See `LICENSE` for details.

---

**Happy folding!** 🧬🍎
