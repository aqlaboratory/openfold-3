#!/bin/bash
################################################################################
# OpenFold3-MLX Installation Script
#
# This script automates the installation of OpenFold3-MLX for Apple Silicon.
# It creates a virtual environment, installs dependencies, and sets up the
# package in development mode.
#
# Usage:
#   chmod +x install.sh && ./install.sh
#
# Or with wget:
#   wget https://raw.githubusercontent.com/YOUR_ORG/openfold-3-mlx/main/install.sh
#   chmod +x install.sh && ./install.sh
#
# Copyright 2025 AlQuraishi Laboratory
################################################################################

set -e  # Exit on error

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Unicode symbols for progress
CHECK_MARK="✓"
CROSS_MARK="✗"
ARROW="→"
GEAR="⚙"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}${GEAR}${NC} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK_MARK}${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}${CROSS_MARK}${NC} $1"
}

print_info() {
    echo -e "${CYAN}${ARROW}${NC} $1"
}

################################################################################
# Pre-flight Checks
################################################################################

print_header "OpenFold3-MLX Installer"

echo -e "${BOLD}Welcome to the OpenFold3-MLX installation script!${NC}"
echo "This script will set up OpenFold3-MLX on your system."
echo ""
echo "What this script does:"
echo "  1. Checks system requirements"
echo "  2. Creates a Python virtual environment"
echo "  3. Installs required dependencies"
echo "  4. Installs OpenFold3-MLX in development mode"
echo "  5. Verifies the installation"
echo ""

# Check if running on macOS (recommended for MLX)
print_step "Checking operating system..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_success "Running on macOS"

    # Check if Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        print_success "Apple Silicon detected (MLX optimizations will be available)"
        IS_APPLE_SILICON=true
    else
        print_warning "Intel Mac detected (MLX optimizations require Apple Silicon)"
        IS_APPLE_SILICON=false
    fi
else
    print_warning "Not running on macOS (MLX optimizations are only available on Apple Silicon)"
    print_info "Installation will continue, but MLX features will not be available"
    IS_APPLE_SILICON=false
fi

# Check Python version
print_step "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    print_success "Python $PYTHON_VERSION detected (>= 3.10 required)"
else
    print_error "Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Check for git
print_step "Checking for git..."
if ! command -v git &> /dev/null; then
    print_error "git is not installed. Please install git first."
    exit 1
fi
print_success "git is installed"

# Check if we're in the repository
print_step "Checking repository structure..."
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the repository root."
    exit 1
fi

if [ ! -d "openfold3" ]; then
    print_error "openfold3 directory not found. Please run this script from the repository root."
    exit 1
fi
print_success "Repository structure verified"

################################################################################
# Virtual Environment Setup
################################################################################

print_header "Setting Up Virtual Environment"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at ./$VENV_DIR"
    read -p "Do you want to remove it and create a fresh environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        print_success "Removed existing virtual environment"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_step "Creating virtual environment at ./$VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

print_step "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Build tools upgraded"

################################################################################
# Dependency Installation
################################################################################

print_header "Installing Dependencies"

# Install PyTorch first (important for compatibility)
print_step "Installing PyTorch..."
print_info "This may take a few minutes..."
pip install torch torchvision torchaudio > /dev/null 2>&1
print_success "PyTorch installed"

# Install MLX if on Apple Silicon
if [ "$IS_APPLE_SILICON" = true ]; then
    print_step "Installing MLX for Apple Silicon optimizations..."
    if pip install mlx mlx-lm > /dev/null 2>&1; then
        print_success "MLX installed successfully"
    else
        print_warning "MLX installation failed (optional, will use PyTorch backend)"
    fi
else
    print_info "Skipping MLX installation (not on Apple Silicon)"
fi

# Install the package in editable mode with dependencies
print_step "Installing OpenFold3-MLX and remaining dependencies..."
print_info "This may take several minutes as dependencies are compiled..."

# Show a progress indicator
pip install -e . 2>&1 | while IFS= read -r line; do
    if [[ $line == *"Successfully installed"* ]]; then
        echo "$line"
    elif [[ $line == *"Requirement already satisfied"* ]]; then
        : # Suppress these messages
    elif [[ $line == *"ERROR"* ]] || [[ $line == *"Error"* ]]; then
        echo -e "${RED}$line${NC}"
    fi
done

if [ $? -eq 0 ]; then
    print_success "OpenFold3-MLX installed successfully"
else
    print_error "Installation failed. Please check the error messages above."
    exit 1
fi

################################################################################
# Optional Dependencies
################################################################################

print_header "Optional Dependencies"

# Check for conda/mamba for kalign2
print_step "Checking for kalign2 (required for MSA alignment)..."
if command -v kalign &> /dev/null; then
    print_success "kalign2 is already installed"
elif command -v mamba &> /dev/null; then
    print_info "Installing kalign2 via mamba..."
    mamba install -y kalign2 -c bioconda > /dev/null 2>&1
    print_success "kalign2 installed via mamba"
elif command -v conda &> /dev/null; then
    print_info "Installing kalign2 via conda..."
    conda install -y kalign2 -c bioconda > /dev/null 2>&1
    print_success "kalign2 installed via conda"
else
    print_warning "kalign2 not found and conda/mamba not available"
    print_info "Please install kalign2 manually for MSA alignment:"
    print_info "  conda install kalign2 -c bioconda"
    print_info "  or"
    print_info "  mamba install kalign2 -c bioconda"
fi

################################################################################
# Verification
################################################################################

print_header "Verifying Installation"

# Test import
print_step "Testing OpenFold3 import..."
if python3 -c "import openfold3; print(f'OpenFold3 version: {openfold3.__version__}')" 2>/dev/null; then
    VERSION=$(python3 -c "import openfold3; print(openfold3.__version__)")
    print_success "OpenFold3 imported successfully (version $VERSION)"
else
    print_error "Failed to import openfold3"
    exit 1
fi

# Test MLX availability
if [ "$IS_APPLE_SILICON" = true ]; then
    print_step "Testing MLX integration..."
    MLX_TEST=$(python3 -c "
try:
    from openfold3.core.model.primitives.attention_mlx import is_mlx_available
    if is_mlx_available():
        print('available')
    else:
        print('not_available')
except Exception as e:
    print('error')
" 2>/dev/null)

    if [ "$MLX_TEST" = "available" ]; then
        print_success "MLX integration is working"
    elif [ "$MLX_TEST" = "not_available" ]; then
        print_warning "MLX is installed but not available (may need to reinstall mlx)"
    else
        print_warning "Could not verify MLX integration"
    fi
fi

# Check for required binaries
print_step "Checking for command-line tools..."
if command -v run_openfold &> /dev/null; then
    print_success "run_openfold command is available"
else
    print_warning "run_openfold command not found in PATH"
    print_info "You may need to activate the virtual environment first"
fi

################################################################################
# Model Setup Information
################################################################################

print_header "Next Steps"

echo -e "${BOLD}Installation complete!${NC}"
echo ""
echo "To get started:"
echo ""
echo -e "${CYAN}1.${NC} Activate the virtual environment (if not already activated):"
echo -e "   ${BOLD}source $VENV_DIR/bin/activate${NC}"
echo ""
echo -e "${CYAN}2.${NC} Download model parameters:"
echo -e "   ${BOLD}setup_openfold${NC}"
echo ""
echo -e "${CYAN}3.${NC} Run your first prediction:"
echo -e "   ${BOLD}run_openfold predict --query_json=examples/example_inference_inputs/query_ubiquitin.json${NC}"
echo ""

if [ "$IS_APPLE_SILICON" = true ]; then
    echo -e "${CYAN}4.${NC} ${GREEN}(Apple Silicon)${NC} Use MLX optimizations for 2.1x speedup:"
    echo -e "   ${BOLD}run_openfold predict --query_json=your_query.json \\${NC}"
    echo -e "   ${BOLD}    --config=openfold3/config/apple_silicon.yaml${NC}"
    echo ""
fi

echo "Documentation:"
echo -e "  ${CYAN}${ARROW}${NC} Full docs: https://openfold-3.readthedocs.io/"
echo -e "  ${CYAN}${ARROW}${NC} MLX integration details: See WOW.md in this repository"
echo -e "  ${CYAN}${ARROW}${NC} Examples: examples/example_inference_inputs/"
echo ""

echo -e "${GREEN}${BOLD}Happy folding! 🧬${NC}"
echo ""

################################################################################
# Cleanup
################################################################################

# Deactivate message (virtual environment stays activated for user)
print_info "Virtual environment is still activated"
print_info "Run 'deactivate' to exit the virtual environment when done"
