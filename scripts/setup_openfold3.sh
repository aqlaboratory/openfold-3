# Usage: ./scripts/setup_openfold3.sh

#!/bin/bash

# Global variable for parameter directory
# Configuration variables
OPENFOLD_CACHE=""
# Path for where the checkpoints were saved previously
CKPT_PATH_FILE=""
PARAM_DIR=""

setup_conda_commands(){
    echo "Setting up conda shell environment..."
    # Check if running in a conda environment
    if [ -z "$CONDA_PREFIX" ]; then
        echo "Error: This script must be run from within a conda environment."
        echo "Please activate your conda environment first:"
        echo "  conda activate your_env_name"
        exit 1
    fi

    # Initialize conda by sourcing conda.sh
    if [ -n "$CONDA_EXE" ]; then
        CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Warning: Could not find conda executable. Conda commands may not work."
    fi
}

# Function to set up the OpenFold cache directory
setup_openfold_cache() {
    echo "Setting up OpenFold cache directory..."
    echo "Please specify the OpenFold cache directory (default: $HOME/.openfold3):"
    read -r user_input
    
    # Use user input if provided, otherwise use default
    if [ -z "$user_input" ]; then
        OPENFOLD_CACHE="$HOME/.openfold3"
    else
        OPENFOLD_CACHE="$user_input"
    fi
    
    # Expand tilde to home directory if present
    OPENFOLD_CACHE="${OPENFOLD_CACHE/#\~/$HOME}"
    
    # Create the cache directory if it doesn't exist
    mkdir -p "$OPENFOLD_CACHE"
    
    # Set the param path file location
    CKPT_PATH_FILE="$OPENFOLD_CACHE/ckpt_path.txt"
    
    # Export as environment variable
    export OPENFOLD_CACHE="$OPENFOLD_CACHE"
    conda env config vars set OPENFOLD_CACHE="$OPENFOLD_CACHE"
    echo "OPENFOLD_CACHE environment variable set to: $OPENFOLD_CACHE"
}

# Function to check and set up the parameter directory
setup_param_directory() {
    
    # Check if parameters have already been downloaded
    if [ -f $CKPT_PATH_FILE ]; then
        EXISTING_PATH=$(cat "$CKPT_PATH_FILE")
        echo "OpenFold3 parameters may already be installed at: $EXISTING_PATH"
        echo "Do you want to:"
        echo "1) Use existing parameters (skip download)"
        echo "2) Download to a new location"
        echo "3) Re-download to existing location"
        read -p "Enter your choice (1/2/3): " choice
        
        case $choice in
            1)
                echo "Using existing parameters at: $EXISTING_PATH" 
                PARAM_DIR="$EXISTING_PATH"
                return 1  # Return non-zero to skip download
                ;;
            2)
                echo "Please specify a new directory to save the parameters:"
                read -r user_input
                if [ -z "$user_input" ]; then
                    echo "No directory specified. Exiting."
                    exit 1
                fi
                PARAM_DIR="$user_input"
                ;;
            3)
                echo "Re-downloading to: $EXISTING_PATH"
                PARAM_DIR="$EXISTING_PATH"
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
    else
        # First time setup
        echo "Downloading OpenFold3 parameters..."
        echo "Please specify the directory to save the parameters (default: $OPENFOLD_CACHE):"
        
        # Read user input with default value
        read -r user_input
        
        # Use user input if provided, otherwise use default
        if [ -z "$user_input" ]; then
            PARAM_DIR="$OPENFOLD_CACHE"
        else
            PARAM_DIR="$user_input"
        fi
    fi

    # Expand tilde to home directory if present
    PARAM_DIR="${PARAM_DIR/#\~/$HOME}"
    
    # Create the directory if it doesn't exist
    mkdir -p "$PARAM_DIR"
    
    # Create the .openfold3 directory if it doesn't exist (for storing the path file)
    mkdir -p "$HOME/.openfold3"
    
    # Save the path to $HOME/.openfold3/param_path.txt
    echo "$PARAM_DIR" > $CKPT_PATH_FILE 
    
    echo "Parameters directory set to: $PARAM_DIR"
    echo "Path saved to: $CKPT_PATH_FILE" 
    
    return 0  # Return zero to proceed with download
}

# Function to perform the download
download_parameters() {
    local PARAM_DIR=$1
    
    echo "Starting parameter download..."
    bash scripts/download_openfold_params.sh --download_dir="$PARAM_DIR"
    
    return $?  # Return the exit code of the download script
}
    

# Main script execution

# Step 0: Setup conda source
setup_conda_commands

# Step 1: Set up OpenFold cache directory
setup_openfold_cache

# Step 1: Set up OpenFold checkpoint directory 
setup_param_directory
SETUP_STATUS=$?

# If setup_param_directory returns non-zero, user chose to use existing params
if [ "$SETUP_STATUS" -eq 0 ]; then
    # 2. Perform download
    download_parameters "$PARAM_DIR"
    if [ $? -ne 0 ]; then
        echo "Download failed. Exiting."
        exit 1
    fi
    echo "Download completed successfully."
fi

# Run tests
pytest -v tests/test_inference_full.py -m inference_verification

echo "Integration tests passed!"