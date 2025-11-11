#!/usr/bin/env python3
"""
🍎🧬 IMPROVED Apple Silicon OpenFold Inference Test

Building on our historic success, this test uses improved settings
to generate better quality protein structures.

Improvements:
- Multiple diffusion samples for better convergence
- Proper inference configuration
- Better test sequence (ubiquitin)
- Enhanced generation settings
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

def setup_environment():
    """Set up the environment for MLX testing."""
    print("🔧 Setting up improved Apple Silicon environment...")

    # Disable CUDA if available to force CPU/MPS usage
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Set MLX to use Apple Silicon GPU
    os.environ['MLX_GPU_ENABLED'] = '1'

    # Disable wandb for testing
    os.environ['WANDB_DISABLED'] = 'true'

    print("  ✅ Environment configured for Apple Silicon")


def create_improved_runner_config():
    """Create an improved runner configuration for high-quality MLX inference."""
    return {
        "model_update": {
            "presets": ["predict", "pae_enabled"],  # Remove low_mem for better quality
            "custom": {
                "settings": {
                    "memory": {
                        "eval": {
                            # 🍎 Enable our Apple Silicon optimizations!
                            "use_mlx_attention": True,
                            "use_mlx_triangle_kernels": True,
                            "use_mlx_activation_functions": True,
                            # Disable CUDA optimizations
                            "use_cueq_triangle_kernels": False,
                            "use_deepspeed_evo_attention": False,
                            "use_lma": False
                        }
                    }
                }
            }
        },
        "experiment_settings": {
            "seed": 42,
            "precision": "fp32"  # Keep fp32 for stability
        },
        "data_module_args": {
            "num_workers": 0  # Disable multiprocessing for Apple Silicon compatibility
        }
    }


def test_improved_mlx_inference():
    """Test improved MLX inference with better settings."""
    print("🚀 Testing Improved Apple Silicon Inference...")

    try:
        # Import required modules
        from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
        from openfold3.entry_points.validator import InferenceExperimentConfig
        from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet

        print("  ✅ Successfully imported inference modules")

        # Load the ubiquitin query (better test case)
        query_path = Path("examples/example_inference_inputs/query_ubiquitin.json")
        if not query_path.exists():
            print(f"  ❌ Query file not found: {query_path}")
            print("  📝 Using our test query instead...")
            query_path = Path("test_query_mlx.json")

        query_set = InferenceQuerySet.from_json(query_path)
        print(f"  ✅ Loaded query set with {len(query_set.queries)} queries")

        # Create improved configuration
        config_dict = create_improved_runner_config()
        expt_config = InferenceExperimentConfig(**config_dict)
        print("  ✅ Created improved experiment configuration")

        # Create experiment runner with better settings
        expt_runner = InferenceExperimentRunner(
            expt_config,
            num_diffusion_samples=5,  # 🔥 More samples for better convergence!
            num_model_seeds=3,        # 🔥 More seeds for robustness!
            use_msa_server=True,      # 🔥 Enable MSA for better predictions!
            use_templates=True,       # 🔥 Enable templates!
            output_dir=Path("mlx_inference_improved_output")
        )

        # Fix DataLoader multiprocessing issue for Apple Silicon
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("  ✅ Created improved experiment runner")

        # Setup (this will download/load model if needed)
        print("  🔄 Setting up experiment runner...")
        start_setup = time.time()
        expt_runner.setup()
        setup_time = time.time() - start_setup
        print(f"  ✅ Setup completed in {setup_time:.2f}s")

        # Run improved inference
        print("  🧬 Running improved MLX inference...")
        start_inference = time.time()
        expt_runner.run(query_set)
        inference_time = time.time() - start_inference

        print(f"  🎉 Improved MLX Inference completed in {inference_time:.2f}s!")

        # Cleanup
        expt_runner.cleanup()

        return True

    except Exception as e:
        print(f"  ❌ Improved MLX inference failed: {e}")
        print("  📋 Full traceback:")
        traceback.print_exc()
        return False


def main():
    """Main improved test function."""
    print("🍎🧬 IMPROVED APPLE SILICON OPENFOLD INFERENCE TEST")
    print("🚀 Enhanced settings for better quality protein structures")
    print("=" * 80)

    # Setup environment
    setup_environment()

    # Run the improved test
    success = test_improved_mlx_inference()

    print("\n" + "=" * 80)
    if success:
        print("🎉 IMPROVED INFERENCE COMPLETED!")
        print("✅ Enhanced MLX inference with better settings!")
        print("\n💡 Improvements made:")
        print("  - Multiple diffusion samples (5) for better convergence")
        print("  - Multiple model seeds (3) for robustness")
        print("  - MSA server enabled for better predictions")
        print("  - Templates enabled for improved accuracy")
        print("  - Using known protein (ubiquitin) as test case")
        print("\n🔍 Check the output directory for improved structures!")
    else:
        print("❌ Improved inference encountered issues.")
        print("🔧 This suggests we may need to further tune the settings.")

    return success


if __name__ == "__main__":
    main()