#!/usr/bin/env python3
"""
🔍 MLX Quality Diagnosis

Quick test to understand what settings produce the best structures
vs. computation time on Apple Silicon.
"""

import os
import time
from pathlib import Path

def setup_mlx_env():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MLX_GPU_ENABLED'] = '1'
    os.environ['WANDB_DISABLED'] = 'true'

def analyze_current_output():
    """Analyze our current output to understand the issues."""
    print("🔍 DIAGNOSING CURRENT OUTPUT QUALITY")
    print("=" * 60)

    confidence_file = Path("mlx_inference_test_output/test_peptide_mlx/seed_2746317213/test_peptide_mlx_seed_2746317213_sample_1_confidences_aggregated.json")

    if confidence_file.exists():
        import json
        with open(confidence_file) as f:
            conf = json.load(f)

        print(f"📊 Current Quality Metrics:")
        print(f"  Average pLDDT: {conf['avg_plddt']:.2f} (target: >70 for good quality)")
        print(f"  PTM Score: {conf['ptm']:.3f} (target: >0.5 for confident prediction)")
        print(f"  Disorder: {conf['disorder']:.1f} (1.0 = completely disordered)")
        print(f"  Has Clash: {conf['has_clash']:.1f} (0.0 = no clashes, good)")

        print(f"\n🎯 Quality Assessment:")
        if conf['avg_plddt'] < 50:
            print(f"  ❌ Very low confidence - likely convergence failure")
        elif conf['avg_plddt'] < 70:
            print(f"  ⚠️  Low confidence - needs improvement")
        else:
            print(f"  ✅ Good confidence")

        if conf['disorder'] > 0.8:
            print(f"  ❌ Highly disordered structure")
        elif conf['disorder'] > 0.5:
            print(f"  ⚠️  Moderately disordered")
        else:
            print(f"  ✅ Well-structured")

    print(f"\n🧬 Structure Analysis:")
    print(f"  The linear backbone suggests:")
    print(f"  1. Insufficient diffusion sampling (only 1 sample used)")
    print(f"  2. Missing evolutionary constraints (no MSA)")
    print(f"  3. Missing structural templates")
    print(f"  4. Possible MLX numerical precision differences")

def recommend_settings():
    """Recommend improved settings."""
    print(f"\n🛠 RECOMMENDED IMPROVEMENTS:")
    print("=" * 60)

    print("📈 Progressive Improvement Strategy:")
    print()

    print("1. 🟡 QUICK TEST (2-3 minutes):")
    print("   - Use ubiquitin (known folder)")
    print("   - 3 diffusion samples")
    print("   - 1 model seed")
    print("   - MSA: disabled (for speed)")
    print("   - Templates: disabled")
    print()

    print("2. 🟠 MEDIUM TEST (5-10 minutes):")
    print("   - Same protein")
    print("   - 5 diffusion samples")
    print("   - 2 model seeds")
    print("   - MSA: enabled")
    print("   - Templates: disabled")
    print()

    print("3. 🟢 FULL TEST (15-20 minutes):")
    print("   - Same protein")
    print("   - 10 diffusion samples")
    print("   - 3 model seeds")
    print("   - MSA: enabled")
    print("   - Templates: enabled")
    print()

    print("💡 Expected Quality Progression:")
    print("   Quick: pLDDT ~40-50 (minimal folding)")
    print("   Medium: pLDDT ~60-70 (reasonable structure)")
    print("   Full: pLDDT >80 (high quality)")

def create_quick_test():
    """Create a quick 2-minute test to verify our hypothesis."""
    print(f"\n🚀 CREATING QUICK QUALITY TEST")
    print("=" * 60)

    test_config = {
        "protein": "ubiquitin (76 residues)",
        "diffusion_samples": 3,
        "model_seeds": 1,
        "msa_enabled": False,
        "templates_enabled": False,
        "expected_time": "2-3 minutes",
        "expected_plddt": "40-60"
    }

    print("📋 Quick Test Configuration:")
    for key, value in test_config.items():
        print(f"   {key}: {value}")

    print(f"\n🎯 This test will help us confirm:")
    print(f"   ✅ MLX infrastructure is working (we know this)")
    print(f"   ✅ Better sampling improves quality")
    print(f"   ✅ Ubiquitin folds better than our test sequence")
    print(f"   ✅ Multiple samples reduce linear backbone artifacts")

def main():
    setup_mlx_env()

    print("🍎🔬 MLX STRUCTURE QUALITY DIAGNOSIS")
    print("Understanding why our first structure was linear")
    print("\n")

    analyze_current_output()
    recommend_settings()
    create_quick_test()

    print(f"\n" + "=" * 80)
    print("🎯 NEXT ACTION: Run the quick test with ubiquitin + 3 samples")
    print("📊 Expected: Better structure quality with minimal time cost")
    print("🍎 This will prove our Apple Silicon implementation quality")

if __name__ == "__main__":
    main()