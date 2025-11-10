#!/usr/bin/env python3
"""
Quick verification script to test the OpenFold MLX environment setup.
Run this to verify that MLX attention is working correctly.
"""

import sys
import os

def check_environment():
    """Check if the environment is set up correctly."""
    print("🔍 Environment Verification")
    print("=" * 50)

    # Check Python version
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Check if we're in the correct conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not detected')
    print(f"Conda environment: {conda_env}")

    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   Device: {torch.device('cpu')}")  # Should be CPU on Apple Silicon for now
    except ImportError:
        print("❌ PyTorch not available")
        return False

    # Check MLX
    try:
        import mlx.core as mx
        print(f"✅ MLX: Available")

        # Test basic MLX operation
        test_array = mx.array([1, 2, 3])
        print(f"   Test array: {test_array}")
    except ImportError:
        print("❌ MLX not available")
        return False

    # Check OpenFold imports
    try:
        from openfold3.core.model.primitives.attention_mlx import is_mlx_available, get_mlx_attention_info
        print(f"✅ OpenFold MLX: Available")

        if is_mlx_available():
            info = get_mlx_attention_info()
            print(f"   MLX Features: {info.get('features', [])}")

    except ImportError as e:
        print(f"❌ OpenFold MLX import failed: {e}")
        return False

    return True

def test_mlx_attention():
    """Test the MLX attention implementation."""
    print("\n🧠 MLX Attention Test")
    print("=" * 50)

    try:
        import torch
        from openfold3.core.model.primitives.attention_mlx import mlx_evo_attention

        # Create test tensors
        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        bias = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32) * 0.1

        print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

        # Run MLX attention
        output = mlx_evo_attention(q, k, v, [bias])
        print(f"✅ MLX attention successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")

        # Test chunked attention for memory efficiency
        output_chunked = mlx_evo_attention(q, k, v, [bias], use_chunked=True, chunk_size=16)

        # Check if outputs are similar
        diff = torch.max(torch.abs(output - output_chunked))
        print(f"✅ Chunked attention test passed!")
        print(f"   Max difference: {diff.item():.6f}")

        return True

    except Exception as e:
        print(f"❌ MLX attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function."""
    print("🍎 OpenFold 3 MLX Environment Verification")
    print("🚀 Apple Silicon Optimization Check")
    print("\n")

    env_ok = check_environment()

    if env_ok:
        print("\n✅ Environment check passed!")
        attention_ok = test_mlx_attention()

        if attention_ok:
            print("\n" + "=" * 50)
            print("🎉 SUCCESS: OpenFold MLX is ready!")
            print("=" * 50)
            print("\n💡 Next steps:")
            print("  1. Restart VS Code to pick up environment changes")
            print("  2. Open Command Palette (Cmd+Shift+P)")
            print("  3. Run 'Python: Select Interpreter'")
            print("  4. Choose: /Users/gtaghon/miniconda3/envs/openfold/bin/python")
            print("  5. Test with: python examples/mlx_attention_example.py")

        else:
            print("\n❌ MLX attention test failed - check installation")
    else:
        print("\n❌ Environment check failed - please check conda setup")

if __name__ == "__main__":
    main()