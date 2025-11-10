#!/usr/bin/env python3
"""
Example demonstrating MLX-optimized attention for Apple Silicon.

This script shows how to use the new MLX backend for evoformer attention
that provides equivalent performance to CUDA-based implementations on Apple hardware.

Usage:
    python examples/mlx_attention_example.py
"""

import time
import torch
import numpy as np

# Import OpenFold components
from openfold3.core.model.primitives.attention import Attention
from openfold3.core.model.primitives.attention_mlx import (
    is_mlx_available,
    get_mlx_attention_info,
    mlx_evo_attention
)

def demonstrate_basic_usage():
    """Demonstrate basic MLX attention usage."""
    print("=== Basic MLX Attention Demo ===")

    # Check MLX availability
    if not is_mlx_available():
        print("❌ MLX not available. Please install MLX for Apple Silicon optimization:")
        print("   pip install mlx")
        return

    print("✅ MLX is available!")
    info = get_mlx_attention_info()
    print(f"MLX Info: {info}")

    # Generate sample data (typical OpenFold sizes)
    batch_size = 2
    seq_len = 256  # Protein sequence length
    num_heads = 8
    head_dim = 64

    print(f"\nGenerating test data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")

    # Create query, key, value tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Create bias terms (common in OpenFold attention)
    pair_bias = torch.randn(batch_size, num_heads, seq_len, seq_len) * 0.1
    msa_bias = torch.randn(batch_size, num_heads, seq_len, seq_len) * 0.1

    print(f"\nTensor shapes:")
    print(f"  Q, K, V: {q.shape}")
    print(f"  Biases: {pair_bias.shape}")

    # Compute attention using MLX
    print("\n🚀 Computing attention with MLX...")
    start_time = time.time()
    output = mlx_evo_attention(q, k, v, [pair_bias, msa_bias])
    mlx_time = time.time() - start_time

    print(f"  Output shape: {output.shape}")
    print(f"  Computation time: {mlx_time:.4f} seconds")
    print(f"  Output statistics:")
    print(f"    Mean: {output.mean().item():.6f}")
    print(f"    Std: {output.std().item():.6f}")
    print(f"    Min: {output.min().item():.6f}")
    print(f"    Max: {output.max().item():.6f}")


def demonstrate_attention_module():
    """Demonstrate integration with OpenFold's Attention module."""
    print("\n=== Attention Module Integration Demo ===")

    if not is_mlx_available():
        print("❌ MLX not available, skipping integration demo.")
        return

    # Create an Attention module (similar to OpenFold's usage)
    c_q = c_k = c_v = 256  # Input dimensions
    c_hidden = 64          # Hidden dimension per head
    no_heads = 8           # Number of attention heads

    attention_module = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True  # Use gating like in OpenFold
    )

    print(f"Created Attention module:")
    print(f"  Input dim: {c_q}")
    print(f"  Hidden dim: {c_hidden}")
    print(f"  Heads: {no_heads}")
    print(f"  Gating: enabled")

    # Generate input data
    batch_size = 2
    seq_len = 128
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)

    # Create bias (e.g., positional or pair representation bias)
    bias = torch.randn(batch_size, no_heads, seq_len, seq_len) * 0.1

    print(f"\nInput shapes:")
    print(f"  Query input: {q_x.shape}")
    print(f"  Key/Value input: {kv_x.shape}")
    print(f"  Bias: {bias.shape}")

    # Compare different backends
    backends = [
        ("PyTorch (default)", False, False),
        ("MLX (Apple Silicon)", True, False),
    ]

    results = {}

    for backend_name, use_mlx, use_lma in backends:
        try:
            print(f"\n🔄 Testing {backend_name}...")

            start_time = time.time()
            output = attention_module(
                q_x=q_x,
                kv_x=kv_x,
                biases=[bias],
                use_mlx_attention=use_mlx,
                use_lma=use_lma
            )
            compute_time = time.time() - start_time

            results[backend_name] = {
                'output': output,
                'time': compute_time,
                'shape': output.shape
            }

            print(f"  ✅ Success! Time: {compute_time:.4f}s, Shape: {output.shape}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    # Compare outputs for correctness
    if len(results) >= 2:
        outputs = list(results.values())
        if len(outputs) >= 2:
            diff = torch.max(torch.abs(outputs[0]['output'] - outputs[1]['output']))
            print(f"\n📊 Numerical comparison:")
            print(f"  Max difference between backends: {diff.item():.2e}")
            if diff.item() < 1e-3:
                print("  ✅ Outputs match within tolerance!")
            else:
                print("  ⚠️  Large difference detected")

    # Performance comparison
    if len(results) >= 2:
        times = [(name, result['time']) for name, result in results.items()]
        times.sort(key=lambda x: x[1])
        fastest_name, fastest_time = times[0]

        print(f"\n🏆 Performance ranking:")
        for i, (name, time_val) in enumerate(times):
            speedup = fastest_time / time_val if time_val > 0 else 1.0
            print(f"  {i+1}. {name}: {time_val:.4f}s (speedup: {speedup:.2f}x)")


def demonstrate_chunked_attention():
    """Demonstrate memory-efficient chunked attention for long sequences."""
    print("\n=== Chunked Attention Demo (Long Sequences) ===")

    if not is_mlx_available():
        print("❌ MLX not available, skipping chunked attention demo.")
        return

    # Simulate a very long protein sequence
    batch_size = 1
    seq_len = 2048  # Very long sequence
    num_heads = 8
    head_dim = 64

    print(f"Testing long sequence attention:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Memory requirement: ~{(seq_len * seq_len * batch_size * num_heads * 4 / 1e9):.2f} GB for attention matrix")

    # Generate data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    bias = torch.randn(batch_size, num_heads, seq_len, seq_len) * 0.1

    # Test chunked attention
    chunk_sizes = [256, 512, 1024]

    print(f"\n🧩 Testing chunked attention with different chunk sizes:")

    for chunk_size in chunk_sizes:
        try:
            print(f"  Chunk size {chunk_size}... ", end="")
            start_time = time.time()

            output = mlx_evo_attention(
                q, k, v, [bias],
                use_chunked=True,
                chunk_size=chunk_size
            )

            compute_time = time.time() - start_time
            print(f"✅ {compute_time:.4f}s")

        except Exception as e:
            print(f"❌ Failed: {e}")

    print(f"\n💡 Chunked attention allows processing of very long sequences")
    print(f"   that would otherwise exceed memory limits!")


def main():
    """Main demonstration function."""
    print("🍎 OpenFold 3 MLX Attention Demo for Apple Silicon")
    print("=" * 60)

    # Run all demonstrations
    demonstrate_basic_usage()
    demonstrate_attention_module()
    demonstrate_chunked_attention()

    print("\n" + "=" * 60)
    print("✅ Demo completed!")

    if is_mlx_available():
        print("\n💡 Tips for using MLX attention in OpenFold:")
        print("  1. Set use_mlx_attention=True in attention calls")
        print("  2. Use chunked attention for sequences > 1000 residues")
        print("  3. MLX automatically handles numerical stability")
        print("  4. Performance is optimized for Apple Silicon hardware")
        print("\n📖 For more details, see: openfold3/core/model/primitives/attention_mlx.py")
    else:
        print("\n📦 To use MLX attention, install MLX:")
        print("     pip install mlx")


if __name__ == "__main__":
    main()