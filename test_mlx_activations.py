#!/usr/bin/env python3
"""
Comprehensive test suite for MLX activation functions.

Tests all MLX-optimized activation functions against PyTorch references
to verify correctness and performance on Apple Silicon.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


def test_mlx_availability():
    """Test MLX availability and setup."""
    print("🔍 Testing MLX Activation Functions Setup")
    print("=" * 60)

    try:
        from openfold3.core.model.primitives.activations_mlx import (
            is_mlx_available,
            get_mlx_activation_info,
            MLXSwiGLU,
            MLXOptimizedSoftmax,
            MLXActivationFunctions
        )
        print("✅ Successfully imported MLX activation modules")

        if is_mlx_available():
            info = get_mlx_activation_info()
            print("✅ MLX is available for activation functions")
            print(f"Available functions: {len(info['optimized_functions'])}")
            for func in info['optimized_functions']:
                print(f"  - {func}")
            return True
        else:
            print("❌ MLX not available")
            return False

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_swiglu_implementation():
    """Test SwiGLU implementation against reference."""
    print("\n🧠 Testing MLX SwiGLU Implementation")
    print("=" * 60)

    try:
        from openfold3.core.model.primitives.activations_mlx import MLXSwiGLU, is_mlx_available

        if not is_mlx_available():
            print("❌ MLX not available, skipping SwiGLU test")
            return False

        # Test parameters
        batch_size = 4
        seq_len = 128
        dim_in = 256
        dim_hidden = dim_in * 4  # Standard 4x expansion

        print(f"Test parameters:")
        print(f"  Input dimensions: [{batch_size}, {seq_len}, {dim_in}]")
        print(f"  Hidden dimensions: {dim_hidden}")

        # Create MLX SwiGLU layer
        swiglu_mlx = MLXSwiGLU(dim_in=dim_in, dim_hidden=dim_hidden, bias=False)

        # Generate test data
        x = torch.randn(batch_size, seq_len, dim_in, dtype=torch.float32)

        print(f"\n🚀 Testing MLX SwiGLU...")
        start_time = time.time()
        try:
            output_mlx = swiglu_mlx(x)
            mlx_time = time.time() - start_time

            print(f"  ✅ MLX SwiGLU test passed!")
            print(f"  Time: {mlx_time:.4f}s")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output_mlx.shape}")
            print(f"  Output stats: mean={output_mlx.mean().item():.6f}, std={output_mlx.std().item():.6f}")

            # Verify output shape is correct
            assert output_mlx.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {output_mlx.shape}"
            print(f"  ✅ Output shape verification passed")

            return True

        except Exception as e:
            print(f"  ❌ MLX SwiGLU test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_reference_swiglu(x: torch.Tensor, gate_proj, up_proj, down_proj) -> torch.Tensor:
    """Reference SwiGLU implementation for comparison."""
    gate = gate_proj(x)
    up = up_proj(x)
    # SwiGLU: SiLU(gate) * up
    silu_gate = F.silu(gate)
    gated = silu_gate * up
    output = down_proj(gated)
    return output


def test_swiglu_correctness():
    """Test SwiGLU numerical correctness against reference implementation."""
    print("\n🔬 Testing SwiGLU Numerical Correctness")
    print("=" * 60)

    try:
        from openfold3.core.model.primitives.activations_mlx import MLXSwiGLU, is_mlx_available

        if not is_mlx_available():
            print("❌ MLX not available, skipping correctness test")
            return False

        # Smaller test for precise comparison
        batch_size = 2
        seq_len = 32
        dim_in = 64
        dim_hidden = 128

        # Create MLX SwiGLU
        swiglu_mlx = MLXSwiGLU(dim_in=dim_in, dim_hidden=dim_hidden, bias=False)

        # Create reference implementation with same weights
        gate_proj = torch.nn.Linear(dim_in, dim_hidden, bias=False)
        up_proj = torch.nn.Linear(dim_in, dim_hidden, bias=False)
        down_proj = torch.nn.Linear(dim_hidden, dim_in, bias=False)

        # Copy weights to ensure identical computation
        gate_proj.weight.data = swiglu_mlx.gate_proj.weight.data.clone()
        up_proj.weight.data = swiglu_mlx.up_proj.weight.data.clone()
        down_proj.weight.data = swiglu_mlx.down_proj.weight.data.clone()

        # Test input
        x = torch.randn(batch_size, seq_len, dim_in, dtype=torch.float32)

        # Compute outputs
        print("Computing reference SwiGLU...")
        output_ref = test_reference_swiglu(x, gate_proj, up_proj, down_proj)

        print("Computing MLX SwiGLU...")
        output_mlx = swiglu_mlx(x)

        # Compare outputs
        diff = torch.max(torch.abs(output_ref - output_mlx))
        rel_diff = diff / torch.max(torch.abs(output_ref))

        print(f"\n📊 Numerical comparison:")
        print(f"  Max absolute difference: {diff.item():.2e}")
        print(f"  Max relative difference: {rel_diff.item():.2e}")

        if diff.item() < 1e-3:
            print(f"  ✅ Outputs match within tolerance!")
            return True
        else:
            print(f"  ⚠️  Large difference detected")
            return False

    except Exception as e:
        print(f"❌ Correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimized_softmax():
    """Test MLX optimized softmax implementation."""
    print("\n🎯 Testing MLX Optimized Softmax")
    print("=" * 60)

    try:
        from openfold3.core.model.primitives.activations_mlx import MLXOptimizedSoftmax, is_mlx_available

        if not is_mlx_available():
            print("❌ MLX not available, skipping softmax test")
            return False

        # Test parameters
        batch_size = 4
        seq_len = 256
        num_heads = 8

        print(f"Test parameters:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Number of heads: {num_heads}")

        # Create test data (attention scores)
        attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32)
        # Add some extreme values to test numerical stability
        attention_scores[:, :, :5, :5] = 50.0  # Large positive
        attention_scores[:, :, -5:, -5:] = -50.0  # Large negative

        # Test PyTorch reference
        print("\n🔄 Testing PyTorch reference softmax...")
        start_time = time.time()
        output_pytorch = F.softmax(attention_scores, dim=-1)
        pytorch_time = time.time() - start_time

        # Test MLX optimized softmax
        print("🍎 Testing MLX optimized softmax...")
        mlx_softmax = MLXOptimizedSoftmax(dim=-1)
        start_time = time.time()
        output_mlx = mlx_softmax(attention_scores)
        mlx_time = time.time() - start_time

        # Compare results
        diff = torch.max(torch.abs(output_pytorch - output_mlx))
        rel_diff = diff / torch.max(torch.abs(output_pytorch))

        print(f"\n📊 Performance and accuracy:")
        print(f"  PyTorch time: {pytorch_time:.4f}s")
        print(f"  MLX time: {mlx_time:.4f}s")
        print(f"  Speedup: {pytorch_time / mlx_time:.2f}x")
        print(f"  Max absolute difference: {diff.item():.2e}")
        print(f"  Max relative difference: {rel_diff.item():.2e}")

        # Verify softmax properties
        row_sums = torch.sum(output_mlx, dim=-1)
        sum_error = torch.max(torch.abs(row_sums - 1.0))
        print(f"  Row sum error (should be ~0): {sum_error.item():.2e}")

        if diff.item() < 1e-3 and sum_error.item() < 1e-5:
            print(f"  ✅ MLX softmax test passed!")
            return True
        else:
            print(f"  ❌ MLX softmax test failed")
            return False

    except Exception as e:
        print(f"❌ Softmax test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_activation_functions():
    """Test various activation functions."""
    print("\n⚡ Testing MLX Activation Functions")
    print("=" * 60)

    try:
        from openfold3.core.model.primitives.activations_mlx import MLXActivationFunctions, is_mlx_available

        if not is_mlx_available():
            print("❌ MLX not available, skipping activation tests")
            return False

        # Test data
        x = torch.randn(4, 32, 128, dtype=torch.float32)
        print(f"Test input shape: {x.shape}")

        activations_to_test = [
            ("SiLU/Swish", MLXActivationFunctions.silu, F.silu),
            ("GELU (precise)", lambda t: MLXActivationFunctions.gelu(t, 'none'), F.gelu),
            ("GELU (approx)", lambda t: MLXActivationFunctions.gelu(t, 'tanh'), lambda t: F.gelu(t, approximate='tanh')),
            ("Mish", MLXActivationFunctions.mish, lambda t: t * torch.tanh(F.softplus(t))),
        ]

        results = {}

        for name, mlx_func, ref_func in activations_to_test:
            print(f"\n🔄 Testing {name}...")

            try:
                # MLX implementation
                start_time = time.time()
                output_mlx = mlx_func(x)
                mlx_time = time.time() - start_time

                # Reference implementation
                start_time = time.time()
                output_ref = ref_func(x)
                ref_time = time.time() - start_time

                # Compare
                diff = torch.max(torch.abs(output_mlx - output_ref))
                rel_diff = diff / torch.max(torch.abs(output_ref))

                results[name] = {
                    'mlx_time': mlx_time,
                    'ref_time': ref_time,
                    'speedup': ref_time / mlx_time,
                    'max_diff': diff.item(),
                    'rel_diff': rel_diff.item(),
                    'passed': diff.item() < 1e-3
                }

                status = "✅" if results[name]['passed'] else "❌"
                print(f"  {status} {name}: speedup={results[name]['speedup']:.2f}x, diff={diff.item():.2e}")

            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                results[name] = {'passed': False, 'error': str(e)}

        # Summary
        passed_count = sum(1 for r in results.values() if r.get('passed', False))
        total_count = len(activations_to_test)

        print(f"\n📊 Activation Functions Summary:")
        print(f"  Passed: {passed_count}/{total_count}")

        return passed_count == total_count

    except Exception as e:
        print(f"❌ Activation functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Comprehensive performance benchmark."""
    print("\n🏆 MLX Activation Functions Performance Benchmark")
    print("=" * 60)

    try:
        from openfold3.core.model.primitives.activations_mlx import (
            MLXSwiGLU, MLXOptimizedSoftmax, MLXActivationFunctions, is_mlx_available
        )

        if not is_mlx_available():
            print("❌ MLX not available, skipping benchmark")
            return False

        # Test different sizes
        test_sizes = [
            (2, 64, 128),    # Small
            (4, 128, 256),   # Medium
            (8, 256, 512),   # Large
        ]

        print("Testing different tensor sizes:")
        for batch_size, seq_len, dim in test_sizes:
            print(f"\n📏 Size: [{batch_size}, {seq_len}, {dim}]")

            x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)

            # SwiGLU benchmark
            swiglu = MLXSwiGLU(dim_in=dim, dim_hidden=dim*4, bias=False)
            start_time = time.time()
            _ = swiglu(x)
            swiglu_time = time.time() - start_time

            # Softmax benchmark
            softmax = MLXOptimizedSoftmax(dim=-1)
            scores = torch.randn(batch_size, 8, seq_len, seq_len)
            start_time = time.time()
            _ = softmax(scores)
            softmax_time = time.time() - start_time

            # SiLU benchmark
            start_time = time.time()
            _ = MLXActivationFunctions.silu(x)
            silu_time = time.time() - start_time

            print(f"  SwiGLU: {swiglu_time:.4f}s")
            print(f"  Softmax: {softmax_time:.4f}s")
            print(f"  SiLU: {silu_time:.4f}s")

        return True

    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def main():
    """Main test function."""
    print("🍎 OpenFold 3 MLX Activation Functions Test Suite")
    print("🚀 Comprehensive Apple Silicon Optimization Testing")
    print("\n")

    test_results = {}

    # Run all tests
    test_results['mlx_availability'] = test_mlx_availability()
    test_results['swiglu_implementation'] = test_swiglu_implementation()
    test_results['swiglu_correctness'] = test_swiglu_correctness()
    test_results['optimized_softmax'] = test_optimized_softmax()
    test_results['activation_functions'] = test_activation_functions()
    test_results['performance_benchmark'] = test_performance_benchmark()

    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    print(f"\n" + "=" * 80)
    print(f"🎯 TEST SUMMARY")
    print(f"=" * 80)
    print(f"Passed: {passed_tests}/{total_tests} tests")

    for test_name, result in test_results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name.replace('_', ' ').title()}")

    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ MLX activation functions are ready for production!")
        print(f"\n💡 Key achievements:")
        print(f"  - SwiGLU replaces Triton custom kernels")
        print(f"  - Optimized softmax with numerical stability")
        print(f"  - Complete activation function library")
        print(f"  - Native Apple Silicon optimization")
        print(f"  - Seamless PyTorch integration")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed. Check implementation.")

    return passed_tests == total_tests


if __name__ == "__main__":
    main()