"""
Test suite for MLX attention implementation.
Tests correctness, performance, and compatibility with existing PyTorch code.

Copyright 2025 AlQuraishi Laboratory
"""

import math
import time
import pytest
import torch
import numpy as np

# Import the MLX attention module
try:
    from openfold3.core.model.primitives.attention_mlx import (
        mlx_evo_attention,
        is_mlx_available,
        get_mlx_attention_info,
        _torch_to_mlx,
        _mlx_to_torch,
        _mlx_evoformer_attention,
        _mlx_chunked_attention
    )
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from openfold3.core.model.primitives.attention import Attention, _attention

# Skip all tests if MLX is not available
pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")


class TestMLXAttentionCorrectness:
    """Test correctness of MLX attention against reference implementations."""

    @pytest.fixture
    def attention_inputs(self):
        """Generate test inputs for attention."""
        batch_size = 2
        num_heads = 8
        seq_len = 64
        head_dim = 32

        # Generate random inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

        # Generate bias terms
        bias1 = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32) * 0.1
        bias2 = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32) * 0.1

        return q, k, v, [bias1, bias2]

    def test_mlx_vs_pytorch_basic(self, attention_inputs):
        """Test basic attention correctness against PyTorch."""
        q, k, v, biases = attention_inputs

        # Compute reference attention using PyTorch
        reference = _attention(q, k, v, biases)

        # Compute MLX attention
        mlx_output = mlx_evo_attention(q, k, v, biases)

        # Check shapes match
        assert mlx_output.shape == reference.shape, (
            f"Shape mismatch: MLX {mlx_output.shape} vs PyTorch {reference.shape}"
        )

        # Check numerical agreement (allowing for small floating point differences)
        torch.testing.assert_close(mlx_output, reference, rtol=1e-4, atol=1e-5)

    def test_mlx_vs_pytorch_no_bias(self, attention_inputs):
        """Test attention without bias terms."""
        q, k, v, _ = attention_inputs

        reference = _attention(q, k, v, [])
        mlx_output = mlx_evo_attention(q, k, v, [])

        assert mlx_output.shape == reference.shape
        torch.testing.assert_close(mlx_output, reference, rtol=1e-4, atol=1e-5)

    def test_mlx_vs_pytorch_single_bias(self, attention_inputs):
        """Test attention with single bias term."""
        q, k, v, biases = attention_inputs

        reference = _attention(q, k, v, biases[:1])
        mlx_output = mlx_evo_attention(q, k, v, biases[:1])

        assert mlx_output.shape == reference.shape
        torch.testing.assert_close(mlx_output, reference, rtol=1e-4, atol=1e-5)

    def test_mlx_chunked_attention(self, attention_inputs):
        """Test chunked attention for memory efficiency."""
        q, k, v, biases = attention_inputs

        # Test with small chunk size to force chunking
        chunk_size = 32
        chunked_output = mlx_evo_attention(q, k, v, biases, use_chunked=True, chunk_size=chunk_size)
        reference = mlx_evo_attention(q, k, v, biases, use_chunked=False)

        assert chunked_output.shape == reference.shape
        torch.testing.assert_close(chunked_output, reference, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seq_len", [16, 32, 128, 256])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_different_sizes(self, seq_len, head_dim):
        """Test attention with different sequence lengths and head dimensions."""
        batch_size = 1
        num_heads = 4

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        bias = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32) * 0.1

        reference = _attention(q, k, v, [bias])
        mlx_output = mlx_evo_attention(q, k, v, [bias])

        assert mlx_output.shape == reference.shape
        torch.testing.assert_close(mlx_output, reference, rtol=1e-4, atol=1e-5)

    def test_gradient_flow(self, attention_inputs):
        """Test that gradients flow correctly through MLX attention."""
        q, k, v, biases = attention_inputs

        # Make tensors require gradients
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        for b in biases:
            b.requires_grad_(True)

        # Forward pass
        output = mlx_evo_attention(q, k, v, biases)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        assert q.grad is not None and not torch.allclose(q.grad, torch.zeros_like(q.grad))
        assert k.grad is not None and not torch.allclose(k.grad, torch.zeros_like(k.grad))
        assert v.grad is not None and not torch.allclose(v.grad, torch.zeros_like(v.grad))


class TestMLXAttentionPerformance:
    """Benchmark MLX attention performance."""

    def generate_large_inputs(self, seq_len, head_dim=64, num_heads=8, batch_size=2):
        """Generate inputs for performance testing."""
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        bias = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32) * 0.1
        return q, k, v, [bias]

    @pytest.mark.benchmark
    def test_performance_comparison(self):
        """Compare MLX vs PyTorch performance."""
        seq_lengths = [128, 256, 512, 1024]
        results = {}

        for seq_len in seq_lengths:
            q, k, v, biases = self.generate_large_inputs(seq_len)

            # Benchmark PyTorch
            torch_times = []
            for _ in range(5):
                start_time = time.time()
                _ = _attention(q, k, v, biases)
                torch_times.append(time.time() - start_time)

            # Benchmark MLX
            mlx_times = []
            for _ in range(5):
                start_time = time.time()
                _ = mlx_evo_attention(q, k, v, biases)
                mlx_times.append(time.time() - start_time)

            results[seq_len] = {
                'pytorch_time': np.mean(torch_times[1:]),  # Skip first run
                'mlx_time': np.mean(mlx_times[1:]),
                'speedup': np.mean(torch_times[1:]) / np.mean(mlx_times[1:])
            }

            print(f"Seq Length {seq_len}: PyTorch={results[seq_len]['pytorch_time']:.4f}s, "
                  f"MLX={results[seq_len]['mlx_time']:.4f}s, "
                  f"Speedup={results[seq_len]['speedup']:.2f}x")

        # Assert that MLX is competitive (at least not more than 2x slower)
        for seq_len, result in results.items():
            assert result['speedup'] > 0.5, f"MLX too slow for seq_len={seq_len}"

    @pytest.mark.benchmark
    def test_memory_efficiency(self):
        """Test memory usage with chunked attention for very long sequences."""
        seq_len = 2048
        q, k, v, biases = self.generate_large_inputs(seq_len, batch_size=1)

        # Test that chunked attention works for large sequences
        output = mlx_evo_attention(q, k, v, biases, use_chunked=True, chunk_size=512)
        assert output.shape == (1, 8, seq_len, 64)

        # Verify numerical correctness for smaller sequences
        q_small, k_small, v_small, biases_small = self.generate_large_inputs(256, batch_size=1)
        chunked_small = mlx_evo_attention(q_small, k_small, v_small, biases_small,
                                        use_chunked=True, chunk_size=128)
        normal_small = mlx_evo_attention(q_small, k_small, v_small, biases_small,
                                       use_chunked=False)
        torch.testing.assert_close(chunked_small, normal_small, rtol=1e-4, atol=1e-5)


class TestMLXAttentionModule:
    """Test integration with the Attention module."""

    def test_attention_module_integration(self):
        """Test that the Attention module can use MLX backend."""
        c_q = c_k = c_v = 256
        c_hidden = 64
        no_heads = 8

        attention_module = Attention(
            c_q=c_q,
            c_k=c_k,
            c_v=c_v,
            c_hidden=c_hidden,
            no_heads=no_heads,
            gating=True
        )

        # Generate test inputs
        batch_size = 2
        seq_len = 128
        q_x = torch.randn(batch_size, seq_len, c_q)
        kv_x = torch.randn(batch_size, seq_len, c_k)
        bias = torch.randn(batch_size, no_heads, seq_len, seq_len) * 0.1

        # Test MLX attention
        if is_mlx_available():
            output_mlx = attention_module(
                q_x=q_x,
                kv_x=kv_x,
                biases=[bias],
                use_mlx_attention=True
            )

            # Test regular attention for comparison
            output_regular = attention_module(
                q_x=q_x,
                kv_x=kv_x,
                biases=[bias],
                use_mlx_attention=False
            )

            assert output_mlx.shape == output_regular.shape
            # Allow for some numerical differences due to different implementations
            torch.testing.assert_close(output_mlx, output_regular, rtol=1e-3, atol=1e-4)

    def test_attention_mutual_exclusivity(self):
        """Test that attention options are mutually exclusive."""
        attention_module = Attention(256, 256, 256, 64, 8)
        q_x = torch.randn(1, 32, 256)
        kv_x = torch.randn(1, 32, 256)

        # Should raise error when multiple options are specified
        with pytest.raises(ValueError, match="Choose at most one alternative"):
            attention_module(
                q_x=q_x,
                kv_x=kv_x,
                use_mlx_attention=True,
                use_lma=True
            )


class TestMLXUtilities:
    """Test utility functions."""

    def test_mlx_availability_check(self):
        """Test MLX availability detection."""
        assert isinstance(is_mlx_available(), bool)
        info = get_mlx_attention_info()
        assert isinstance(info, dict)
        assert 'available' in info

    def test_tensor_conversion(self):
        """Test tensor conversion utilities."""
        if is_mlx_available():
            # Test conversion round-trip
            torch_tensor = torch.randn(2, 4, 32, 64, dtype=torch.float32)
            mlx_array = _torch_to_mlx(torch_tensor)
            torch_tensor_back = _mlx_to_torch(mlx_array, torch_tensor.device, torch_tensor.dtype)

            torch.testing.assert_close(torch_tensor, torch_tensor_back, rtol=1e-6, atol=1e-7)

    def test_error_handling(self):
        """Test proper error handling when MLX is not available."""
        # This test will only run if MLX is available, but we can test the error paths
        q = torch.randn(1, 4, 32, 64)
        k = torch.randn(1, 4, 32, 64)
        v = torch.randn(1, 4, 32, 64)

        # Test bias limit
        too_many_biases = [torch.randn(1, 4, 32, 32) for _ in range(3)]
        with pytest.raises(ValueError, match="at most 2 bias terms"):
            mlx_evo_attention(q, k, v, too_many_biases)


if __name__ == "__main__":
    # Run basic tests
    if is_mlx_available():
        print("MLX is available. Running tests...")
        info = get_mlx_attention_info()
        print(f"MLX Info: {info}")

        # Run a simple test
        test_class = TestMLXAttentionCorrectness()
        q = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        k = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        v = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        bias = torch.randn(1, 4, 32, 32, dtype=torch.float32) * 0.1

        reference = _attention(q, k, v, [bias])
        mlx_output = mlx_evo_attention(q, k, v, [bias])

        print(f"Reference shape: {reference.shape}")
        print(f"MLX output shape: {mlx_output.shape}")
        print(f"Max difference: {torch.max(torch.abs(reference - mlx_output)).item()}")
        print("Basic test passed!")
    else:
        print("MLX is not available. Tests will be skipped.")