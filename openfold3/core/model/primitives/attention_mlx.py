"""
MLX-optimized attention implementation for Apple Silicon.
Replaces CUDA-based evoformer attention with MLX backend.

Copyright 2025 AlQuraishi Laboratory
"""

import math
from typing import List, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import torch


def _torch_to_mlx(tensor: torch.Tensor) -> "mx.array":
    """Convert PyTorch tensor to MLX array."""
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available")
    return mx.array(tensor.detach().cpu().numpy())


def _mlx_to_torch(array: "mx.array", device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor."""
    # Convert MLX array to numpy, then to PyTorch tensor
    import numpy as np
    numpy_array = np.array(array)
    return torch.from_numpy(numpy_array).to(device=device, dtype=dtype)


def _mlx_evoformer_attention(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    bias1: Optional["mx.array"] = None,
    bias2: Optional["mx.array"] = None,
    scale: Optional[float] = None
) -> "mx.array":
    """
    MLX-optimized evoformer attention implementation.

    This implements the same mathematical operations as the CUDA kernel:
    1. Q @ K.T computation with bias terms
    2. Iterative softmax with numerical stability
    3. Attention @ V with proper normalization

    Args:
        q: Query tensor [*, H, Q, C_hidden]
        k: Key tensor [*, H, K, C_hidden]
        v: Value tensor [*, H, V, C_hidden]
        bias1: First bias term that broadcasts to [*, H, Q, K]
        bias2: Second bias term that broadcasts to [*, H, Q, K]
        scale: Attention scale factor (default: 1/sqrt(C_hidden))

    Returns:
        Attention output [*, H, Q, C_hidden]
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available for attention computation")

    # Get dimensions
    *batch_dims, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[-2]

    # Set default scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores: Q @ K.T
    # Use MLX's optimized einsum for better performance on Apple Silicon
    scores = mx.einsum("...qc,...kc->...qk", q, k) * scale

    # Add bias terms if provided
    if bias1 is not None:
        scores = scores + bias1
    if bias2 is not None:
        scores = scores + bias2

    # Apply softmax with numerical stability
    # MLX automatically handles numerical stability in softmax
    attn_weights = mx.softmax(scores, axis=-1)

    # Apply attention to values: attn @ V
    output = mx.einsum("...qk,...kc->...qc", attn_weights, v)

    return output


def _mlx_chunked_attention(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    bias1: Optional["mx.array"] = None,
    bias2: Optional["mx.array"] = None,
    scale: Optional[float] = None,
    chunk_size: int = 1024
) -> "mx.array":
    """
    Memory-efficient chunked attention for very long sequences.

    This implements a chunked version similar to the LMA algorithm
    but optimized for MLX with proper memory management.

    Args:
        q, k, v: Query, key, value tensors
        bias1, bias2: Optional bias terms
        scale: Attention scale factor
        chunk_size: Chunk size for memory efficiency

    Returns:
        Attention output
    """
    seq_len_q = q.shape[-2]
    seq_len_k = k.shape[-2]

    if seq_len_q <= chunk_size and seq_len_k <= chunk_size:
        # No chunking needed for small sequences
        return _mlx_evoformer_attention(q, k, v, bias1, bias2, scale)

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Process in chunks along query dimension and collect outputs
    output_chunks = []

    for q_start in range(0, seq_len_q, chunk_size):
        q_end = min(q_start + chunk_size, seq_len_q)
        q_chunk = q[..., q_start:q_end, :]

        # Get corresponding bias chunks
        bias1_chunk = bias1[..., q_start:q_end, :] if bias1 is not None else None
        bias2_chunk = bias2[..., q_start:q_end, :] if bias2 is not None else None

        # Process this query chunk against all keys
        if seq_len_k <= chunk_size:
            # Keys fit in memory - process normally
            chunk_output = _mlx_evoformer_attention(
                q_chunk, k, v, bias1_chunk, bias2_chunk, scale
            )
        else:
            # Need to chunk both query and key dimensions (LMA-style)
            chunk_output = _mlx_lma_attention(
                q_chunk, k, v, bias1_chunk, bias2_chunk, scale, chunk_size
            )

        output_chunks.append(chunk_output)

    # Concatenate all chunks along the sequence dimension
    return mx.concatenate(output_chunks, axis=-2)


def _mlx_lma_attention(
    q_chunk: "mx.array",
    k: "mx.array",
    v: "mx.array",
    bias1_chunk: Optional["mx.array"],
    bias2_chunk: Optional["mx.array"],
    scale: float,
    kv_chunk_size: int
) -> "mx.array":
    """
    Low-memory attention implementation in MLX.

    Implements the LMA algorithm with proper numerical stability.
    """
    seq_len_k = k.shape[-2]

    # Initialize accumulators
    output = mx.zeros_like(q_chunk)
    normalizer = mx.zeros(q_chunk.shape[:-1] + (1,))
    max_score = mx.full(q_chunk.shape[:-1] + (1,), -float('inf'))

    # Process key-value chunks
    for kv_start in range(0, seq_len_k, kv_chunk_size):
        kv_end = min(kv_start + kv_chunk_size, seq_len_k)
        k_chunk = k[..., kv_start:kv_end, :]
        v_chunk = v[..., kv_start:kv_end, :]

        # Get bias chunks
        b1_chunk = bias1_chunk[..., kv_start:kv_end] if bias1_chunk is not None else None
        b2_chunk = bias2_chunk[..., kv_start:kv_end] if bias2_chunk is not None else None

        # Compute scores for this chunk
        chunk_scores = mx.einsum("...qc,...kc->...qk", q_chunk, k_chunk) * scale

        if b1_chunk is not None:
            chunk_scores = chunk_scores + b1_chunk
        if b2_chunk is not None:
            chunk_scores = chunk_scores + b2_chunk

        # Numerical stability: update running max
        chunk_max = mx.max(chunk_scores, axis=-1, keepdims=True)
        new_max = mx.maximum(max_score, chunk_max)

        # Update previous output and normalizer
        exp_diff_old = mx.exp(max_score - new_max)
        exp_diff_chunk = mx.exp(chunk_max - new_max)

        output = output * exp_diff_old
        normalizer = normalizer * exp_diff_old

        # Process current chunk
        chunk_weights = mx.exp(chunk_scores - new_max)
        chunk_output = mx.einsum("...qk,...kc->...qc", chunk_weights, v_chunk)
        chunk_normalizer = mx.sum(chunk_weights, axis=-1, keepdims=True)

        # Accumulate
        output = output + chunk_output
        normalizer = normalizer + chunk_normalizer
        max_score = new_max

    # Final normalization
    output = output / normalizer

    return output


def mlx_evo_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
    use_chunked: bool = False,
    chunk_size: int = 1024
) -> torch.Tensor:
    """
    PyTorch-compatible MLX evoformer attention function.

    This function provides a drop-in replacement for _deepspeed_evo_attn
    but using MLX for computation on Apple Silicon.

    Args:
        q: Query tensor [*, H, Q, C_hidden]
        k: Key tensor [*, H, K, C_hidden]
        v: Value tensor [*, H, V, C_hidden]
        biases: List of bias tensors (max 2 supported)
        use_chunked: Whether to use memory-efficient chunked attention
        chunk_size: Chunk size for memory efficiency

    Returns:
        Attention output [*, H, Q, C_hidden]
    """
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX not available. Install MLX for Apple Silicon optimization: "
            "pip install mlx"
        )

    if len(biases) > 2:
        raise ValueError("MLX evoformer attention supports at most 2 bias terms")

    # Store original device and dtype
    orig_device = q.device
    orig_dtype = q.dtype

    # Convert to MLX arrays
    q_mlx = _torch_to_mlx(q)
    k_mlx = _torch_to_mlx(k)
    v_mlx = _torch_to_mlx(v)

    # Convert biases
    bias1_mlx = _torch_to_mlx(biases[0]) if len(biases) > 0 else None
    bias2_mlx = _torch_to_mlx(biases[1]) if len(biases) > 1 else None

    # Compute attention using MLX
    if use_chunked:
        output_mlx = _mlx_chunked_attention(
            q_mlx, k_mlx, v_mlx, bias1_mlx, bias2_mlx, chunk_size=chunk_size
        )
    else:
        output_mlx = _mlx_evoformer_attention(
            q_mlx, k_mlx, v_mlx, bias1_mlx, bias2_mlx
        )

    # Convert back to PyTorch
    output = _mlx_to_torch(output_mlx, orig_device, orig_dtype)

    return output


# Compatibility check
def is_mlx_available() -> bool:
    """Check if MLX is available for attention computation."""
    return MLX_AVAILABLE


def get_mlx_attention_info() -> dict:
    """Get information about MLX attention capabilities."""
    if not MLX_AVAILABLE:
        return {"available": False, "reason": "MLX not installed"}

    try:
        import mlx.core as mx
        # Test basic functionality
        test_q = mx.random.normal((1, 4, 32, 64))
        test_k = mx.random.normal((1, 4, 32, 64))
        test_v = mx.random.normal((1, 4, 32, 64))

        _ = _mlx_evoformer_attention(test_q, test_k, test_v)

        return {
            "available": True,
            "device": "Apple Silicon (MLX)",
            "max_sequence_length": "Limited by available memory",
            "supported_dtypes": ["float32", "float16"],
            "features": [
                "Numerical stability",
                "Dual bias support",
                "Memory-efficient chunking",
                "LMA-style attention for long sequences",
                "Triangle attention (cuEquivariance replacement)"
            ]
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"MLX test failed: {str(e)}"
        }


def mlx_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
    scale: float
) -> torch.Tensor:
    """
    MLX-optimized triangle attention implementation.

    This provides a drop-in replacement for cuEquivariance triangle attention
    using MLX for computation on Apple Silicon.

    Args:
        q: Query tensor [*, H, Q, C_hidden]
        k: Key tensor [*, H, K, C_hidden]
        v: Value tensor [*, H, V, C_hidden]
        biases: List containing [mask_bias, triangle_bias]
        scale: Attention scale factor

    Returns:
        Attention output [*, H, Q, C_hidden]
    """
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX not available. Install MLX for Apple Silicon optimization: "
            "pip install mlx"
        )

    if len(biases) != 2:
        raise ValueError("Triangle attention requires exactly 2 bias terms: mask_bias and triangle_bias")

    mask_bias, triangle_bias = biases

    # Handle high-dimensional inputs (like cuEquivariance does)
    is_batched_input = False
    original_shape = None

    if len(q.shape) > 5:
        if len(q.shape) != 6:
            raise ValueError("Max number of dimensions for triangle attention is 6")

        is_batched_input = True
        original_shape = q.shape
        batch, n_tmpl = q.shape[:2]

        # Flatten batch and template dimensions
        q = q.view(batch * n_tmpl, *q.shape[2:])
        k = k.view(batch * n_tmpl, *k.shape[2:])
        v = v.view(batch * n_tmpl, *v.shape[2:])
        mask_bias = mask_bias.view(batch * n_tmpl, *mask_bias.shape[2:])
        triangle_bias = triangle_bias.view(batch * n_tmpl, *triangle_bias.shape[2:])

    # Convert mask from additive format to boolean format (like cuEquivariance does)
    if mask_bias.dtype != torch.bool:
        # Convert -inf masked positions to False, valid positions (0) to True
        mask_bias = mask_bias == 0

    # Convert boolean mask back to additive format for our MLX attention
    # True (valid) -> 0, False (masked) -> -inf
    mask_bias_additive = torch.where(
        mask_bias,
        torch.zeros_like(mask_bias, dtype=q.dtype),
        torch.full_like(mask_bias, -float('inf'), dtype=q.dtype)
    )

    # Call our MLX attention with the processed biases
    processed_biases = [mask_bias_additive, triangle_bias]
    output = mlx_evo_attention(q, k, v, processed_biases)

    # Handle dimension issues (like cuEquivariance does)
    if len(q.shape) == 4 and output.shape[0] == 1:
        # Remove spurious batch dimension if it was added
        output = output.squeeze(0)

    # Restore original batch/template dimensions if needed
    if is_batched_input:
        output = output.view(original_shape[0], original_shape[1], *output.shape[1:])

    # Apply transpose (like cuEquivariance does)
    output = output.transpose(-2, -3)

    return output


# Export the triangle attention function
__all__ = [
    "mlx_evo_attention",
    "is_mlx_available",
    "get_mlx_attention_info",
    "mlx_triangle_attention"
]