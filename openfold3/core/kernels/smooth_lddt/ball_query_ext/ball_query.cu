/*
 * CUDA ball-query kernels for OpenFold3 smooth lDDT.
 *
 * Modified from PyTorch3D (Meta Platforms, Inc.) by
 * Liang Hong <lhong22@cse.cuhk.edu.hk>:
 *   - Sequential ball-query kernel adapted from
 *     https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query.cu
 *   - Backward kernel (BallQueryPredBackwardKernel) adapted from the
 *     atomicAdd-scatter pattern in
 *     https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/knn/knn.cu
 *     (KNearestNeighborBackwardKernel)
 *
 * Modifications added for OpenFold3:
 *   - Warp-cooperative variant with reservoir sampling for unbiased
 *     random neighbor selection (BallQueryKernelCoop, W=8 lanes/atom)
 *   - Coop variant that also emits per-pair predicted squared distances
 *     (BallQueryKernelCoopWithPred), eliminating the Python-side x_j gather
 *   - Dedicated backward (BallQueryPredBackwardKernel) that reloads
 *     positions from saved (pred, idx) and never materializes [B,N,K,3]
 *
 * The portions adapted from PyTorch3D remain under their original license:
 *   Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 *   BSD-3-Clause; see
 *   https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE.
 *
 * The OpenFold3 modifications are released under Apache-2.0 alongside the
 * rest of this repository (see openfold3/core/kernels/smooth_lddt/__init__.py).
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <math.h>

// Fast hash-based RNG for reservoir sampling (deterministic per seed+position)
__device__ __forceinline__ uint32_t hash_rng(uint32_t state) {
  state ^= state >> 16;
  state *= 0x45d9f3bu;
  state ^= state >> 16;
  state *= 0x45d9f3bu;
  state ^= state >> 16;
  return state;
}

// Original sequential kernel (fallback for small K)
template <typename scalar_t>
__global__ void BallQueryKernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> p1,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> p2,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths1,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths2,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> dists,
    const int64_t K,
    const float radius,
    const float radius2,
    const bool skip_points_outside_cube) {
  const int64_t N = p1.size(0);
  const int64_t chunks_per_cloud = (1 + (p1.size(1) - 1) / blockDim.x);
  const int64_t chunks_to_do = N * chunks_per_cloud;
  const int D = p1.size(2);

  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t n = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    const int64_t i = start_point + threadIdx.x;

    if (i >= lengths1[n]) {
      continue;
    }

    int64_t count = 0;
    for (int64_t j = 0; j < lengths2[n] && count < K; ++j) {
      if (skip_points_outside_cube) {
        bool is_within_radius = true;
        for (int d = 0; is_within_radius && d < D; ++d) {
          scalar_t abs_diff = fabs(p1[n][i][d] - p2[n][j][d]);
          is_within_radius = (abs_diff <= radius);
        }
        if (!is_within_radius) {
          continue;
        }
      }

      scalar_t dist2 = 0.0;
      for (int d = 0; d < D; ++d) {
        scalar_t diff = p1[n][i][d] - p2[n][j][d];
        dist2 += (diff * diff);
      }

      if (dist2 < radius2) {
        idxs[n][i][count] = j;
        dists[n][i][count] = dist2;
        ++count;
      }
    }
  }
}

// Warp-cooperative kernel: W threads per query atom, strided scan,
// reservoir sampling for unbiased random K selection.
// Phase 1 (fill): threads claim slots via shared atomic counter.
// Phase 2 (reservoir): once K slots are full, each thread independently
// replaces random slots. Concurrent writes are benign (training loss is
// robust to the rare torn idx/dist pair from two threads hitting the same slot).
template <typename scalar_t, int W>
__global__ void BallQueryKernelCoop(
    const scalar_t* __restrict__ p1_ptr,
    const scalar_t* __restrict__ p2_ptr,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    int64_t* __restrict__ idxs_ptr,
    scalar_t* __restrict__ dists_ptr,
    const int64_t N,
    const int64_t P1,
    const int64_t K,
    const float radius,
    const float radius2,
    const uint32_t seed) {

  const int atoms_per_block = blockDim.x / W;
  const int group_id = threadIdx.x / W;
  const int lane = threadIdx.x % W;
  const int64_t global_atom_flat = static_cast<int64_t>(blockIdx.x) * atoms_per_block + group_id;

  const int64_t total_atoms = N * P1;
  if (global_atom_flat >= total_atoms) return;

  const int64_t n = global_atom_flat / P1;
  const int64_t i = global_atom_flat % P1;

  if (i >= lengths1[n]) return;

  // Shared memory: one counter per query atom in this block
  extern __shared__ int smem_counters[];
  int* my_counter = &smem_counters[group_id];
  if (lane == 0) *my_counter = 0;
  __syncwarp(__activemask());

  // Output base for this query atom
  const int64_t out_base = (n * P1 + i) * K;
  int64_t* out_idx = idxs_ptr + out_base;
  scalar_t* out_dist = dists_ptr + out_base;

  // Query point coordinates
  const scalar_t* p1_ni = p1_ptr + (n * P1 + i) * 3;
  const scalar_t* p2_n = p2_ptr + n * P1 * 3;
  const scalar_t qi_x = p1_ni[0];
  const scalar_t qi_y = p1_ni[1];
  const scalar_t qi_z = p1_ni[2];

  // Per-thread reservoir sampling state
  int seen = 0;
  uint32_t rng = seed ^ static_cast<uint32_t>(
      n * 1000003u + i * 997u + lane * 31u);
  rng = hash_rng(rng);

  const int64_t len2 = lengths2[n];
  const int K_int = static_cast<int>(K);

  // Strided scan: thread t checks j = lane, lane+W, lane+2W, ...
  for (int64_t j = lane; j < len2; j += W) {
    const scalar_t* p2_j = p2_n + j * 3;
    const scalar_t dx = qi_x - p2_j[0];
    const scalar_t dy = qi_y - p2_j[1];
    const scalar_t dz = qi_z - p2_j[2];

    if (fabs(dx) > radius || fabs(dy) > radius || fabs(dz) > radius) {
      continue;
    }

    const scalar_t dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 >= radius2) continue;

    seen++;

    // Claim a slot atomically
    int slot = atomicAdd(my_counter, 1);
    if (slot < K_int) {
      // Fill phase: each slot is written by exactly one thread
      out_idx[slot] = j;
      out_dist[slot] = dist2;
    } else {
      // Reservoir phase: replace random slot with decreasing probability
      rng = hash_rng(rng);
      // Probability of replacement: K / global_seen ≈ K / (seen * W)
      // But we use per-thread reservoir: replace with prob 1/seen
      // then pick a random victim in [0, K). This gives each item
      // in this thread's stream probability K/(K + seen_excess) of survival.
      int r = static_cast<int>(rng % static_cast<uint32_t>(seen));
      if (r == 0) {
        rng = hash_rng(rng);
        int victim = static_cast<int>(rng % static_cast<uint32_t>(K_int));
        // Benign race: at most one other thread may also write this slot
        // simultaneously. The result is still a valid random sample.
        out_idx[victim] = j;
        out_dist[victim] = dist2;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> BallQueryCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    bool skip_points_outside_cube) {
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2}, lengths1_t{lengths1, "lengths1", 3},
      lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "BallQueryCuda";
  at::checkAllSameGPU(c, {p1_t, p2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {p1_t, p2_t});

  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(p2.size(2) == p1.size(2), "Point sets must have same last dim");

  const int64_t N = p1.size(0);
  const int64_t P1 = p1.size(1);
  const int64_t K64 = static_cast<int64_t>(K);
  const float radius2 = radius * radius;

  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = at::full({N, P1, K}, -1, long_dtype);
  auto dists = at::zeros({N, P1, K}, p1.options());

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(idxs, dists);
  }

  const int64_t chunks_per_cloud = (P1 + 255) / 256;
  const size_t blocks = static_cast<size_t>(
      std::min((int64_t)65535, N * chunks_per_cloud));
  const size_t threads = 256;

  BallQueryKernel<float><<<blocks, threads, 0, stream>>>(
      p1.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      p2.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      lengths1.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      lengths2.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
      dists.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      K64,
      radius,
      radius2,
      skip_points_outside_cube);

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(idxs, dists);
}

std::tuple<at::Tensor, at::Tensor> BallQueryCoopCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    int64_t seed) {
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2}, lengths1_t{lengths1, "lengths1", 3},
      lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "BallQueryCoopCuda";
  at::checkAllSameGPU(c, {p1_t, p2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {p1_t, p2_t});

  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(p2.size(2) == p1.size(2), "Point sets must have same last dim");
  TORCH_CHECK(K % 8 == 0, "K must be divisible by 8 for cooperative kernel");

  const int64_t N = p1.size(0);
  const int64_t P1 = p1.size(1);
  const int64_t K64 = static_cast<int64_t>(K);
  const float radius2 = radius * radius;
  const uint32_t seed32 = static_cast<uint32_t>(seed & 0xFFFFFFFF);

  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = at::full({N, P1, K}, -1, long_dtype);
  auto dists = at::zeros({N, P1, K}, p1.options());

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(idxs, dists);
  }

  constexpr int W = 8;
  const int threads_per_block = 256;
  const int atoms_per_block = threads_per_block / W;  // 32
  const int64_t total_atoms = N * P1;
  const size_t n_blocks = static_cast<size_t>(
      std::min((int64_t)65535,
               (total_atoms + atoms_per_block - 1) / atoms_per_block));
  const size_t smem_size = atoms_per_block * sizeof(int);  // one counter per atom

  BallQueryKernelCoop<float, W><<<n_blocks, threads_per_block, smem_size, stream>>>(
      p1.data_ptr<float>(),
      p2.data_ptr<float>(),
      lengths1.data_ptr<int64_t>(),
      lengths2.data_ptr<int64_t>(),
      idxs.data_ptr<int64_t>(),
      dists.data_ptr<float>(),
      N, P1, K64,
      radius, radius2,
      seed32);

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(idxs, dists);
}

// ---------------------------------------------------------------------------
// Warp-cooperative forward kernel with predicted distance output.
// GT neighbor finding uses fp32 (p1, p2). For each qualifying neighbor pair,
// also computes |pred_i - pred_j|^2 in fp32 and stores in pred's dtype.
// ---------------------------------------------------------------------------
template <typename scalar_pred, int W>
__global__ void BallQueryKernelCoopWithPred(
    const float* __restrict__ p1_ptr,
    const float* __restrict__ p2_ptr,
    const scalar_pred* __restrict__ pred_ptr,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    int64_t* __restrict__ idxs_ptr,
    float* __restrict__ dists_gt_ptr,
    scalar_pred* __restrict__ dists_pred_ptr,
    const int64_t N,
    const int64_t P1,
    const int64_t K,
    const float radius,
    const float radius2,
    const uint32_t seed) {

  const int atoms_per_block = blockDim.x / W;
  const int group_id = threadIdx.x / W;
  const int lane = threadIdx.x % W;
  const int64_t global_atom_flat = static_cast<int64_t>(blockIdx.x) * atoms_per_block + group_id;

  const int64_t total_atoms = N * P1;
  if (global_atom_flat >= total_atoms) return;

  const int64_t n = global_atom_flat / P1;
  const int64_t i = global_atom_flat % P1;

  if (i >= lengths1[n]) return;

  extern __shared__ int smem_counters[];
  int* my_counter = &smem_counters[group_id];
  if (lane == 0) *my_counter = 0;
  __syncwarp(__activemask());

  const int64_t out_base = (n * P1 + i) * K;
  int64_t* out_idx = idxs_ptr + out_base;
  float* out_dist_gt = dists_gt_ptr + out_base;
  scalar_pred* out_dist_pred = dists_pred_ptr + out_base;

  // GT query point
  const float* p1_ni = p1_ptr + (n * P1 + i) * 3;
  const float qi_x = p1_ni[0];
  const float qi_y = p1_ni[1];
  const float qi_z = p1_ni[2];

  // Predicted query point (load in native dtype, promote to fp32)
  const scalar_pred* pred_ni = pred_ptr + (n * P1 + i) * 3;
  const float pi_x = static_cast<float>(pred_ni[0]);
  const float pi_y = static_cast<float>(pred_ni[1]);
  const float pi_z = static_cast<float>(pred_ni[2]);

  const float* p2_n = p2_ptr + n * P1 * 3;
  const scalar_pred* pred_n = pred_ptr + n * P1 * 3;

  int seen = 0;
  uint32_t rng = seed ^ static_cast<uint32_t>(
      n * 1000003u + i * 997u + lane * 31u);
  rng = hash_rng(rng);

  const int64_t len2 = lengths2[n];
  const int K_int = static_cast<int>(K);

  for (int64_t j = lane; j < len2; j += W) {
    const float* p2_j = p2_n + j * 3;
    const float dx = qi_x - p2_j[0];
    const float dy = qi_y - p2_j[1];
    const float dz = qi_z - p2_j[2];

    if (fabs(dx) > radius || fabs(dy) > radius || fabs(dz) > radius) {
      continue;
    }

    const float dist2_gt = dx * dx + dy * dy + dz * dz;
    if (dist2_gt >= radius2) continue;

    // Compute predicted distance for this pair
    const scalar_pred* pred_j = pred_n + j * 3;
    const float pdx = pi_x - static_cast<float>(pred_j[0]);
    const float pdy = pi_y - static_cast<float>(pred_j[1]);
    const float pdz = pi_z - static_cast<float>(pred_j[2]);
    const float dist2_pred = pdx * pdx + pdy * pdy + pdz * pdz;

    seen++;
    int slot = atomicAdd(my_counter, 1);
    if (slot < K_int) {
      out_idx[slot] = j;
      out_dist_gt[slot] = dist2_gt;
      out_dist_pred[slot] = static_cast<scalar_pred>(dist2_pred);
    } else {
      rng = hash_rng(rng);
      int r = static_cast<int>(rng % static_cast<uint32_t>(seen));
      if (r == 0) {
        rng = hash_rng(rng);
        int victim = static_cast<int>(rng % static_cast<uint32_t>(K_int));
        out_idx[victim] = j;
        out_dist_gt[victim] = dist2_gt;
        out_dist_pred[victim] = static_cast<scalar_pred>(dist2_pred);
      }
    }
  }
}

// Launcher for BallQueryKernelCoopWithPred
std::tuple<at::Tensor, at::Tensor, at::Tensor> BallQueryCoopWithPredCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& pred,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    int64_t seed) {
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2},
      pred_t{pred, "pred", 3},
      lengths1_t{lengths1, "lengths1", 4},
      lengths2_t{lengths2, "lengths2", 5};
  at::CheckedFrom c = "BallQueryCoopWithPredCuda";
  at::checkAllSameGPU(c, {p1_t, p2_t, pred_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {p1_t, p2_t});

  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(p1.scalar_type() == at::kFloat, "p1/p2 must be float32");
  TORCH_CHECK(p2.size(2) == p1.size(2), "Point sets must have same last dim");
  TORCH_CHECK(K % 8 == 0, "K must be divisible by 8 for cooperative kernel");

  const int64_t N = p1.size(0);
  const int64_t P1 = p1.size(1);
  const int64_t K64 = static_cast<int64_t>(K);
  const float radius2 = radius * radius;
  const uint32_t seed32 = static_cast<uint32_t>(seed & 0xFFFFFFFF);

  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = at::full({N, P1, K}, -1, long_dtype);
  auto dists_gt = at::zeros({N, P1, K}, p1.options());
  auto dists_pred = at::zeros({N, P1, K}, pred.options());

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(idxs, dists_gt, dists_pred);
  }

  constexpr int W = 8;
  const int threads_per_block = 256;
  const int atoms_per_block = threads_per_block / W;
  const int64_t total_atoms = N * P1;
  const size_t n_blocks = static_cast<size_t>(
      std::min((int64_t)65535,
               (total_atoms + atoms_per_block - 1) / atoms_per_block));
  const size_t smem_size = atoms_per_block * sizeof(int);

  AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, pred.scalar_type(), "ball_query_coop_with_pred", [&] {
    BallQueryKernelCoopWithPred<scalar_t, W><<<n_blocks, threads_per_block, smem_size, stream>>>(
        p1.data_ptr<float>(),
        p2.data_ptr<float>(),
        pred.data_ptr<scalar_t>(),
        lengths1.data_ptr<int64_t>(),
        lengths2.data_ptr<int64_t>(),
        idxs.data_ptr<int64_t>(),
        dists_gt.data_ptr<float>(),
        dists_pred.data_ptr<scalar_t>(),
        N, P1, K64,
        radius, radius2,
        seed32);
  });

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(idxs, dists_gt, dists_pred);
}

// ---------------------------------------------------------------------------
// Backward kernel for predicted distances.
// Adapted from PyTorch3D's KNearestNeighborBackwardKernel (BSD-3-Clause).
// For each (n, i, k): loads pred[i] and pred[idx[i][k]], computes
//   grad_val = 2 * grad_dists[i][k] * (pred[i] - pred[j])
// and atomicAdds ±grad_val into grad_pred[i] and grad_pred[j].
// ---------------------------------------------------------------------------
template <typename scalar_pred>
__global__ void BallQueryPredBackwardKernel(
    const scalar_pred* __restrict__ pred,
    const int64_t* __restrict__ idxs,
    const float* __restrict__ grad_dists,
    float* __restrict__ grad_pred,
    const int64_t N,
    const int64_t P1,
    const int64_t K) {
  const size_t total = static_cast<size_t>(N) * P1 * K;
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t linear = tid; linear < total; linear += stride) {
    const int64_t n = static_cast<int64_t>(linear / (P1 * K));
    const int64_t rem = static_cast<int64_t>(linear % (P1 * K));
    const int64_t i = rem / K;
    const int64_t k = rem % K;

    const int64_t j = idxs[n * P1 * K + i * K + k];
    if (j < 0) continue;

    const float g = grad_dists[n * P1 * K + i * K + k];
    if (g == 0.0f) continue;

    const int64_t base_i = (n * P1 + i) * 3;
    const int64_t base_j = (n * P1 + j) * 3;

    for (int d = 0; d < 3; ++d) {
      float diff = static_cast<float>(pred[base_i + d]) - static_cast<float>(pred[base_j + d]);
      float grad_val = 2.0f * g * diff;
      atomicAdd(&grad_pred[base_i + d], grad_val);
      atomicAdd(&grad_pred[base_j + d], -grad_val);
    }
  }
}

// Launcher for BallQueryPredBackwardKernel
at::Tensor BallQueryPredBackwardCuda(
    const at::Tensor& pred,
    const at::Tensor& idxs,
    const at::Tensor& grad_dists) {
  at::cuda::CUDAGuard device_guard(pred.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t N = pred.size(0);
  const int64_t P1 = pred.size(1);
  const int64_t K = idxs.size(2);

  auto grad_pred = at::zeros({N, P1, 3}, pred.options().dtype(at::kFloat));

  const size_t total = static_cast<size_t>(N) * P1 * K;
  if (total == 0) {
    return grad_pred;
  }

  const int threads = 256;
  const size_t blocks = std::min(
      static_cast<size_t>(65535),
      (total + threads - 1) / threads);

  auto grad_dists_f = grad_dists.to(at::kFloat);

  AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, pred.scalar_type(), "ball_query_pred_backward", [&] {
    BallQueryPredBackwardKernel<scalar_t><<<blocks, threads, 0, stream>>>(
        pred.data_ptr<scalar_t>(),
        idxs.data_ptr<int64_t>(),
        grad_dists_f.data_ptr<float>(),
        grad_pred.data_ptr<float>(),
        N, P1, K);
  });

  AT_CUDA_CHECK(cudaGetLastError());

  return grad_pred;
}
