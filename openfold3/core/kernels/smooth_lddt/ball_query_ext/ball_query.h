/*
 * Header for OpenFold3 smooth lDDT CUDA ball-query extension.
 *
 * Modified from
 * https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query.h
 * (Meta Platforms, Inc., BSD-3-Clause; see
 *  https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE)
 * by Liang Hong <lhong22@cse.cuhk.edu.hk>: added declarations for the
 * cooperative variants (BallQueryCoop, BallQueryCoopWithPred) and the
 * dedicated backward (BallQueryPredBackward) consumed by the autograd
 * Function in openfold3/core/kernels/smooth_lddt/__init__.py.
 */
#pragma once

#include <torch/extension.h>
#include <tuple>

// CUDA implementation — sequential (original)
std::tuple<at::Tensor, at::Tensor> BallQueryCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int K,
    const float radius,
    const bool skip_points_outside_cube);

// CUDA implementation — warp-cooperative with reservoir sampling
std::tuple<at::Tensor, at::Tensor> BallQueryCoopCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    int64_t seed);

// Public API which dispatches to CUDA and errors on CPU.
inline std::tuple<at::Tensor, at::Tensor> BallQuery(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    bool skip_points_outside_cube) {
  TORCH_CHECK(p1.is_cuda() && p2.is_cuda(), "BallQuery: only CUDA tensors supported");
  return BallQueryCuda(
      p1.contiguous(), p2.contiguous(), lengths1.contiguous(), lengths2.contiguous(),
      K, radius, skip_points_outside_cube);
}

inline std::tuple<at::Tensor, at::Tensor> BallQueryCoop(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    int64_t seed) {
  TORCH_CHECK(p1.is_cuda() && p2.is_cuda(), "BallQueryCoop: only CUDA tensors supported");
  return BallQueryCoopCuda(
      p1.contiguous(), p2.contiguous(), lengths1.contiguous(), lengths2.contiguous(),
      K, radius, seed);
}

// CUDA implementation — warp-cooperative with pred distance output
std::tuple<at::Tensor, at::Tensor, at::Tensor> BallQueryCoopWithPredCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& pred,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    int64_t seed);

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> BallQueryCoopWithPred(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& pred,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius,
    int64_t seed) {
  TORCH_CHECK(p1.is_cuda() && p2.is_cuda() && pred.is_cuda(),
              "BallQueryCoopWithPred: only CUDA tensors supported");
  return BallQueryCoopWithPredCuda(
      p1.contiguous(), p2.contiguous(), pred.contiguous(),
      lengths1.contiguous(), lengths2.contiguous(),
      K, radius, seed);
}

// CUDA implementation — backward for pred distances
at::Tensor BallQueryPredBackwardCuda(
    const at::Tensor& pred,
    const at::Tensor& idxs,
    const at::Tensor& grad_dists);

inline at::Tensor BallQueryPredBackward(
    const at::Tensor& pred,
    const at::Tensor& idxs,
    const at::Tensor& grad_dists) {
  TORCH_CHECK(pred.is_cuda(), "BallQueryPredBackward: only CUDA tensors supported");
  return BallQueryPredBackwardCuda(
      pred.contiguous(), idxs.contiguous(), grad_dists.contiguous());
}
