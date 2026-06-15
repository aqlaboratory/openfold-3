/*
 * pybind11 bindings for the OpenFold3 smooth lDDT CUDA ball-query extension.
 *
 * Modified from PyTorch3D's ball-query bindings
 * (https://github.com/facebookresearch/pytorch3d, Meta Platforms, Inc.,
 *  BSD-3-Clause; see
 *  https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE)
 * by Liang Hong <lhong22@cse.cuhk.edu.hk>: trimmed to the ops smooth
 * lDDT needs and added the cooperative + with-pred + backward entry
 * points implemented in ball_query.cu.
 */

#include <torch/extension.h>

#include "ball_query.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &BallQuery, "Ball query (CUDA)");
  m.def("ball_query_coop", &BallQueryCoop,
        "Cooperative ball query with reservoir sampling (CUDA)");
  m.def("ball_query_coop_with_pred", &BallQueryCoopWithPred,
        "Cooperative ball query with predicted distance output (CUDA)");
  m.def("ball_query_pred_backward", &BallQueryPredBackward,
        "Backward for predicted distances (CUDA)");
}
