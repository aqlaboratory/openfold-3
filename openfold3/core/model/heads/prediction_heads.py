# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.atomize_utils import max_atom_per_token_masked_select


class PairformerEmbedding(nn.Module):
    """
    Implements AF3 Algorithm 31, line 1 - 6
    """

    def __init__(
        self,
        pairformer: ConfigDict,
        c_s_input: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bin: int,
        inf: float,
        linear_init_params: ConfigDict = lin_init.pairformer_head_init,
    ):
        """
        Args:
            pairformer:
                Config for PairFormerStack used
            c_s_input:
                Single (input) embedding dimension
            c_z:
                Pair embedding dimension
            min_bin:
                Minimum value for bin (3.25). The value is slightly
                different from SI. Previous AF2 implementation utilized these values
                for bins.
            max_bin:
                Maximum value for bin (20.75). ibid
            no_bin:
                Number of bins (15). ibid
            inf:
                Inf (1e8). ibid
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bin = no_bin
        self.inf = inf

        self.linear_i = Linear(c_s_input, c_z, **linear_init_params.linear_i)
        self.linear_j = Linear(c_s_input, c_z, **linear_init_params.linear_j)

        self.linear_distance = Linear(
            self.no_bin, c_z, **linear_init_params.linear_distance
        )
        self.pairformer_stack = PairFormerStack(**pairformer)

    def embed_zij(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
    ):
        orig_dtype = zij.dtype
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            # si projection to zij
            zij = (
                zij
                + self.linear_i(si_input.unsqueeze(-2))
                + self.linear_j(si_input.unsqueeze(-3))
            )

            # Embed pair distances of representative atoms
            bins = torch.linspace(
                self.min_bin,
                self.max_bin,
                self.no_bin,
                device=zij.device,
                dtype=zij.dtype,
            )
            squared_bins = bins**2
            upper = torch.cat(
                [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
            )
            dij = torch.sum(
                (x_pred[..., None, :] - x_pred[..., None, :, :]) ** 2,
                dim=-1,
                keepdims=True,
            )
            dij = ((dij > squared_bins) * (dij < upper)).type(x_pred.dtype)
            zij = zij + self.linear_distance(dij)

        return zij.to(dtype=orig_dtype)

    def per_sample_pairformer_emb(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        offload_inference: bool = False,
        _mask_trans: bool = True,
    ):
        batch_dims = x_pred.shape[:-2]
        no_samples = x_pred.shape[-3]

        device = "cpu" if offload_inference else x_pred.device

        # Prepare output tensors
        si_out = torch.zeros_like(
            si.expand(*(batch_dims + si.shape[-2:])), device=device
        )
        zij_out = torch.zeros_like(
            zij.expand(*(batch_dims + zij.shape[-3:])), device=device
        )

        # TODO: Refactor to support inplace ops
        for i in range(no_samples):
            zij_chunk = self.embed_zij(
                si_input=si_input, zij=zij, x_pred=x_pred[:, i : i + 1]
            )

            si_chunk, zij_chunk = self.pairformer_stack(
                si.clone(),  # Avoid inplace ops on si for now
                zij_chunk,
                single_mask,
                pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )

            si_out[..., i : i + 1, :, :] = si_chunk.to(device=device)
            zij_out[..., i : i + 1, :, :, :] = zij_chunk.to(device=device)

        # If offloading, do not return to device for now and let caller handle it
        return si_out, zij_out

    def pairformer_emb(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ):
        zij = self.embed_zij(si_input=si_input, zij=zij, x_pred=x_pred)

        # Expand sample dimension
        batch_dims = x_pred.shape[:-2]
        si = si.expand(*(batch_dims + si.shape[-2:])).clone()
        single_mask = single_mask.expand(*(batch_dims + single_mask.shape[-1:]))
        pair_mask = pair_mask.expand(*(batch_dims + pair_mask.shape[-2:]))

        # TODO: Make this less awkward, DS kernel has strict shape asserts
        #  and expects batch and seq dims to exist, but no sample dim
        if use_deepspeed_evo_attention:
            si = si.reshape(-1, *si.shape[-2:])
            zij = zij.reshape(-1, *zij.shape[-3:])
            single_mask = single_mask.reshape(-1, single_mask.shape[-1])
            pair_mask = pair_mask.reshape(-1, *pair_mask.shape[-2:])

        # PairFormer embedding
        si, zij = self.pairformer_stack(
            si,
            zij,
            single_mask,
            pair_mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        if use_deepspeed_evo_attention:
            si = si.reshape(*batch_dims, *si.shape[-2:])
            zij = zij.reshape(*batch_dims, *zij.shape[-3:])

        return si, zij

    def forward(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        offload_inference: bool = False,
        _mask_trans: bool = True,
        apply_per_sample: bool = False,
    ):
        """
        Args:
            si_input:
                [*, N_token, C_s] Output of InputFeatureEmbedder
            si:
                [*, N_token, C_s] Single embedding
            zij:
                [*, N_token, N_token, C_z] Pairwise embedding
            x_pred:
                [*, N_token, 3] Representative atom predicted coordinates per token
            single_mask:
                [*, N_token] Single mask
            pair_mask:
                [*, N_token, N_token] Pair mask
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            offload_inference:
                Whether to offload some computation to CPU
            _mask_trans:
                Whether to mask the output of the transition layers
            apply_per_sample:
                Run PairFormer embedding for each sample individually.
                This is a memory optimization which is only used during
                validation/inference and will depend on the number of samples
                in the full rollout.


        Returns:
            si:
                [*, N_token, C_s] Updated single representation
            zij:
                [*, N_token, N_token, C_z] Updated pair representation
        """
        if apply_per_sample:
            si, zij = self.per_sample_pairformer_emb(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=x_pred,
                single_mask=single_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                offload_inference=offload_inference,
                _mask_trans=_mask_trans,
            )
        else:
            # TODO: Fix chunking issues with > 1 sample
            #  Chunking disabled for now
            si, zij = self.pairformer_emb(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=x_pred,
                single_mask=single_mask,
                pair_mask=pair_mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )

        return si, zij


class PredictedAlignedErrorHead(nn.Module):
    """
    Implements PredictedAlignedError Head (Algorithm 31, Line 5) for
    AF3 (subsection 4.3.2)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.pae_init,
        **kwargs,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedAlignedError (PAE) bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.layer_norm = LayerNorm(self.c_z)
        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def _compute_logits(self, zij: torch.Tensor):
        logits = self.linear(self.layer_norm(zij))
        return logits

    def _chunk(
        self,
        zij: torch.Tensor,
    ) -> torch.Tensor:
        zij_out = torch.zeros(
            (*zij.shape[:-1], self.c_out), device=zij.device, dtype=zij.dtype
        )
        no_samples = zij.shape[-4]
        for i in range(no_samples):
            zij_out[:, i : i + 1] = self._compute_logits(zij[:, i : i + 1])

        return zij_out

    def forward(self, zij, apply_per_sample: bool = False):
        """
        Args:
            zij:
                [*, N, N, C_z] Pair embedding
            apply_per_sample:
                Run PAE head for each sample individually.
                This is a memory optimization which is only used during
                validation/inference and will depend on the number of samples
                in the full rollout.
        Returns:
            logits:
                [*, N, N, C_out] Logits
        """
        if apply_per_sample:
            logits = self._chunk(zij=zij)
        else:
            logits = self._compute_logits(zij=zij)

        return logits


class PredictedDistanceErrorHead(nn.Module):
    """
    Implements PredictedDistanceError Head (Algorithm 31, Line 6) for
    AF3 (subsection 4.3.3)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.pde_init,
        **kwargs,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedDistanceError (PDE) bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.layer_norm = LayerNorm(self.c_z)
        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def _compute_logits(self, zij: torch.Tensor):
        logits = self.linear(self.layer_norm(zij))
        logits = logits + logits.transpose(-2, -3)
        return logits

    def _chunk(
        self,
        zij: torch.Tensor,
    ) -> torch.Tensor:
        zij_out = torch.zeros(
            (*zij.shape[:-1], self.c_out), device=zij.device, dtype=zij.dtype
        )
        no_samples = zij.shape[-4]
        for i in range(no_samples):
            zij_out[:, i : i + 1] = self._compute_logits(zij[:, i : i + 1])

        return zij_out

    def forward(self, zij, apply_per_sample: bool = False):
        """
        Args:
            zij:
                [*, N, N, C_z] Pair embedding
            apply_per_sample:
                Run PDE head for each sample individually.
                This is a memory optimization which is only used during
                validation/inference and will depend on the number of samples
                in the full rollout.
        Returns:
            logits:
                [*, N, N, C_out] Logits
        """
        if apply_per_sample:
            logits = self._chunk(zij=zij)
        else:
            logits = self._compute_logits(zij=zij)

        return logits


class PerResidueLDDTAllAtom(nn.Module):
    """
    Implements Plddt Head (Algorithm 31, Line 7) for AF3 (subsection 4.3.1)
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        max_atoms_per_token: int,
        linear_init_params: ConfigDict = lin_init.lddt_init,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            max_atoms_per_token:
                Maximum atoms per token
            c_out:
                Number of PLDDT bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_s = c_s
        self.max_atoms_per_token = max_atoms_per_token
        self.c_out = c_out

        self.layer_norm = LayerNorm(self.c_s)
        self.linear = Linear(
            self.c_s, self.max_atoms_per_token * self.c_out, **linear_init_params.linear
        )

    def forward(self, s: torch.Tensor, max_atom_per_token_mask: torch.Tensor):
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            max_atom_per_token_mask:
                [*, N_token * max_atoms_per_token] Flat mask of atoms per token
                padded to max_atoms_per_token
        Returns:
            logits:
                [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        # Flatten batch dims
        max_atom_per_token_mask = max_atom_per_token_mask.reshape(
            -1, n_token * self.max_atoms_per_token
        )

        # [*, N_token, max_atoms_per_token * c_out]
        logits = self.linear(self.layer_norm(s))

        # [*, N_token * max_atoms_per_token, c_out]
        logits = logits.reshape(
            *batch_dims, n_token * self.max_atoms_per_token, self.c_out
        )

        # [*, N_atom, c_out]
        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
        )

        return logits


class PerResidueLDDTCaPredictor(nn.Module):
    """
    Implements plddtHead for AF2, subsection 1.9.10

    Source: OpenFold
    """

    def __init__(
        self,
        no_bins: int,
        c_in: int,
        c_hidden: int,
        linear_init_params: ConfigDict = lin_init.lddt_ca_init,
        **kwargs,
    ):
        super().__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, **linear_init_params.linear_1)
        self.linear_2 = Linear(
            self.c_hidden, self.c_hidden, **linear_init_params.linear_2
        )
        self.linear_3 = Linear(
            self.c_hidden, self.no_bins, **linear_init_params.linear_3
        )

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class ExperimentallyResolvedHeadAllAtom(nn.Module):
    """
    Implements resolvedHeads for AF3, subsection 4.3.3
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        max_atoms_per_token: int,
        linear_init_params: ConfigDict = lin_init.exp_res_all_atom_init,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            max_atoms_per_token:
                Maximum atoms per token
            c_out:
                Number of ExperimentallyResolved Head AllAtom bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_s = c_s
        self.max_atoms_per_token = max_atoms_per_token
        self.c_out = c_out

        self.layer_norm = LayerNorm(self.c_s)
        self.linear = Linear(
            self.c_s, self.max_atoms_per_token * self.c_out, **linear_init_params.linear
        )

    def forward(self, s: torch.Tensor, max_atom_per_token_mask: torch.Tensor):
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            max_atom_per_token_mask:
                [*, N_token * max_atoms_per_token] Flat mask of atoms per token
                padded to max_atoms_per_token
        Returns:
            logits:
                [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        # Flatten batch dims
        max_atom_per_token_mask = max_atom_per_token_mask.reshape(
            -1, n_token * self.max_atoms_per_token
        )

        # [*, N_token, max_atoms_per_token * c_out]
        logits = self.linear(self.layer_norm(s))

        # [*, N_token * max_atoms_per_token, c_out]
        logits = logits.reshape(
            *batch_dims, n_token * self.max_atoms_per_token, self.c_out
        )

        # [*, N_atom, c_out]
        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
        )

        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    Implements resolvedHeads for AF2.
    For use in computation of experimentally resolved loss, subsection 1.9.10 (AF2)

    Source: OpenFold
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.exp_res_init,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of experimentally resolved atom bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, **linear_init_params.linear)

    def forward(self, s):
        """
        Args:
            s:
                [*, N, C_s] Single embedding
        Returns:
            logits:
                [*, N, C_out] Logits
        """

        logits = self.linear(s)
        return logits


class DistogramHead(nn.Module):
    """
    Implementation of distogram head for both AF2 and AF3.

    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2), section 4.4 (AF3)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.distogram_init,
        **kwargs,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of distogram bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def forward(self, z):
        """
        Args:
            z:
                [*, N, N, C_z] Pair embedding
        Returns:
            logit:
                [*, N, N, C_out] Distogram probability distribution

        Note:
            For symmetric pairwise PairDistanceError loss (PDE),
            logits are calculated by linear(zij + zij.transpose(-2, -3))
            In SI this happens before the linear layer is applied.
        """

        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7 (AF2)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.tm_score_init,
        **kwargs,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def forward(self, z):
        """
        Args:
            z:
                [*, N, N, C_z] Pairwise embedding
        Returns:
            logits:
                [*, N, N, C_out] Logits
        """

        logits = self.linear(z)
        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9 (AF2)

    Source: OpenFold
    """

    def __init__(
        self,
        c_m: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.masked_msa_init,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, **linear_init_params.linear)

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            logits:
                [*, N_seq, N_res, C_out] Logits
        """

        logits = self.linear(m)
        return logits
