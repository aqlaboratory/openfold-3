from torch import nn

from .diffusion_transformer import DiffusionTransformer
from openfold3.base.model.primitives import LayerNorm, Linear


class AtomTransformer(nn.Module):
    """
    Implements AF3 Algorithm 7.
    """
    def __init__(self, c_q, c_p, c_hidden, no_heads, no_blocks, n_transition, inf):
        """

        Args:
            c_q:
            c_p:
            c_hidden:
            no_heads:
            no_blocks:
            n_transition:
            inf:
        """
        super(AtomTransformer, self).__init__()

        self.diffusion_transformer = DiffusionTransformer(c_s=c_q,
                                                          c_z=c_p,
                                                          c_hidden=c_hidden,
                                                          no_heads=no_heads,
                                                          no_blocks=no_blocks,
                                                          n_transition=n_transition,
                                                          inf=inf)

    def forward(self):
        pass


class AtomFeatureEmbedder(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_atom: int,
        c_atom_pair: int
    ):
        """

        Args:
            c_in:
            c_atom:
            c_atom_pair:
        """
        super(AtomFeatureEmbedder, self).__init__()
        self.linear_feats = Linear(c_in, c_atom, bias=False)
        self.linear_ref_offset = Linear(3, c_atom_pair, bias=False)
        self.linear_inv_dists = Linear(1, c_atom_pair, bias=False)
        self.linear_valid_mask = Linear(1, c_atom_pair, bias=False)

    def forward(self):
        pass


class NoisyPositionEmbedder(nn.Module):
    def __init__(
        self,
        c_atom: int,
        c_atom_pair: int
    ):
        """

        Args:
            c_atom:
            c_atom_pair:
        """
        super(NoisyPositionEmbedder, self).__init__()
        self.layer_norm_s = LayerNorm(c_atom)
        self.linear_s = Linear(c_atom, c_atom, bias=False)
        self.layer_norm_z = LayerNorm(c_atom_pair)
        self.linear_z = Linear(c_atom_pair, c_atom_pair, bias=False)
        self.linear_r = Linear(c_atom, c_atom, bias=False)

    def forward(self):
        pass


class AtomAttentionEncoder(nn.Module):
    """
    Implements AF3 Algorithm 5.
    """
    def __init__(
        self,
        c_in: int,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        c_hidden: int,
        add_noisy_pos: bool,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        inf: float = 1e9
    ):
        """

        Args:
            c_in:
            c_atom:
            c_atom_pair:
            c_token:
            c_hidden:
            add_noisy_pos:
            no_heads:
            no_blocks:
            n_transition:
            inf:
        """
        super(AtomAttentionEncoder, self).__init__()

        self.add_noisy_pos = add_noisy_pos

        self.atom_feature_emb = AtomFeatureEmbedder(c_in=c_in,
                                                    c_atom=c_atom,
                                                    c_atom_pair=c_atom_pair)

        if add_noisy_pos:
            self.noisy_position_emb = NoisyPositionEmbedder(c_atom=c_atom,
                                                            c_atom_pair=c_atom_pair)

        self.relu = nn.ReLU()
        self.linear_l = Linear(c_atom, c_atom, bias=False, init="relu")
        self.linear_m = Linear(c_atom, c_atom, bias=False, init="relu")

        self.pair_mlp = nn.Sequential(nn.ReLU(),
                                      Linear(c_atom_pair, c_atom_pair, bias=False, init="relu"),
                                      nn.ReLU(),
                                      Linear(c_atom_pair, c_atom_pair, bias=False, init="relu"),
                                      nn.ReLU(),
                                      Linear(c_atom_pair, c_atom_pair, bias=False, init="relu"))

        self.atom_transformer = AtomTransformer(c_q=c_atom,
                                                c_p=c_atom_pair,
                                                c_hidden=c_hidden,
                                                no_heads=no_heads,
                                                no_blocks=no_blocks,
                                                n_transition=n_transition,
                                                inf=inf)

        self.linear_q_out = Linear(c_atom, c_token, bias=False, init="final")

    def forward(self):
        pass


class AtomAttentionDecoder(nn.Module):
    """
    Implements AF3 Algorithm 6.
    """
    def __init__(
        self,
        c_q: int,
        c_p: int,
        c_hidden: int,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        inf: float = 1e9
    ):
        """

        Args:
            c_q:
            c_p:
            c_hidden:
            no_heads:
            no_blocks:
            n_transition:
            inf:
        """
        super(AtomAttentionDecoder, self).__init__()

        self.linear_q_in = Linear(c_q, c_q, bias=False)

        self.atom_transformer = AtomTransformer(c_q=c_q,
                                                c_p=c_p,
                                                c_hidden=c_hidden,
                                                no_heads=no_heads,
                                                no_blocks=no_blocks,
                                                n_transition=n_transition,
                                                inf=inf)

        self.layer_norm = LayerNorm(c_in=c_q)
        self.linear_q_out = Linear(c_q, c_q, bias=False, init="final")

    def forward(self):
        pass
