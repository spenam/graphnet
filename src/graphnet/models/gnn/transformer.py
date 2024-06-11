"""
    Vanilla transformer.
"""
import torch
import math
import torch.nn as nn
from typing import Set, Dict, Any, Optional, Union, List

# Modify here the encoder layers
from graphnet.models.components.layers import (
    Encoder_block,
)

from graphnet.models.components.embedding import (
    FeaturesProcessing,
    PositionalEncoding,
)

from graphnet.models.gnn.gnn import GNN # Base class for all core GNN models in graphnet.
from graphnet.models.utils import array_to_sequence # Convert `x` of shape [n, d] into a padded sequence of shape [B, L, D].

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


class Transformer(GNN):
    """Vanilla transformer model."""

    def __init__(
        self,
        emb_dims: Union[List, int],
        seq_length: int = 300,
        n_features: int = 8,
        position_encoding: bool = True,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        ):
        """ Construct a Vanilla Transformer.

        Args:
            seq_lenght: The total length of the event.
            n_features: The number of features in the input data.
            position_encoder: Wether or not, include position Fourier encoding.
            emb_dims: Embedding dimensions and/or dimension of the model.
            num_heads: Number of heads in MHA.
            dropout_attn: Dropout to be applied in MHA.
            hidden_dim: Dimension of FFNN.
            dropout_FFNN: Dropout to be applied in MHA.
            no_hits_blocks: Number of Encoder block using only hit information.
            no_evt_blocks: Number of Encoder block including cls token, i.e. considering global event information.
        """
        super().__init__(n_features, emb_dims if isinstance(emb_dims, int) else emb_dims[-1]) #nb_inputs, nb_outputs

        # Take the dimension of the model as the last dimension from emb_dims
        if isinstance(emb_dims, int):
            dim = emb_dims
        elif isinstance(emb_dims, List):
            dim = emb_dims[-1]

        self.n_features = n_features
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Layers:
        self.processing = FeaturesProcessing(emb_dims, n_features)
        self.position_encoding = position_encoding
        self.pos_enc = PositionalEncoding(dim, seq_length)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        self.hits_blocks = nn.Sequential(*[Encoder_block(dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN) for _ in range(no_hits_blocks)])
        self.evt_blocks = nn.Sequential(*[Encoder_block(dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN) for _ in range(no_evt_blocks)])

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        x0, mask0, evt_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )

        B, L, _ = x0.shape

        x = self.processing(x0)
        cls_token = self.cls_token.repeat(B, 1, 1)

        if self.position_encoding:
            x = self.pos_enc(x)

        mask = torch.zeros(mask0.shape, dtype = mask0.dtype, device = mask0.device)
        mask[~mask0] = -torch.inf

        if self.no_evt_blocks is None or self.no_evt_blocks == 0:
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype = mask0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask)
        else:
            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask)

            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype = mask0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for evt_block in self.evt_blocks:
                x = evt_block(x, mask=mask)

        return x[:, 0]
