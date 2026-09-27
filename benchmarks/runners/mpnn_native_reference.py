"""Precision-only adapters around the frozen CSSB naive MPNN forward.

Keep dense distance construction, concatenations, ordering, masks and parameter
layout. Subclasses change dtype boundaries only; no custom MiniWorld kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_engine.modules.mpnn._functional import MPNNLayerNorm
from miniworld_engine.modules.mpnn.naive import (
    DecLayer,
    EncLayer,
    NaiveProteinMPNN,
    PositionalEncodings,
    ProteinFeatures,
)


class NativePositions(PositionalEncodings):
    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask
        d = d + (1 - mask) * (2 * self.max_relative_feature + 1)
        onehot = F.one_hot(d, 2 * self.max_relative_feature + 2)
        return self.linear(onehot.to(self.linear.weight.dtype))


class NativeFeatures(ProteinFeatures):
    def _rbf(self, distances):
        # Distance and exp calculations remain FP32; projection operands are BF16.
        return super()._rbf(distances).to(self.edge_embedding.weight.dtype)


class NativeEncoder(EncLayer):
    def forward(self, h_v, h_e, edge_idx, mask_v=None, mask_attend=None):
        # The frozen top-level forward initializes node zeros in FP32.
        return super().forward(
            h_v.to(h_e.dtype), h_e, edge_idx,
            None if mask_v is None else mask_v.to(h_e.dtype),
            None if mask_attend is None else mask_attend.to(h_e.dtype),
        )


class NativeDecoder(DecLayer):
    def forward(self, h_v, h_e, mask_v=None, mask_attend=None):
        return super().forward(
            h_v, h_e,
            None if mask_v is None else mask_v.to(h_v.dtype),
            None if mask_attend is None else mask_attend.to(h_v.dtype),
        )


def _replace_norms(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            norm = MPNNLayerNorm(child.normalized_shape, eps=child.eps)
            norm.load_state_dict(child.state_dict())
            setattr(module, name, norm)
        else:
            _replace_norms(child)


class NativeNaiveMPNN(NaiveProteinMPNN):
    """The original naive algorithm with explicit native precision boundaries."""

    def __init__(self, *, dropout=0.25):
        super().__init__(k_neighbors=48, augment_trans=0, augment_rot=0, dropout=dropout)
        initial = self.state_dict()
        self.features = NativeFeatures(128, top_k=48, augment_trans=0, augment_rot=0)
        self.features.embeddings = NativePositions(16)
        self.encoder_layers = nn.ModuleList(
            NativeEncoder(128, 128, 128, dropout=dropout, scale=48) for _ in range(3)
        )
        self.decoder_layers = nn.ModuleList(
            NativeDecoder(128, 128, 128, dropout=dropout, scale=48) for _ in range(3)
        )
        self.load_state_dict(initial)
        _replace_norms(self)

    @staticmethod
    def get_decoding_masks(edge_idx, mask, decoding_order, patch_index_batch,
                           fixed_decoding_order_len=0):
        future, past = NaiveProteinMPNN.get_decoding_masks(
            edge_idx, mask, decoding_order, patch_index_batch, fixed_decoding_order_len
        )
        return future.to(torch.bfloat16), past.to(torch.bfloat16)
