# -*- coding: utf-8 -*-
import torch.nn as nn
from .estimator import GraphCardinalityEstimatorMultiSubgraph

class _IdentityGNN(nn.Module):
    def forward(self, x, edge_index=None, *args, **kwargs):
        return x

class _IdentityEncoder(nn.Module):
    def forward(self, src, *args, **kwargs):
        return src

class GraphCE_NoGIN(GraphCardinalityEstimatorMultiSubgraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_encoder_data  = _IdentityGNN()
        self.gnn_encoder_query = _IdentityGNN()

class GraphCE_NoSelfAttn(GraphCardinalityEstimatorMultiSubgraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = _IdentityEncoder()

class GraphCE_NoCrossAttn(GraphCardinalityEstimatorMultiSubgraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_cross_attention = False
        self.cross_attn = None
        self.cross_ffn = nn.Identity()
        self.cross_attn_norm = nn.Identity()
        self.cross_ffn_norm = nn.Identity()

    def cross_interact(self, memory, query, memory_key_padding_mask=None):
        return query


def make_model(variant: str, **cfg):

    v = (variant or "BASE").strip().upper()
    if v == "BASE":
        return GraphCardinalityEstimatorMultiSubgraph(**cfg)
    if v in ("NO_GIN", "NOGIN", "-GIN"):
        return GraphCE_NoGIN(**cfg)
    if v in ("NO_ENCODER", "NO_SELF_ATTN", "NOSELFATTN", "-ENCODER"):
        return GraphCE_NoSelfAttn(**cfg)
    if v in ("NO_DECODER", "NO_CROSS_ATTN", "NOCROSSATTN", "-DECODER"):
        return GraphCE_NoCrossAttn(**cfg)
    raise ValueError(f"Unknown MODEL_VARIANT: {variant}")
