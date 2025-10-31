# -*- coding: utf-8 -*-
import torch.nn as nn
from .estimator import GraphCardinalityEstimatorMultiSubgraph

# —— 恒等层（保持接口但“什么也不做”） —— #
class _IdentityGNN(nn.Module):
    def forward(self, x, edge_index=None, *args, **kwargs):
        return x

class _IdentityEncoder(nn.Module):
    def forward(self, src, *args, **kwargs):
        return src

# —— 三个消融变体 —— #
class GraphCE_NoGIN(GraphCardinalityEstimatorMultiSubgraph):
    """取消 GIN：不做消息传递，仅用嵌入+池化"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_encoder_data  = _IdentityGNN()
        self.gnn_encoder_query = _IdentityGNN()

class GraphCE_NoSelfAttn(GraphCardinalityEstimatorMultiSubgraph):
    """取消自注意力：子图 token 不相互交互"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = _IdentityEncoder()

class GraphCE_NoCrossAttn(GraphCardinalityEstimatorMultiSubgraph):
    """取消轻量跨注意力：仅用查询 token 预测"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_cross_attention = False
        self.cross_attn = None
        self.cross_ffn = nn.Identity()
        self.cross_attn_norm = nn.Identity()
        self.cross_ffn_norm = nn.Identity()

    def cross_interact(self, memory, query, memory_key_padding_mask=None):
        return query

# —— 工厂 —— #
def make_model(variant: str, **cfg):
    """
    variant: 'BASE' | 'NO_GIN' | 'NO_ENCODER' | 'NO_DECODER'
    其余参数与 GraphCardinalityEstimatorMultiSubgraph 一致
    """
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
