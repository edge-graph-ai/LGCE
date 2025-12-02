"""Factory functions and lightweight ablation model variants."""
import torch
import torch.nn as nn

from src.models.estimator import GraphCardinalityEstimatorMultiSubgraph


class _IdentityGNN(nn.Module):
    """Identity GNN that optionally projects embeddings to match output channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        if int(in_ch) == int(out_ch):
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_ch, out_ch)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, edge_index=None, *args, **kwargs):
        return self.proj(x)


class _IdentityEncoder(nn.Module):
    def forward(self, src, *args, **kwargs):
        return src


class _MeanPool(nn.Module):
    """Simple mean pooling as a drop-in for attention pooling."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0, keepdim=True)


class GraphCE_NoGIN(GraphCardinalityEstimatorMultiSubgraph):
    """Remove GIN message passing and rely on embedding + pooling."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_ch = int(getattr(self.embed, "dim", self.project[2].in_features))
        out_ch = int(self.project[2].in_features)
        self.gnn_encoder_data = _IdentityGNN(in_ch, out_ch)
        self.gnn_encoder_query = _IdentityGNN(in_ch, out_ch)


class GraphCE_NoSelfAttn(GraphCardinalityEstimatorMultiSubgraph):
    """Disable self-attention encoder."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = _IdentityEncoder()


class GraphCE_NoCrossAttn(GraphCardinalityEstimatorMultiSubgraph):
    """Disable cross-attention between query and memory tokens."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_cross_attention = False
        self.cross_attn = None
        self.cross_ffn = nn.Identity()
        self.cross_attn_norm = nn.Identity()
        self.cross_ffn_norm = nn.Identity()

    def cross_interact(self, memory, query, memory_key_padding_mask=None):
        return query


class GraphCE_NoAllAttention(GraphCardinalityEstimatorMultiSubgraph):
    """Remove all attention modules and replace pooling with mean aggregation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pool_data = _MeanPool()
        self.pool_query = _MeanPool()

        self.transformer_encoder = _IdentityEncoder()
        self.enable_cross_attention = False
        self.cross_attn = None
        self.cross_ffn = nn.Identity()
        self.cross_attn_norm = nn.Identity()
        self.cross_ffn_norm = nn.Identity()

        with torch.no_grad():
            if isinstance(self.cls_token, torch.Tensor):
                self.cls_token.zero_()
            if isinstance(self.query_token, torch.Tensor):
                self.query_token.zero_()
        try:
            self.cls_token.requires_grad = False
            self.query_token.requires_grad = False
        except Exception:
            pass


class GraphCE_NoGINNoAttention(GraphCardinalityEstimatorMultiSubgraph):
    """Disable both GIN encoders and all attention blocks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        in_ch = int(getattr(self.embed, "dim", self.project[2].in_features))
        out_ch = int(self.project[2].in_features)
        self.gnn_encoder_data = _IdentityGNN(in_ch, out_ch)
        self.gnn_encoder_query = _IdentityGNN(in_ch, out_ch)

        self.pool_data = _MeanPool()
        self.pool_query = _MeanPool()

        self.transformer_encoder = _IdentityEncoder()
        self.enable_cross_attention = False
        self.cross_attn = None
        self.cross_ffn = nn.Identity()
        self.cross_attn_norm = nn.Identity()
        self.cross_ffn_norm = nn.Identity()

        if hasattr(self, "apply_memory_positional_encoding") and callable(self.apply_memory_positional_encoding):
            self.apply_memory_positional_encoding = (lambda x: x)

        with torch.no_grad():
            if isinstance(self.cls_token, torch.Tensor):
                self.cls_token.zero_()
            if isinstance(self.query_token, torch.Tensor):
                self.query_token.zero_()
        try:
            self.cls_token.requires_grad = False
            self.query_token.requires_grad = False
        except Exception:
            pass

    def cross_interact(self, memory, query, memory_key_padding_mask=None):
        return query


def make_ablation_model(variant: str, **cfg):
    """Select an ablation variant of GraphCardinalityEstimatorMultiSubgraph."""
    v = (variant or "BASE").strip().upper()
    if v == "BASE":
        return GraphCardinalityEstimatorMultiSubgraph(**cfg)
    if v in ("NO_GIN", "NOGIN", "-GIN"):
        return GraphCE_NoGIN(**cfg)
    if v in ("NO_ATTENTION", "NO_ALL_ATTN", "NO_ATTN", "NA", "-ATTN", "NO_ALL_ATTENTION"):
        return GraphCE_NoAllAttention(**cfg)
    if v in ("NO_GIN_NO_ATTENTION", "NO_GIN_ATTENTION", "NO_GIN_ALL_OFF", "NOGIN_NOATTN"):
        return GraphCE_NoGINNoAttention(**cfg)
    if v in ("NO_ENCODER", "NO_SELF_ATTN", "NOSELFATTN", "-ENCODER"):
        return GraphCE_NoSelfAttn(**cfg)
    if v in ("NO_DECODER", "NO_CROSS_ATTN", "NOCROSSATTN", "-DECODER"):
        return GraphCE_NoCrossAttn(**cfg)

    raise ValueError(f"Unknown MODEL_VARIANT: {variant}")
