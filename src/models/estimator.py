# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiScaleTokenPool
from .GNN import GINEncoder

def _maybe_log1p_degree(deg_tensor):
    if deg_tensor is None or not torch.is_tensor(deg_tensor):
        return None
    deg = deg_tensor.float()
    deg = torch.clamp(deg, min=0.0, max=1e6)  # ★ 限制最大值，防止 Inf
    deg = torch.where(torch.isfinite(deg), deg, torch.tensor(1.0, device=deg.device))
    return torch.log1p(deg)

class _Embed(nn.Module):
    def __init__(self, num_vertices: int, num_labels: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.id_emb    = nn.Embedding(num_vertices, dim)
        self.label_emb = nn.Embedding(num_labels,  dim)
        self.deg_proj  = nn.Linear(1, dim)

        self.ln   = nn.LayerNorm(dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.id_emb.weight)
        nn.init.xavier_uniform_(self.label_emb.weight)
        nn.init.xavier_uniform_(self.deg_proj.weight)
        nn.init.zeros_(self.deg_proj.bias)
        self.vertex_emb  = self.id_emb      # 让 _maybe_resize_embeddings() 能发现顶点 embedding
        self.label_embed = self.label_emb   # 两个名字都能被发现（可选，但推荐）
    def _safe_gather(self, emb: nn.Embedding, idx: torch.Tensor):
        return emb(idx)

    def _fuse(self, id_vec, label_vec, degree):
        if degree is not None:
            d = _maybe_log1p_degree(degree).unsqueeze(-1)  # [N,1]
            deg_vec = self.deg_proj(d)                     # [N,D]
            x = id_vec + label_vec + deg_vec
        else:
            x = id_vec + label_vec
        x = self.drop(self.act(self.ln(x)))
        return x

    def forward_data(self, vertex_ids: torch.Tensor, labels: torch.Tensor, degree: torch.Tensor | None):
        id_vec    = self._safe_gather(self.id_emb, vertex_ids)   # [N,D]
        label_vec = self._safe_gather(self.label_emb, labels)    # [N,D]
        return self._fuse(id_vec, label_vec, degree)

    def forward_query(self, labels: torch.Tensor, degree: torch.Tensor | None):
        label_vec = self._safe_gather(self.label_emb, labels)    # [N,D]
        id_vec = torch.zeros_like(label_vec)
        return self._fuse(id_vec, label_vec, degree)

class GraphCardinalityEstimatorMultiSubgraph(nn.Module):
    def __init__(self,
                 gnn_in_ch: int = 16, gnn_hidden_ch: int = 16, gnn_out_ch: int = 16, num_gnn_layers: int = 2,
                 transformer_dim: int = 16, transformer_heads: int = 4, transformer_ffn_dim: int = 32,
                 transformer_layers: int = 2, num_subgraphs: int = 8,
                 num_vertices: int = 100_000, num_labels: int = 64,
                 dropout: float = 0.1,
                 enable_embed_shortcut: bool = True,
                 embed_shortcut_init: float = 0.0,
                 use_multi_scale_pool: bool = True,
                 multi_scale_fusion: str = "gate",
                 multi_scale_dropout: float | None = None,
                 multi_scale_gate_init: float = 0.0,
                 multi_scale_attn_hidden: int | None = None):
        super().__init__()

        D = transformer_dim
        self.embed = _Embed(num_vertices=num_vertices, num_labels=num_labels, dim=gnn_in_ch, dropout=dropout)

        self.gnn_encoder_data = GINEncoder(
            in_ch=gnn_in_ch, hidden_ch=gnn_hidden_ch, out_ch=gnn_out_ch, num_layers=num_gnn_layers, dropout=dropout,
            jk_mode="sum", residual=True, train_eps=True
        )
        self.gnn_encoder_query = GINEncoder(
            in_ch=gnn_in_ch, hidden_ch=gnn_hidden_ch, out_ch=gnn_out_ch, num_layers=num_gnn_layers, dropout=dropout,
            jk_mode="sum", residual=True, train_eps=True
        )

        ms_attn_hidden = multi_scale_attn_hidden if multi_scale_attn_hidden is not None else D
        ms_dropout = multi_scale_dropout if multi_scale_dropout is not None else dropout
        self.pool_data = MultiScaleTokenPool(
            gnn_out_ch,
            attn_hidden=ms_attn_hidden,
            dropout=dropout,
            enabled=use_multi_scale_pool,
            fusion=multi_scale_fusion,
            gate_init=multi_scale_gate_init,
            fusion_dropout=ms_dropout,
        )
        self.pool_query = MultiScaleTokenPool(
            gnn_out_ch,
            attn_hidden=ms_attn_hidden,
            dropout=dropout,
            enabled=use_multi_scale_pool,
            fusion=multi_scale_fusion,
            gate_init=multi_scale_gate_init,
            fusion_dropout=ms_dropout,
        )

        # ★ 新增：池化温度（正值）。softplus 保证 >0，初值≈1.1，让注意力更“尖锐”
        self._pool_scale_data  = nn.Parameter(torch.tensor(0.2))
        self._pool_scale_query = nn.Parameter(torch.tensor(0.2))

        # ★ 嵌入捷径门控
        def _make_shortcut_layer():
            if int(gnn_in_ch) == int(gnn_out_ch):
                return nn.Identity()
            layer = nn.Linear(gnn_in_ch, gnn_out_ch)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            return layer

        self.enable_embed_shortcut = bool(enable_embed_shortcut)
        self.shortcut_proj_data = _make_shortcut_layer()
        self.shortcut_proj_query = _make_shortcut_layer()
        self._shortcut_gate_act = nn.Sigmoid()
        if self.enable_embed_shortcut:
            self.embed_shortcut_alpha_data = nn.Parameter(torch.tensor(float(embed_shortcut_init)))
            self.embed_shortcut_alpha_query = nn.Parameter(torch.tensor(float(embed_shortcut_init)))
        else:
            self.embed_shortcut_alpha_data = None
            self.embed_shortcut_alpha_query = None

        # 原有投影（保留，兼容你的外部 fallback 调用）
        self.project = nn.Sequential(
            nn.LayerNorm(gnn_out_ch),
            nn.GELU(),
            nn.Linear(gnn_out_ch, D),
            nn.Dropout(dropout),
        )
        nn.init.xavier_uniform_(self.project[2].weight)
        nn.init.zeros_(self.project[2].bias)

        # ★ Transformer：改为 pre-norm（norm_first=True），更稳定不易塌到常数
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=transformer_heads, dim_feedforward=transformer_ffn_dim,
            batch_first=True, dropout=dropout, activation="gelu", norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=transformer_heads, dim_feedforward=transformer_ffn_dim,
            batch_first=True, dropout=dropout, activation="gelu", norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=transformer_layers)

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        self.query_token = nn.Parameter(torch.zeros(1, 1, D))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.query_token, std=0.02)

        self.mem_norm   = nn.LayerNorm(D)
        self.query_norm = nn.LayerNorm(D)

        # ★ Head：移除首个 LayerNorm，保留可分性；最后一层仍是 Linear，兼容 set_head_bias_to_mu()
        self.head = nn.Sequential(
            nn.Linear(D, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        nn.init.xavier_uniform_(self.head[0].weight); nn.init.zeros_(self.head[0].bias)
        nn.init.xavier_uniform_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

        # 这些属性用于外部推断尺寸（可选）
        self.num_vertices = num_vertices
        self.num_labels   = num_labels

    def forward_memory_token_from_subgraph(self, vertex_ids, labels, degree, edge_index=None, batch: torch.Tensor | None = None):
        embed_x = self.embed.forward_data(vertex_ids, labels, degree)  # [N,C]
        x = self.gnn_encoder_data(embed_x, edge_index)                 # [N,C]
        if self.enable_embed_shortcut:
            shortcut = self.shortcut_proj_data(embed_x)
            gate = self._shortcut_gate_act(self.embed_shortcut_alpha_data)
            x = gate * x + (1.0 - gate) * shortcut
        # ★ 池化前做可学习缩放，促使注意力分布更尖锐
        x = x * F.softplus(self._pool_scale_data)
        pooled = self.pool_data(x)                               # [1,C]
        return self.project(pooled)                              # [1,D]

    def forward_query_token(self, labels, degree, edge_index=None, batch: torch.Tensor | None = None):
        embed_q = self.embed.forward_query(labels, degree)            # [N,C]
        xq = self.gnn_encoder_query(embed_q, edge_index)              # [N,C]
        if self.enable_embed_shortcut:
            shortcut = self.shortcut_proj_query(embed_q)
            gate = self._shortcut_gate_act(self.embed_shortcut_alpha_query)
            xq = gate * xq + (1.0 - gate) * shortcut
        xq = xq * F.softplus(self._pool_scale_query)
        pooled = self.pool_query(xq)                             # [1,C]
        return self.project(pooled)                              # [1,D]
