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
                 multi_scale_attn_hidden: int | None = None,
                 enable_cross_attention: bool = True,
                 cross_attn_heads: int = 1,
                 cross_attn_dropout: float | None = None,
                 cross_ffn_hidden: int | None = None,
                 use_memory_positional_encoding: bool = True):
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

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        self.query_token = nn.Parameter(torch.zeros(1, 1, D))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.query_token, std=0.02)

        self.mem_norm   = nn.LayerNorm(D)
        self.query_norm = nn.LayerNorm(D)

        self.use_memory_positional_encoding = bool(use_memory_positional_encoding)
        if self.use_memory_positional_encoding:
            # +1 以防止序列长度刚好等于 num_subgraphs 时越界
            self.mem_pos_embedding = nn.Embedding(max(1, num_subgraphs + 1), D)
            nn.init.normal_(self.mem_pos_embedding.weight, std=0.02)
            self.register_buffer(
                "_mem_pos_indices",
                torch.arange(max(1, num_subgraphs + 1), dtype=torch.long),
                persistent=False,
            )
        else:
            self.mem_pos_embedding = None
            self.register_buffer("_mem_pos_indices", None, persistent=False)

        self.enable_cross_attention = bool(enable_cross_attention)
        heads = max(1, int(cross_attn_heads))
        cross_drop = dropout if cross_attn_dropout is None else float(cross_attn_dropout)
        if self.enable_cross_attention:
            self.cross_attn = nn.MultiheadAttention(D, heads, dropout=cross_drop, batch_first=True)
            self.cross_attn_norm = nn.LayerNorm(D)
        else:
            self.cross_attn = None
            self.cross_attn_norm = nn.Identity()

        hidden = cross_ffn_hidden if cross_ffn_hidden is not None else transformer_ffn_dim
        hidden = max(1, int(hidden))
        self.cross_ffn = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, D),
        )
        self.cross_ffn_norm = nn.LayerNorm(D)

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

    def apply_memory_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_memory_positional_encoding or self.mem_pos_embedding is None:
            return tokens
        if tokens.numel() == 0:
            return tokens
        max_idx = self.mem_pos_embedding.num_embeddings - 1
        if tokens.dim() == 3:
            B, S, _ = tokens.shape
            device = tokens.device
            if self._mem_pos_indices is None or int(self._mem_pos_indices.numel()) < S:
                pos = torch.arange(S, device=device)
            else:
                pos = self._mem_pos_indices[:S].to(device=device)
            pos = pos.clamp_max(max_idx)
            pos = pos.unsqueeze(0).expand(B, S)
        else:
            S = tokens.shape[0]
            device = tokens.device
            if self._mem_pos_indices is None or int(self._mem_pos_indices.numel()) < S:
                pos = torch.arange(S, device=device)
            else:
                pos = self._mem_pos_indices[:S].to(device=device)
            pos = pos.clamp_max(max_idx)
        pos_emb = self.mem_pos_embedding(pos)
        return tokens + pos_emb

    def cross_interact(self,
                       memory: torch.Tensor,
                       query: torch.Tensor,
                       memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = query
        if self.enable_cross_attention and self.cross_attn is not None:
            attn_out, _ = self.cross_attn(query, memory, memory, key_padding_mask=memory_key_padding_mask)
            x = self.cross_attn_norm(attn_out + query)
        else:
            x = self.cross_attn_norm(x)
        ff = self.cross_ffn(x)
        if isinstance(self.cross_ffn, nn.Identity):
            return self.cross_ffn_norm(ff)
        if isinstance(self.cross_ffn_norm, nn.Identity):
            return ff
        ff = self.cross_ffn_norm(ff + x)
        return ff

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D401 - docstring inherited
        # 兼容旧 checkpoint：忽略 TransformerDecoder 的权重
        if not isinstance(state_dict, dict):
            incompatible = super().load_state_dict(state_dict, strict=strict)
            return incompatible
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("transformer_decoder.")}
        if strict and len(filtered) != len(state_dict):
            strict = False
        if strict:
            new_prefixes = (
                "mem_pos_embedding.",
                "cross_attn.",
                "cross_attn_norm.",
                "cross_ffn.",
                "cross_ffn_norm.",
            )
            if not any(any(k.startswith(p) for k in filtered) for p in new_prefixes):
                strict = False
        incompatible = super().load_state_dict(filtered, strict=strict)

        if not hasattr(incompatible, "missing_keys") and not hasattr(incompatible, "unexpected_keys"):
            return incompatible

        prefix = "transformer_decoder."

        def _filter_keys(keys):
            if keys is None:
                return None, False
            original = list(keys)
            filtered_keys = [k for k in original if not k.startswith(prefix)]
            changed = len(filtered_keys) != len(original)
            return filtered_keys, changed

        missing = getattr(incompatible, "missing_keys", None)
        unexpected = getattr(incompatible, "unexpected_keys", None)

        filtered_missing, missing_changed = _filter_keys(missing)
        filtered_unexpected, unexpected_changed = _filter_keys(unexpected)

        if not missing_changed and not unexpected_changed:
            return incompatible

        if missing_changed and isinstance(missing, list):
            missing[:] = filtered_missing
        if unexpected_changed and isinstance(unexpected, list):
            unexpected[:] = filtered_unexpected

        need_replace = (
            (missing_changed and not isinstance(missing, list))
            or (unexpected_changed and not isinstance(unexpected, list))
        )
        if not need_replace:
            return incompatible

        def _value_for_replace(original, filtered, changed):
            if not changed:
                return original
            if original is None:
                return None
            if isinstance(original, list):
                return original
            try:
                return type(original)(filtered)
            except TypeError:
                return filtered

        new_missing = _value_for_replace(missing, filtered_missing, missing_changed)
        new_unexpected = _value_for_replace(unexpected, filtered_unexpected, unexpected_changed)

        if hasattr(incompatible, "_replace"):
            return incompatible._replace(missing_keys=new_missing, unexpected_keys=new_unexpected)
        return type(incompatible)(new_missing, new_unexpected)
