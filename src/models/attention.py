# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAttentionPool(nn.Module):
    """
    对一组 token（节点表示）进行注意力加权池化：
      input:  X  [N, D] 或 [B, N, D]
      output: y  [D]     或 [B, D]
    权重 = softmax( W2(GELU(W1 * x)) )，可学习到对关键节点的偏置。
    """
    def __init__(self, hidden_dim: int, attn_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj1 = nn.Linear(hidden_dim, attn_hidden)
        self.proj2 = nn.Linear(attn_hidden, 1)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

        # 小初始化，避免初期过于尖锐
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.5)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x):
        if x.numel() == 0:
            # 空输入：返回零向量
            if x.dim() == 2:
                return torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
            else:
                return torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[WARN] TokenAttentionPool input contains NaN/Inf!")
            print("x =", x)
            x = torch.where(torch.isfinite(x), x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
        if x.dim() == 2:
            # [N, D]
            a = self.proj2(self.act(self.proj1(self.drop(x)))).squeeze(-1)  # [N]
            w = F.softmax(a, dim=0)
            return torch.sum(w.unsqueeze(-1) * x, dim=0, keepdim=True)      # [1, D]
        elif x.dim() == 3:
            # [B, N, D]
            a = self.proj2(self.act(self.proj1(self.drop(x)))).squeeze(-1)  # [B, N]
            w = F.softmax(a, dim=1)
            return torch.sum(w.unsqueeze(-1) * x, dim=1)                    # [B, D]
        else:
            raise ValueError(f"TokenAttentionPool expects [N,D] or [B,N,D], got {x.shape}")


class MultiScaleTokenPool(nn.Module):
    """Combine TokenAttentionPool with a secondary pooling branch (mean-pool).

    When ``enabled`` is True, the module fuses the attention pooled feature with a
    global average pooled feature either via a sigmoid-gated weighted sum or a
    learnable linear projection on the concatenation of both features.  When it is
    disabled, it behaves exactly like :class:`TokenAttentionPool`.
    """

    def __init__(self,
                 hidden_dim: int,
                 attn_hidden: int = 64,
                 dropout: float = 0.1,
                 *,
                 enabled: bool = True,
                 fusion: str = "gate",
                 gate_init: float = 0.0,
                 fusion_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_pool = TokenAttentionPool(hidden_dim, attn_hidden=attn_hidden, dropout=dropout)
        self.enabled = bool(enabled)
        self.fusion = (fusion or "gate").lower()
        if self.enabled:
            if self.fusion in {"gate", "gated", "sigmoid", "weighted"}:
                self._gate = nn.Parameter(torch.tensor(float(gate_init)))
                self._gate_act = nn.Sigmoid()
                self._fuse = None
            elif self.fusion in {"concat", "linear"}:
                self._gate = None
                self._gate_act = None
                self._fuse = nn.Linear(hidden_dim * 2, hidden_dim)
                nn.init.xavier_uniform_(self._fuse.weight)
                nn.init.zeros_(self._fuse.bias)
            else:
                raise ValueError(f"Unsupported fusion mode: {fusion}")
        else:
            self._gate = None
            self._gate_act = None
            self._fuse = None
        self._drop = nn.Dropout(fusion_dropout) if (fusion_dropout and fusion_dropout > 0) else nn.Identity()

    @staticmethod
    def _mean_pool(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.mean(dim=0, keepdim=True)
        if x.dim() == 3:
            return x.mean(dim=1)
        raise ValueError(f"MultiScaleTokenPool expects [N,D] or [B,N,D], got {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn_pool(x)
        if not self.enabled or x.numel() == 0:
            return attn

        mean = self._mean_pool(x)
        if self._gate is not None:
            gate = self._gate_act(self._gate)
            fused = gate * attn + (1.0 - gate) * mean
            return self._drop(fused)
        if self._fuse is not None:
            fused = torch.cat([attn, mean], dim=-1)
            fused = self._drop(fused)
            return self._fuse(fused)
        return attn
