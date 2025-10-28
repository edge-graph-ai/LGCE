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
