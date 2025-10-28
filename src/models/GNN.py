import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, GraphNorm, JumpingKnowledge


class _GraphOrLayerNorm(nn.Module):
    """优先用 GraphNorm(batch)；无 batch 时退化为 LayerNorm。"""
    def __init__(self, hidden_ch: int):
        super().__init__()
        self.gn = GraphNorm(hidden_ch)
        self.ln = nn.LayerNorm(hidden_ch)

    def forward(self, x, batch=None):
        if batch is None:
            return self.ln(x)
        return self.gn(x, batch)


class GINEncoder(nn.Module):
    """
    增强版 GINEncoder（依赖 torch_geometric）
    - GINConv(MLP) 保留 ReLU
    - 每层后：GraphNorm(有 batch)/LayerNorm(无 batch) -> GELU -> Dropout
    - 残差连接（首层线性短接）
    - JK 聚合：
        * 本地实现：'sum' | 'mean' | 'last'
        * 使用 PyG JumpingKnowledge：'cat' | 'max' | 'lstm'
    - 末端：LayerNorm + 线性 -> out_ch
    """
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        num_layers: int,
        dropout: float = 0.1,
        jk_mode: str = "sum",       # 'sum' | 'mean' | 'last' | 'cat' | 'max' | 'lstm'
        residual: bool = True,
        train_eps: bool = True,     # GIN 的 eps 可学习
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.out_ch = out_ch
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.use_residual = residual
        self.jk_mode = jk_mode = (jk_mode or "last").lower()

        # --- 堆叠 GIN 层 ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_ch if i == 0 else hidden_ch
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_ch),
                nn.ReLU(),                  # GIN MLP 内部 ReLU
                nn.Linear(hidden_ch, hidden_ch),
            )
            self.convs.append(GINConv(nn=mlp, train_eps=train_eps))
            self.norms.append(_GraphOrLayerNorm(hidden_ch))
            self.drops.append(nn.Dropout(dropout))

        # 残差短接（首层匹配维度）
        self.res_proj0 = None
        if residual and in_ch != hidden_ch:
            self.res_proj0 = nn.Linear(in_ch, hidden_ch)
            nn.init.xavier_uniform_(self.res_proj0.weight)
            nn.init.zeros_(self.res_proj0.bias)

        # JK 聚合：仅对 PyG 支持的模式构建模块，其余在 forward 中本地实现
        pyg_jk_modes = {"cat", "max", "lstm"}
        if jk_mode in pyg_jk_modes:
            self.jk = JumpingKnowledge(jk_mode)
        else:
            assert jk_mode in {"sum", "mean", "last"}, \
                f"Unsupported jk_mode: {jk_mode}. Choose from 'sum','mean','last','cat','max','lstm'."
            self.jk = None

        # 输出维度（cat 会乘以层数）
        self.out_feat_dim = hidden_ch * num_layers if jk_mode == "cat" else hidden_ch

        # 末端投影
        self.proj = nn.Sequential(
            nn.LayerNorm(self.out_feat_dim),
            nn.Linear(self.out_feat_dim, out_ch),
        )
        nn.init.xavier_uniform_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x, edge_index, batch: torch.Tensor | None = None):
        """
        x: [N, in_ch]
        edge_index: [2, E]
        batch: [N] 或 None（无则自动用 LayerNorm）
        return: [N, out_ch]
        """
        xs = []
        h = x
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.drops)):
            h_conv = conv(h, edge_index)      # 内含 MLP(Linear-ReLU-Linear)
            h_conv = norm(h_conv, batch)      # GraphNorm 或 LayerNorm
            h_conv = F.gelu(h_conv)
            h_conv = drop(h_conv)

            if self.use_residual:
                if i == 0:
                    res = self.res_proj0(h) if self.res_proj0 is not None else h
                else:
                    res = h
                h = h_conv + res
            else:
                h = h_conv
            xs.append(h)

        # JK 聚合
        if self.jk is not None:               # cat / max / lstm
            h_out = self.jk(xs)
        else:
            if self.jk_mode == "last":
                h_out = xs[-1]
            elif self.jk_mode == "sum":
                h_out = torch.stack(xs, dim=0).sum(dim=0)
            elif self.jk_mode == "mean":
                h_out = torch.stack(xs, dim=0).mean(dim=0)
            else:
                # 理论到不了这里，兜底
                h_out = xs[-1]

        return self.proj(h_out)
