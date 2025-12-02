import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, GraphNorm, JumpingKnowledge


class _GraphOrLayerNorm(nn.Module):
    
    def __init__(self, hidden_ch: int):
        super().__init__()
        self.gn = GraphNorm(hidden_ch)
        self.ln = nn.LayerNorm(hidden_ch)

    def forward(self, x, batch=None):
        if batch is None:
            return self.ln(x)
        return self.gn(x, batch)


class GINEncoder(nn.Module):
    
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        num_layers: int,
        dropout: float = 0.1,
        jk_mode: str = "sum",       
        residual: bool = True,
        train_eps: bool = True,     
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
                nn.ReLU(),                  
                nn.Linear(hidden_ch, hidden_ch),
            )
            self.convs.append(GINConv(nn=mlp, train_eps=train_eps))
            self.norms.append(_GraphOrLayerNorm(hidden_ch))
            self.drops.append(nn.Dropout(dropout))


        self.res_proj0 = None
        if residual and in_ch != hidden_ch:
            self.res_proj0 = nn.Linear(in_ch, hidden_ch)
            nn.init.xavier_uniform_(self.res_proj0.weight)
            nn.init.zeros_(self.res_proj0.bias)

        
        pyg_jk_modes = {"cat", "max", "lstm"}
        if jk_mode in pyg_jk_modes:
            self.jk = JumpingKnowledge(jk_mode)
        else:
            assert jk_mode in {"sum", "mean", "last"}, \
                f"Unsupported jk_mode: {jk_mode}. Choose from 'sum','mean','last','cat','max','lstm'."
            self.jk = None

       
        self.out_feat_dim = hidden_ch * num_layers if jk_mode == "cat" else hidden_ch


        self.proj = nn.Sequential(
            nn.LayerNorm(self.out_feat_dim),
            nn.Linear(self.out_feat_dim, out_ch),
        )
        nn.init.xavier_uniform_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x, edge_index, batch: torch.Tensor | None = None):
       
        xs = []
        h = x
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.drops)):
            h_conv = conv(h, edge_index)      
            h_conv = norm(h_conv, batch)      
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
                h_out = xs[-1]

        return self.proj(h_out)
