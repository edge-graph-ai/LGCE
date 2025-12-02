import torch
from torch import nn

class GraphEmbedding(nn.Module):
    def __init__(self, num_vertices_data, num_labels, embedding_dim, wildcard_label=-1):
        super().__init__()
        self.num_vertices_data = int(num_vertices_data)
        self.num_labels = int(num_labels)
        self.wildcard_label = int(wildcard_label)

        self.vertex_embed = nn.Embedding(self.num_vertices_data, embedding_dim)

        self.wildcard_idx = self.num_labels
        self.label_embed = nn.Embedding(self.num_labels + 1, embedding_dim)

        self.degree_lin = nn.Linear(1, embedding_dim)

    @torch.no_grad()
    def _sanitize_inputs(self, vertex_ids=None, labels=None, degrees=None, where="data"):
        if vertex_ids is not None:
            if vertex_ids.dtype != torch.long:
                vertex_ids = vertex_ids.long()

            vmax = int(vertex_ids.max().item()) if vertex_ids.numel() else -1
            if vmax >= self.num_vertices_data or int(vertex_ids.min().item()) < 0:
                raise RuntimeError(
                    f"[GraphEmbedding:{where}] vertex_ids out of range "
                    f"(min={int(vertex_ids.min())}, max={vmax}, table={self.num_vertices_data}). "
                    f"Ensure num_vertices_data covers the global ID range of the data graph and that subgraphs use global IDs."
                )
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()

            bad = (labels >= self.num_labels) & (labels != -1)
            if bad.any():

                labels = labels.clone()
                labels[bad] = -1
        if degrees is not None:
            degrees = degrees.to(torch.float)

            bad = ~torch.isfinite(degrees)
            if bad.any():
                degrees = degrees.clone()
                degrees[bad] = 1.0

            degrees.clamp_(min=1.0)
        return vertex_ids, labels, degrees

    def forward_data(self, vertex_ids, labels, degrees):

        with torch.no_grad():
            vertex_ids, labels, degrees = self._sanitize_inputs(vertex_ids, labels, degrees, where="data")

        vid_feat = self.vertex_embed(vertex_ids)                         # [N,D]
        mapped = torch.where(labels == -1, torch.full_like(labels, self.wildcard_idx), labels)
        lbl_feat = self.label_embed(mapped)                              # [N,D]
        deg_feat = self.degree_lin(degrees.unsqueeze(-1).to(torch.float))# [N,D]
        return vid_feat + lbl_feat + deg_feat

    def forward_query(self, labels, degrees):

        with torch.no_grad():
            _, labels, degrees = self._sanitize_inputs(None, labels, degrees, where="query")

        mapped = torch.where(labels == -1, torch.full_like(labels, self.wildcard_idx), labels)
        lbl_feat = self.label_embed(mapped)                               # [M,D]
        deg_feat = self.degree_lin(degrees.unsqueeze(-1).to(torch.float)) # [M,D]
        return lbl_feat + deg_feat
