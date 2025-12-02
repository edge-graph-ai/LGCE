# src/data_preprocessing/graphdata_to_pyg_data.py
import torch
import numpy as np
from torch_geometric.data import Data

def _edge_pairs_iterator(edge_index):


    if isinstance(edge_index, (list, tuple)) and len(edge_index) == 2:
        src, dst = edge_index

        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
        if isinstance(dst, torch.Tensor):
            dst = dst.detach().cpu().numpy()
        src = np.asarray(src)
        dst = np.asarray(dst)
        assert src.shape == dst.shape
        for u, v in zip(src.tolist(), dst.tolist()):
            yield int(u), int(v)
        return


    if isinstance(edge_index, np.ndarray) and edge_index.ndim == 2 and edge_index.shape[0] == 2:
        src = edge_index[0, :].tolist()
        dst = edge_index[1, :].tolist()
        for u, v in zip(src, dst):
            yield int(u), int(v)
        return

    if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 and edge_index.size(0) == 2:
        ei = edge_index.detach().cpu().long()
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for u, v in zip(src, dst):
            yield int(u), int(v)
        return


    if isinstance(edge_index, (list, tuple)) and len(edge_index) > 0 and isinstance(edge_index[0], (list, tuple)):
        for pair in edge_index:
            if len(pair) != 2:
                raise ValueError(f"edge pair must be length-2, got {pair}")
            u, v = pair
            yield int(u), int(v)
        return

    raise TypeError(f"Unsupported edge_index type/shape: {type(edge_index)} (example: {str(edge_index)[:80]}...)")

def graphdata_to_pyg_data(graphdata, device, data_type: str):


    local_nodes = sorted(graphdata.vertices.keys())
    vid2loc = {int(v): i for i, v in enumerate(local_nodes)}
    N = len(local_nodes)

    node_vertices = []
    node_labels   = []
    node_degrees  = []

    for vid in local_nodes:
        attrs = graphdata.vertices[vid]
        if data_type == "data graph":

            node_vertices.append(int(attrs['original_id']))
        node_labels.append(int(attrs['label']))
        node_degrees.append(int(attrs['degree']))


    src_mapped, dst_mapped = [], []
    drop_cnt = 0
    for (u, v) in _edge_pairs_iterator(graphdata.edge_index):
        iu = vid2loc.get(int(u), -1)
        iv = vid2loc.get(int(v), -1)
        if iu == -1 or iv == -1:
            drop_cnt += 1
            continue
        src_mapped.append(iu)
        dst_mapped.append(iv)

    if len(src_mapped) == 0:

        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src_mapped, dst_mapped], dtype=torch.long, device=device)


    node_labels_t  = torch.tensor(node_labels,  dtype=torch.long,  device=device)
    node_degrees_t = torch.tensor(node_degrees, dtype=torch.float, device=device)  
    data_kwargs = dict(edge_index=edge_index, labels=node_labels_t, degree=node_degrees_t)

    if data_type == "data graph":
        node_vertices_t = torch.tensor(node_vertices, dtype=torch.long, device=device)
        data_kwargs["vertex_ids"] = node_vertices_t


        if node_vertices_t.numel() and int(node_vertices_t.min().item()) < 0:
            raise RuntimeError(f"[graphdata_to_pyg_data] negative vertex_id detected.")

    pyg = Data(**data_kwargs)

    return pyg
