# src/data_preprocessing/graphdata_to_pyg_data.py
import torch
import numpy as np
from torch_geometric.data import Data

def _edge_pairs_iterator(edge_index):
    """
    将各种常见 edge_index 规格统一为 (u,v) 迭代器：
    - 列表/元组形如 [src_list, dst_list] 且 len==2
    - numpy.ndarray / torch.Tensor 形如 (2, E)
    - 已经是 [(u,v), ...] 的列表
    """
    # 1) list/tuple of length 2 -> [src_list, dst_list]
    if isinstance(edge_index, (list, tuple)) and len(edge_index) == 2:
        src, dst = edge_index
        # 支持 numpy / torch / list
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

    # 2) numpy.ndarray 或 torch.Tensor，形状(2, E)
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

    # 3) list of pairs
    if isinstance(edge_index, (list, tuple)) and len(edge_index) > 0 and isinstance(edge_index[0], (list, tuple)):
        for pair in edge_index:
            if len(pair) != 2:
                raise ValueError(f"edge pair must be length-2, got {pair}")
            u, v = pair
            yield int(u), int(v)
        return

    raise TypeError(f"Unsupported edge_index type/shape: {type(edge_index)} (example: {str(edge_index)[:80]}...)")

def graphdata_to_pyg_data(graphdata, device, data_type: str):
    """
    将自定义 graphdata 转成 PyG Data，并把 edge_index 映射为“本地 0..N-1”索引域。
    - data_type == "data graph": 保留 vertex_ids（原始全局ID）字段
    - data_type == "query graph": 不需要 vertex_ids
    graphdata 要求：
      - graphdata.vertices: dict[vid] -> {'original_id','label','degree'}
      - graphdata.edge_index: 支持多种形态（见 _edge_pairs_iterator）
    """
    # 1) 以顶点键的有序性确定“本地节点顺序 0..N-1”
    local_nodes = sorted(graphdata.vertices.keys())
    vid2loc = {int(v): i for i, v in enumerate(local_nodes)}
    N = len(local_nodes)

    node_vertices = []
    node_labels   = []
    node_degrees  = []

    for vid in local_nodes:
        attrs = graphdata.vertices[vid]
        if data_type == "data graph":
            # 记录“原始全局ID”给嵌入层使用
            node_vertices.append(int(attrs['original_id']))
        node_labels.append(int(attrs['label']))
        node_degrees.append(int(attrs['degree']))

    # 2) 将边从“原 vid 域”（可能是原始ID或旧本地ID）映射到“新本地 0..N-1”
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
        # 没有有效边也要构造一个空的 edge_index，避免后续算子崩溃
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src_mapped, dst_mapped], dtype=torch.long, device=device)

    # 3) 构造张量
    node_labels_t  = torch.tensor(node_labels,  dtype=torch.long,  device=device)
    node_degrees_t = torch.tensor(node_degrees, dtype=torch.float, device=device)  # 度当作 float
    data_kwargs = dict(edge_index=edge_index, labels=node_labels_t, degree=node_degrees_t)

    if data_type == "data graph":
        node_vertices_t = torch.tensor(node_vertices, dtype=torch.long, device=device)
        data_kwargs["vertex_ids"] = node_vertices_t

        # 额外安全检查：vertex_ids >= 0
        if node_vertices_t.numel() and int(node_vertices_t.min().item()) < 0:
            raise RuntimeError(f"[graphdata_to_pyg_data] negative vertex_id detected.")

    pyg = Data(**data_kwargs)

    # 4)（可选）打印一次丢边统计，确认数据健康（默认关闭）
    # print(f"[graphdata_to_pyg_data] N={N} E_in={len(list(_edge_pairs_iterator(graphdata.edge_index)))} "
    #       f"E_kept={edge_index.size(1)} dropped={drop_cnt} ({drop_cnt/max(1, edge_index.size(1)+drop_cnt):.1%}) "
    #       f"type={data_type}")

    return pyg
