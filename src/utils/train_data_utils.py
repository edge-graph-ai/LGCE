"""Utility helpers for training data handling and tensor safety."""
import glob
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from src.data_preprocessing.create_dataloader import create_dataloader
from src.models.estimator import GraphCardinalityEstimatorMultiSubgraph

USE_AMP = False


def resolve_query_dir(query_root: str, query_num_dir: int | None, query_dir: str | None) -> str:
    """Resolve the directory that contains query graphs and validate availability."""
    if query_dir:
        qdir = query_dir
        print(f"[QueryDir] 使用显式目录: {qdir}")
    elif query_num_dir is not None:
        qdir = os.path.join(query_root, f"query_{query_num_dir}")
        print(f"[QueryDir] 由 QUERY_NUM_DIR 计算得到: {qdir}")
    else:
        qdir = query_root
        print(f"[QueryDir] 未指定 QUERY_DIR/QUERY_NUM_DIR，使用根目录（递归读取）: {qdir}")
    if not os.path.isdir(qdir):
        raise RuntimeError(f"查询目录不存在: {qdir}")
    n_graphs = len(glob.glob(os.path.join(qdir, "**", "*.graph"), recursive=True))
    if n_graphs == 0:
        raise RuntimeError(f"查询目录下未发现 .graph 文件: {qdir}")
    print(f"[QueryDir] 发现 {n_graphs} 个 .graph 查询文件")
    return qdir


def clip_gradients(model, max_norm: float = 10.0) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def sanitize_log1p_pred(log1p_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(log1p_pred, min=eps)


def check_invalid_data(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def _sanitize_tensor_(t: torch.Tensor, is_degree: bool = False) -> torch.Tensor:
    if t is None or not torch.is_tensor(t):
        return t
    if is_degree:
        t = t.float()
        t = torch.clamp(t, min=1.0)
    if t.is_floating_point():
        bad_mask = ~torch.isfinite(t)
        if bad_mask.any():
            t = t.clone()
            t[bad_mask] = 1.0 if is_degree else 0.0
    return t


def _filter_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return edge_index
    src, dst = edge_index[0], edge_index[1]
    mask = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
    if mask.all():
        return edge_index
    return edge_index[:, mask]


def _get_embed_fields(ge) -> Dict[str, Any]:
    return dict(
        vertex=getattr(ge, "vertex_embed", None) or getattr(ge, "vertex_emb", None),
        label=getattr(ge, "label_embed", None) or getattr(ge, "label_emb", None),
        wildcard_idx=getattr(ge, "wildcard_idx", None),
        num_labels=getattr(ge, "num_labels", None),
    )


def _infer_bounds_from_model_or_batch(model, subgraphs_batch, queries) -> Tuple[int, int]:
    nv = getattr(model, "num_vertices", None) or getattr(model, "num_vertices_data", None)
    nl = getattr(model, "num_labels", None) or getattr(model, "num_labels_data", None)

    ge = getattr(model, "embed", None)
    if ge is not None:
        fields = _get_embed_fields(ge)
        ve, le = fields["vertex"], fields["label"]
        if nv is None and isinstance(ve, nn.Embedding):
            nv = int(ve.num_embeddings)
        if nl is None and isinstance(le, nn.Embedding):
            total = int(le.num_embeddings)
            nl = total - 1

    if nv is None:
        vmax = 0
        for subs in subgraphs_batch:
            for g in subs:
                if "vertex_ids" in g and g["vertex_ids"].numel() > 0:
                    vmax = max(vmax, int(g["vertex_ids"].max().item()))
        nv = vmax + 1
    if nl is None:
        lmax = 0
        for subs in subgraphs_batch:
            for g in subs:
                if "labels" in g and g["labels"].numel() > 0:
                    m = int(torch.clamp(g["labels"], min=0).max().item())
                    lmax = max(lmax, m)
        for q in queries:
            if "labels" in q and q["labels"].numel() > 0:
                m = int(torch.clamp(q["labels"], min=0).max().item())
                lmax = max(lmax, m)
        nl = lmax + 1

    return int(nv), int(nl)


def _maybe_resize_embeddings(model: GraphCardinalityEstimatorMultiSubgraph, num_vertices_needed: int | None,
                             num_labels_needed: int | None):
    ge = getattr(model, "embed", None)
    if ge is None:
        return
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    fields = _get_embed_fields(ge)
    ve = fields["vertex"]
    le = fields["label"]

    if isinstance(ve, nn.Embedding) and num_vertices_needed is not None:
        old = ve
        d = old.embedding_dim
        cur_n = old.num_embeddings
        need_n = int(num_vertices_needed)
        if need_n > cur_n:
            new = nn.Embedding(need_n, d).to(dev, dtype=dtype)
            with torch.no_grad():
                new.weight[:cur_n].copy_(old.weight)
                nn.init.xavier_uniform_(new.weight[cur_n:])
            if hasattr(ge, "vertex_emb"):
                ge.vertex_emb = new
            else:
                ge.vertex_embed = new
            print(f"[Resize] vertex_embedding: {cur_n} -> {need_n}")

    if isinstance(le, nn.Embedding) and num_labels_needed is not None:
        old = le
        d = old.embedding_dim
        cur_n = old.num_embeddings
        need_n = int(num_labels_needed) + 1
        if need_n > cur_n:
            new = nn.Embedding(need_n, d).to(dev, dtype=dtype)
            with torch.no_grad():
                new.weight[:cur_n].copy_(old.weight)
                nn.init.xavier_uniform_(new.weight[cur_n:])
            if hasattr(ge, "label_emb"):
                ge.label_emb = new
            else:
                ge.label_embed = new
            try:
                setattr(ge, "num_labels", int(num_labels_needed))
            except Exception:
                pass
            print(f"[Resize] label_embedding: {cur_n} -> {need_n}")


def _resize_to_fit_checkpoint(model, model_keys):
    nv, nl = _infer_bounds_from_model_or_batch(model, [], [])
    nv_ckpt = None
    nl_ckpt = None
    for k, v in model_keys.items():
        if "embed.vertex" in k and v.dim() == 2:
            nv_ckpt = v.size(0)
        if "embed.label" in k and v.dim() == 2:
            nl_ckpt = v.size(0) - 1
    nv = max(nv or 0, nv_ckpt or 0)
    nl = max(nl or 0, nl_ckpt or 0)
    if nv or nl:
        _maybe_resize_embeddings(model, nv, nl)


def _clamp_ids_inplace(model: GraphCardinalityEstimatorMultiSubgraph, subgraphs_batch, queries):
    ge = getattr(model, "embed", None)
    if ge is None:
        return
    fields = _get_embed_fields(ge)
    ve, le = fields["vertex"], fields["label"]
    wildcard_idx = fields["wildcard_idx"]
    num_labels = fields["num_labels"]
    max_vid = int(getattr(model, "num_vertices_data", getattr(model, "num_vertices", 0)) or 0)
    max_label = int(num_labels if num_labels is not None else 0)
    if isinstance(ve, nn.Embedding):
        max_vid = max(max_vid, int(ve.num_embeddings) - 1)
    if isinstance(le, nn.Embedding):
        max_label = max(max_label, int(le.num_embeddings) - 2)
    for subs in subgraphs_batch:
        for g in subs:
            if "vertex_ids" in g and torch.is_tensor(g["vertex_ids"]):
                g["vertex_ids"].clamp_(min=0, max=max_vid)
            if "labels" in g and torch.is_tensor(g["labels"]):
                g["labels"] = _sanitize_tensor_(g["labels"], is_degree=False)
                if isinstance(wildcard_idx, int) and wildcard_idx >= 0:
                    g["labels"].clamp_(min=0, max=max_label + 1)
                else:
                    g["labels"].clamp_(min=0, max=max_label)
            if "degree" in g and torch.is_tensor(g["degree"]):
                g["degree"] = _sanitize_tensor_(g["degree"], is_degree=True)
            if "edge_index" in g and torch.is_tensor(g["edge_index"]):
                g["edge_index"] = _filter_edge_index(g["edge_index"], int(g.get("labels", torch.tensor([])).size(0)))
    for q in queries:
        if "labels" in q and torch.is_tensor(q["labels"]):
            q["labels"] = _sanitize_tensor_(q["labels"], is_degree=False)
            if isinstance(wildcard_idx, int) and wildcard_idx >= 0:
                q["labels"].clamp_(min=0, max=max_label + 1)
            else:
                q["labels"].clamp_(min=0, max=max_label)
        if "degree" in q and torch.is_tensor(q.get("degree")):
            q["degree"] = _sanitize_tensor_(q["degree"], is_degree=True)
        if "edge_index" in q and torch.is_tensor(q.get("edge_index")):
            q["edge_index"] = _filter_edge_index(q["edge_index"], int(q.get("labels", torch.tensor([])).size(0)))


def move_batch_to_device(data_graph_batch, query_batch, y, device):
    def to_dev(x):
        return x.to(device) if torch.is_tensor(x) else x

    y = to_dev(y)
    new_dg = [[{k: to_dev(v) for k, v in g.items()} for g in subs] for subs in data_graph_batch]
    new_q = [{k: to_dev(v) for k, v in q.items()} for q in query_batch]
    return new_dg, new_q, y


def build_weighted_loader(dataset, indices, batch_size, seed, strata_bins, replacement=True, num_workers=0,
                          pin_memory=False):
    import numpy as np

    y = np.array([dataset.true_cardinalities[i] for i in indices], dtype=np.float64)
    qids = [dataset.query_ids[i] for i in indices]
    logy = np.log1p(y)
    bin_ids = _digitize_by_quantiles(logy, strata_bins)
    dens_ids = np.array([_parse_density_from_qid(qid) for qid in qids], dtype=int)

    from collections import Counter

    layer_keys = [(int(b), int(d)) for b, d in zip(bin_ids, dens_ids)]
    cnt = Counter(layer_keys)
    weights = np.array([1.0 / cnt[k] for k in layer_keys], dtype=np.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    gen = torch.Generator().manual_seed(int(seed) + 999)
    try:
        sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=replacement, generator=gen)
    except TypeError:
        torch.manual_seed(int(seed) + 999)
        sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=replacement)

    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        sampler=sampler,
        indices=indices,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def _parse_density_from_qid(qid: str) -> int:
    low = qid.lower()
    if "dense" in low:
        return 1
    if "sparse" in low:
        return 0
    return -1


def _digitize_by_quantiles(values: torch.Tensor | Any, n_bins: int):
    import numpy as np

    qs = np.linspace(0, 1, n_bins + 1)
    cuts = np.quantile(values, qs)
    cuts[0], cuts[-1] = -np.inf, np.inf
    bins = np.digitize(values, cuts[1:-1], right=True)
    return bins.astype(int)
