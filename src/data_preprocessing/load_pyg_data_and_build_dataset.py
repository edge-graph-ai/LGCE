
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

try:
    
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover
    Data = Any  # fallback

from src.data_preprocessing.estimatedataset import EstimateDataset




_QUERY_K_PAT_OLD = re.compile(r"query_(?:dense|sparse)_(\d+)_", re.IGNORECASE)

_QUERY_K_PAT_TYPED = re.compile(r"^(?:chain|cycle|tree|star|petal|graph)_(12|3|6|9)\b", re.IGNORECASE)

_QUERY_K_PAT_FIRST = re.compile(r"(?:^|_)(12|3|6|9)(?=(_|$))", re.IGNORECASE)

_QUERY_K_PAT_ANY = re.compile(r"(12|3|6|9)", re.IGNORECASE)


def _extract_query_k(qid: str) -> int:

    name = Path(qid).name
    stem = Path(name).stem
    m = (
        _QUERY_K_PAT_OLD.search(stem)
        or _QUERY_K_PAT_TYPED.search(stem)
        or _QUERY_K_PAT_FIRST.search(stem)
        or _QUERY_K_PAT_ANY.search(stem)
    )
    if not m:
        raise ValueError(f"Cannot parse query number (3/6/9/12) from id: {qid}")
    return int(m.group(1))


# ======== 核心 I/O 接口 ========
def load_prepared_pyg(prepared_pt_path: str) -> Dict[str, Any]:

    pack = torch.load(prepared_pt_path, map_location="cpu", weights_only=False)
    pyg_data_graphs = pack["pyg_data_graphs"]     # [repeats][subgraph_num]
    pyg_query_graphs = pack["pyg_query_graphs"]   # [num_queries]
    true_cardinalities = pack["true_cardinalities"]
    query_ids = pack["query_ids"]

    assert len(pyg_query_graphs) == len(true_cardinalities) == len(query_ids), \
        f"Length mismatch: queries={len(pyg_query_graphs)}, labels={len(true_cardinalities)}, ids={len(query_ids)}"

    query_k_list = [_extract_query_k(qid) for qid in query_ids]
    return {
        "pyg_data_graphs": pyg_data_graphs,
        "pyg_query_graphs": pyg_query_graphs,
        "true_cardinalities": true_cardinalities,
        "query_ids": query_ids,
        "query_k_list": query_k_list,
    }


def build_dataset_from_prepared(prepared: Dict[str, Any], repeat_idx: int = 0) -> EstimateDataset:
    data_subgraphs = prepared["pyg_data_graphs"][repeat_idx]
    dataset = EstimateDataset(
        data_pygs=data_subgraphs,
        query_pygs=prepared["pyg_query_graphs"],
        true_cardinalities=prepared["true_cardinalities"],
        query_ids=prepared["query_ids"],
    )
    return dataset



def make_splits_indices(
    prepared: Dict[str, Any],
    selected_query_num: int,
    pretrain_ratio: float = 0.2,
    k_folds: int = 5,
    seed: int = 42,
    exclude_overlap: bool = True,
) -> Dict[str, Any]:

    n = len(prepared["query_ids"])
    rng = np.random.RandomState(seed)
    all_indices = np.arange(n)


    pre_n = max(1, int(n * pretrain_ratio))
    pretrain_idx = rng.permutation(all_indices)[:pre_n].tolist()

    
    qk = prepared["query_k_list"]
    pool = [i for i in all_indices if qk[i] == int(selected_query_num)]
    if exclude_overlap:
        s = set(pretrain_idx)
        pool = [i for i in pool if i not in s]

    if len(pool) < k_folds:
        raise ValueError(
            f"Not enough samples ({len(pool)}) in query_{selected_query_num} for k={k_folds} folds "
            f"after exclude_overlap={exclude_overlap}. Reduce k_folds or disable exclusion."
        )

    
    pool = rng.permutation(pool).tolist()
    fold_sizes = [len(pool) // k_folds] * k_folds
    for i in range(len(pool) % k_folds):
        fold_sizes[i] += 1

    folds, start = [], 0
    fold_slices: List[List[int]] = []
    for sz in fold_sizes:
        fold_slices.append(pool[start:start + sz])
        start += sz
    for i in range(k_folds):
        val_idx = fold_slices[i]
        train_idx = [idx for j, sl in enumerate(fold_slices) if j != i for idx in sl]
        folds.append({"train_idx": train_idx, "val_idx": val_idx})

    return {
        "pretrain_idx": pretrain_idx,
        "folds": folds,
        "pool_counts": {
            "global_total": n,
            "pool_total": len([i for i in all_indices if qk[i] == int(selected_query_num)]),
        },
    }


def build_loaders_for_splits(
    dataset: EstimateDataset,
    splits: Dict[str, Any],
    batch_size_pretrain: int = 64,
    batch_size_finetune: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
) -> Dict[str, Any]:
    
    pretrain_subset = Subset(dataset, splits["pretrain_idx"])
    pretrain_loader = DataLoader(
        pretrain_subset,
        batch_size=batch_size_pretrain,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    fold_loaders = []
    for fold in splits["folds"]:
        tr = Subset(dataset, fold["train_idx"])
        va = Subset(dataset, fold["val_idx"])
        tr_loader = DataLoader(
            tr, batch_size=batch_size_finetune, shuffle=shuffle_train,
            num_workers=num_workers, pin_memory=pin_memory
        )
        va_loader = DataLoader(
            va, batch_size=batch_size_finetune, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        fold_loaders.append({"train": tr_loader, "val": va_loader})
    return {"pretrain_loader": pretrain_loader, "fold_loaders": fold_loaders}


def preview_splits(prepared: Dict[str, Any], splits: Dict[str, Any], k_preview: int = 5) -> str:

    qids = prepared["query_ids"]

    def sample_names(idxs: List[int]) -> str:
        return ", ".join(qids[i] for i in idxs[:k_preview])

    lines = []
    lines.append(f"Global queries: {len(qids)}")
    lines.append(f"Pretrain: {len(splits['pretrain_idx'])} examples. e.g., [{sample_names(splits['pretrain_idx'])}]")
    for fi, fold in enumerate(splits["folds"]):
        lines.append(
            f"Fold {fi + 1}: train={len(fold['train_idx'])}, val={len(fold['val_idx'])}. "
            f"val e.g., [{sample_names(fold['val_idx'])}]"
        )
    return "\n".join(lines)



def _check_one_graph(g: "Data") -> List[str]:
    errs: List[str] = []
    
    n = None
    if hasattr(g, "x") and g.x is not None:
        n = int(g.x.size(0))
    elif hasattr(g, "num_nodes") and g.num_nodes is not None:
        n = int(g.num_nodes)
    
    if hasattr(g, "edge_index") and g.edge_index is not None:
        ei = g.edge_index
        if not torch.is_tensor(ei):
            errs.append("edge_index is not a tensor")
        else:
            if ei.dtype != torch.long:
                errs.append(f"edge_index dtype={ei.dtype}, expect torch.long")
            if ei.numel() > 0:
                m = int(ei.min())
                M = int(ei.max())
                if m < 0:
                    errs.append(f"edge_index min={m} < 0")
                if n is not None and M >= n:
                    errs.append(f"edge_index max={M} >= num_nodes={n}")
    else:
        errs.append("missing edge_index")
    if n is None:
        errs.append("cannot infer num_nodes (no x and no num_nodes)")
    return errs


def verify_prepared_pack(prepared: Dict[str, Any], repeat_idx: int = 0, max_print: int = 20) -> None:
   
    print("[Verify] scanning graphs for out-of-range indices ...")
    bad = 0
    # data subgraphs
    for j, g in enumerate(prepared["pyg_data_graphs"][repeat_idx]):
        errs = _check_one_graph(g)
        if errs:
            bad += 1
            if bad <= max_print:
                print(f"[BAD] data[{j}]: " + "; ".join(errs))
    # query graphs
    for i, g in enumerate(prepared["pyg_query_graphs"]):
        errs = _check_one_graph(g)
        if errs:
            bad += 1
            if bad <= max_print:
                qid = prepared["query_ids"][i] if "query_ids" in prepared else f"#{i}"
                print(f"[BAD] query[{i}] ({qid}): " + "; ".join(errs))
    if bad == 0:
        print("[Verify] OK: no out-of-range indices.")
    else:
        print(f"[Verify] Found {bad} problematic graphs. Please fix preprocessing.")


def drop_bad_graphs_in_place(prepared: Dict[str, Any], repeat_idx: int = 0) -> int:
   
    keep: List[int] = []
    removed = 0
    for i, g in enumerate(prepared["pyg_query_graphs"]):
        if not _check_one_graph(g):
            keep.append(i)
        else:
            removed += 1
    if removed:
        prepared["pyg_query_graphs"] = [prepared["pyg_query_graphs"][i] for i in keep]
        prepared["true_cardinalities"] = [prepared["true_cardinalities"][i] for i in keep]
        prepared["query_ids"] = [prepared["query_ids"][i] for i in keep]
        prepared["query_k_list"] = [prepared["query_k_list"][i] for i in keep]
        print(f"[CLEAN] dropped {removed} bad query graphs.")
   
    bad_data = sum(1 for g in prepared["pyg_data_graphs"][repeat_idx] if _check_one_graph(g))
    if bad_data:
        print(f"[WARN] detected {bad_data} bad data subgraphs; please fix preprocessing at source.")
    return removed
