r"""
本模块负责：
1) 从 prepared_pyg_data.pt 加载 PyG 图数据；
2) 解析 query_k（支持 Chain/Cycle/Tree/Star/Petal/Graph_* 与旧的 query_dense/sparse_*）；
3) 构建 Dataset 与数据划分；
4) 提供数据完整性体检与可选清洗工具。

注意：
- 正则写成 raw-string，避免 \d \b 等转义告警。
- torch.load 显式 weights_only=False，避免未来 PyTorch 版本默认改变引发兼容问题。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

try:
    # 仅用于体检的类型标注，不强依赖也不影响运行
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover
    Data = Any  # fallback

from src.data_preprocessing.estimatedataset import EstimateDataset


# ======== Query-K 解析：兼容多种命名 ========
# 1) 旧格式：query_(dense|sparse)_(\d+)_
_QUERY_K_PAT_OLD = re.compile(r"query_(?:dense|sparse)_(\d+)_", re.IGNORECASE)
# 2) 新格式：Chain/Cycle/Tree/Star/Petal/Graph_数字_...
_QUERY_K_PAT_TYPED = re.compile(r"^(?:chain|cycle|tree|star|petal|graph)_(12|3|6|9)\b", re.IGNORECASE)
# 3) 兜底（更严格）：从左到右找第一个独立的 12/3/6/9（被 _ 或边界分隔）
_QUERY_K_PAT_FIRST = re.compile(r"(?:^|_)(12|3|6|9)(?=(_|$))", re.IGNORECASE)
# 4) 最后兜底（可选，宽松；若想严格，可删除此项）
_QUERY_K_PAT_ANY = re.compile(r"(12|3|6|9)", re.IGNORECASE)


def _extract_query_k(qid: str) -> int:
    """
    从 query_id/文件名中抽取 query_k（∈ {3,6,9,12}），按优先级匹配：
      1) query_(dense|sparse)_(\d+)_
      2) ^(Chain|Cycle|Tree|Star|Petal|Graph)_(12|3|6|9)\b
      3) 独立片段的 12/3/6/9（用 '_' 或边界分隔）
      4) 宽松匹配第一个 12/3/6/9（如不想宽松，删除这一步）
    """
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
    """
    读取 prepare_data.py 保存的 pack，并派生 query_k_list。
    返回字段：
      - pyg_data_graphs: List[List[PyG Data]]（多次 repeat 的子图划分）
      - pyg_query_graphs: List[PyG Data]
      - true_cardinalities: List[int]
      - query_ids: List[str]
      - query_k_list: List[int]
    """
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


# ======== 划分工具 ========
def make_splits_indices(
    prepared: Dict[str, Any],
    selected_query_num: int,
    pretrain_ratio: float = 0.2,
    k_folds: int = 5,
    seed: int = 42,
    exclude_overlap: bool = True,
) -> Dict[str, Any]:
    """
    - 全局预训练：随机从【所有查询】里抽取 pretrain_ratio
    - 微调/评估：只在指定 query_num 的样本上做 K 折；默认把已用于预训练的样本剔除，避免泄露
    """
    n = len(prepared["query_ids"])
    rng = np.random.RandomState(seed)
    all_indices = np.arange(n)

    # 20% 全局预训练
    pre_n = max(1, int(n * pretrain_ratio))
    pretrain_idx = rng.permutation(all_indices)[:pre_n].tolist()

    # 只取指定的 query_k = selected_query_num
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

    # 打乱并等分成 K 份
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
    """
    DataLoader 打包：
      - 'pretrain_loader': 预训练子集
      - 'fold_loaders'   : 每折一个 {'train': ..., 'val': ...}
    """
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
    """
    打印一个可人工核对的预览字符串（展示数量和部分样本名）。
    """
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


# ======== 数据体检与可选清洗（训练前调用，定位/处理越界） ========
def _check_one_graph(g: "Data") -> List[str]:
    errs: List[str] = []
    # 推断节点数
    n = None
    if hasattr(g, "x") and g.x is not None:
        n = int(g.x.size(0))
    elif hasattr(g, "num_nodes") and g.num_nodes is not None:
        n = int(g.num_nodes)
    # 边索引检查
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
    """对 data 子图与 query 图进行一致性检查；发现越界立即打印具体样本。"""
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
    """
    可选：把体检失败的 query 图从 pack 中剔除（避免训练期崩溃）。
    返回：剔除的数量。
    注意：data 子图如果坏，建议回到预处理阶段修复，而不是这里静默更改拓扑。
    """
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
    # data 子图仅报告
    bad_data = sum(1 for g in prepared["pyg_data_graphs"][repeat_idx] if _check_one_graph(g))
    if bad_data:
        print(f"[WARN] detected {bad_data} bad data subgraphs; please fix preprocessing at source.")
    return removed
