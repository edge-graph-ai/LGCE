# src/utils/splits_and_sampling.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import torch
from typing import Callable, Dict, List, Sequence, Tuple, Optional, Mapping
from collections import Counter
from torch.utils.data import WeightedRandomSampler

__all__ = [
    "make_splits_indices_stratified",
    "build_weighted_loader",
]

# ---------- small helpers (pure) ----------
def _parse_density_from_qid(qid: str) -> int:

    low = (qid or "").lower()
    if "dense" in low:  return 1
    if "sparse" in low: return 0
    return -1

def _digitize_by_quantiles(values: np.ndarray, n_bins: int) -> np.ndarray:

    if values.size == 0:
        return np.zeros(0, dtype=int)
    qs = np.linspace(0, 1, n_bins + 1)
    cuts = np.quantile(values, qs)

    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i-1]:
            cuts[i] = cuts[i-1] + 1e-12
    cuts[0], cuts[-1] = -np.inf, np.inf
    bins = np.digitize(values, cuts[1:-1], right=True)
    return bins.astype(int)

# ---------- public APIs (no external imports) ----------
def make_splits_indices_stratified(
    prepared: Dict,
    selected_query_num: int,
    pretrain_ratio: float,
    k_folds: int,
    seed: int,
    exclude_overlap: bool,
    n_bins: int,
):

    rng = np.random.RandomState(seed)
    qids = prepared['query_ids']
    qk_list = prepared['query_k_list']
    y_all = np.array(prepared['true_cardinalities'], dtype=np.float64)

    n = len(qids)
    all_indices = np.arange(n)


    pre_n = max(1, int(round(pretrain_ratio * n)))
    pretrain_idx = rng.permutation(all_indices)[:pre_n].tolist()


    pool = [i for i in all_indices if int(qk_list[i]) == int(selected_query_num)]
    if exclude_overlap:
        s = set(pretrain_idx)
        pool = [i for i in pool if i not in s]


    if len(pool) < k_folds:
        if exclude_overlap:
            pool = [i for i in all_indices if int(qk_list[i]) == int(selected_query_num)]
        if len(pool) < k_folds:
            k_folds = max(2, len(pool))
            if k_folds <= 1:
                raise ValueError("Not enough samples to perform K-fold even after relaxing constraints.")


    logy  = np.log1p(y_all[pool])
    bin_ids  = _digitize_by_quantiles(logy, n_bins)
    dens_ids = np.array([_parse_density_from_qid(qids[i]) for i in pool], dtype=int)


    strata = {}
    for li, gi in enumerate(pool):
        key = (int(bin_ids[li]), int(dens_ids[li]))
        strata.setdefault(key, []).append(gi)

    folds = [{'train_idx': [], 'val_idx': []} for _ in range(k_folds)]
    for _, idxs in strata.items():
        idxs = rng.permutation(idxs).tolist()
        base = len(idxs) // k_folds
        rem  = len(idxs) % k_folds
        sizes = [base + (1 if i < rem else 0) for i in range(k_folds)]
        cur = 0
        slices = []
        for sz in sizes:
            slices.append(idxs[cur:cur+sz]); cur += sz
        for f in range(k_folds):
            folds[f]['val_idx'].extend(slices[f])
            folds[f]['train_idx'].extend([x for j, sl in enumerate(slices) if j != f for x in sl])

    return {'pretrain_idx': pretrain_idx, 'folds': folds}

def build_weighted_loader(
    dataset,
    indices: Sequence[int],
    batch_size: int,
    seed: int,
    *,
    strata_num_bins: int = 5,
    replacement: bool = True,
    create_dl_fn: Callable = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    per_index_multipliers: Optional[Mapping[int, float]] = None,
):

    if create_dl_fn is None:
        raise RuntimeError("build_weighted_loader: create_dl_fn must be provided.")

    if len(indices) == 0:
        return create_dl_fn(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            indices=indices,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    idx_list = list(indices)
    y = np.array([dataset.true_cardinalities[i] for i in idx_list], dtype=np.float64)
    qids = [dataset.query_ids[i] for i in idx_list]
    logy = np.log1p(y)

    bin_ids  = _digitize_by_quantiles(logy, strata_num_bins)
    dens_ids = np.array([_parse_density_from_qid(qid) for qid in qids], dtype=int)

    layer_keys = [(int(b), int(d)) for b, d in zip(bin_ids, dens_ids)]
    cnt = Counter(layer_keys)
    weights = np.array([1.0 / cnt[k] for k in layer_keys], dtype=np.float32)
    if per_index_multipliers:
        multipliers = np.array(
            [float(per_index_multipliers.get(int(idx), 1.0)) for idx in idx_list], dtype=np.float32
        )
        weights *= multipliers
    weights = np.clip(weights, a_min=1e-8, a_max=None)
    weights = torch.tensor(weights, dtype=torch.float32)

    gen = torch.Generator().manual_seed(int(seed) + 999)
    try:
        sampler = WeightedRandomSampler(
            weights, num_samples=len(idx_list), replacement=replacement, generator=gen
        )
    except TypeError:
        torch.manual_seed(int(seed) + 999)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(idx_list), replacement=replacement
        )

    loader = create_dl_fn(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        sampler=sampler,
        indices=idx_list,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader
