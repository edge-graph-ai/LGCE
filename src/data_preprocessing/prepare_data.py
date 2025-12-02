#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import math
import random
import glob
from pathlib import Path

import numpy as np
import torch

# Fix random seeds for reproducible preprocessing.
random.seed(42); np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "train_data"

from src.data_preprocessing.graphdata_to_pyg_data import graphdata_to_pyg_data
from src.data_preprocessing.load_data import (
    load_graph_from_file,
    load_true_cardinalities_aligned,
)
from src.data_preprocessing.substrate_data_graph import (
    partition_graph_bfs,  
)


def _build_default_paths(dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT):
    """
    根据 dataset 名字和 data_root 构造默认的数据路径。
    data_root 默认指向: <PROJECT_ROOT>/data/train_data
    """
    data_root = Path(data_root).resolve()
    base = data_root / dataset


    print(data_root)
    print(base / "query_graph")

    return {
        "data_graph_filename": base / "data_graph" / f"{dataset}.graph",
        "query_graph_root":    base / "query_graph",
        "matches_path":        base / "matches_output.txt",
        "output_path":         base / "prepared_pyg_data.pt",
        "graph_info_out":      base / "prepared_pyg_data_graph.pt",
    }


def prepare_data(
    *,
    dataset: str = "wordnet",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    device: str = "cuda",


    data_graph_filename: str | Path | None = None,
    query_graph_root: str | Path | None = None,
    matches_path: str | Path | None = None,
    output_path: str | Path | None = None,


    subgraph_num: int = 8,
    repeats: int = 1,
    seed: int = 42,

    **kwargs,
):


    defaults = _build_default_paths(dataset, data_root)


    data_graph_filename = Path(data_graph_filename or defaults["data_graph_filename"])
    query_graph_root    = Path(query_graph_root    or defaults["query_graph_root"])
    matches_path        = Path(matches_path        or defaults["matches_path"])
    output_path         = Path(output_path         or defaults["output_path"])
    graph_info_out      = Path(defaults["graph_info_out"])


    print(f"[INFO] dataset      = {dataset}")
    print(f"[INFO] data_root    = {Path(data_root).resolve()}")
    print(f"[INFO] data_graph   = {data_graph_filename.resolve()}")
    print(f"[INFO] query_root   = {query_graph_root.resolve()}")
    print(f"[INFO] matches_path = {matches_path.resolve()}")
    print(f"[INFO] output_path  = {output_path.resolve()}")

    t0 = time.perf_counter()
    data_graph = load_graph_from_file(str(data_graph_filename))
    print(f"Elapsed time reading data graph: {time.perf_counter() - t0:.2f} s")


    t0 = time.perf_counter()
    query_files = glob.glob(str(query_graph_root / "**" / "*.graph"), recursive=True)
    query_files = sorted(query_files, key=lambda p: os.path.basename(p))
    print(f"Loading {len(query_files)} query graphs (sorted by filename)")


    query_ids = [os.path.splitext(os.path.basename(p))[0] for p in query_files]


    t1 = time.perf_counter()
    true_cardinalities = load_true_cardinalities_aligned(str(matches_path), query_ids)
    print(f"Elapsed time reading and aligning true cardinalities: {time.perf_counter() - t1:.2f} s")
    assert len(true_cardinalities) == len(query_ids), \
        f"Count mismatch: labels={len(true_cardinalities)} vs queries={len(query_ids)}"


    t0 = time.perf_counter()
    vertex_label_counts: dict[int, int] = {}
    for attrs in data_graph.vertices.values():
        lbl = attrs['label']
        vertex_label_counts[lbl] = vertex_label_counts.get(lbl, 0) + 1
    print(f"Elapsed time counting vertex label distribution: {time.perf_counter() - t0:.2f} s")


    t2 = time.perf_counter()
    query_graphs = [load_graph_from_file(p) for p in query_files]
    pyg_query_graphs = [graphdata_to_pyg_data(qg, device, "query graph") for qg in query_graphs]
    print(f"Elapsed time converting query graphs to PyG Data: {time.perf_counter() - t2:.2f} s")


    t3 = time.perf_counter()
    np.random.seed(seed)
    print(f"Partitioning into {subgraph_num} subgraphs, repeated {repeats} times")
    data_graphs = [partition_graph_bfs(data_graph, subgraph_num) for _ in range(repeats)]

    pyg_data_graphs = []
    for subgraphs in data_graphs:
        pyg_subgraphs = []
        for subg in subgraphs:
            if subg.num_vertices == 0:
                continue
            pyg_subgraphs.append(graphdata_to_pyg_data(subg, device, "data graph"))
        pyg_data_graphs.append(pyg_subgraphs)
    print(f"Elapsed time partitioning data graphs: {time.perf_counter() - t3:.2f} s")


    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'pyg_data_graphs': pyg_data_graphs,        # [repeats][subgraph_num]
        'pyg_query_graphs': pyg_query_graphs,      # [num_queries]
        'true_cardinalities': true_cardinalities,  # [num_queries]
        'query_ids': query_ids,                    # [num_queries]
        'dataset': dataset,
    }, str(output_path))
    print(f"All processed results saved to {output_path}")


    sample_pyg_data_graphs = [pyg_data_graphs[0]]
    graph_info_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'pyg_data_graphs': sample_pyg_data_graphs,
        'num_vertices_data': data_graph.num_vertices,
        'vertex_label_counts': vertex_label_counts,
        'dataset': dataset,
    }, str(graph_info_out))
    print(f"Data graph snapshot saved to {graph_info_out} (single partition)")

    return pyg_data_graphs, pyg_query_graphs, true_cardinalities


if __name__ == "__main__":

    DATASET = "hprd"

    prepare_data(
        dataset=DATASET,
        device="cuda",
        subgraph_num=8,
        repeats=1,
        seed=42,
    )
