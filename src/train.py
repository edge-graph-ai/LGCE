# -*- coding: utf-8 -*-
import argparse
import os

from src.utils.splits_and_sampling import build_weighted_loader, make_splits_indices_stratified

from torch.optim.swa_utils import AveragedModel
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys, importlib, importlib.util, inspect, glob, time
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import WeightedRandomSampler

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for CLI config
    yaml = None

# --------- 解决 "No module named 'src'" ----------
THIS = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# 清掉其它路径里缓存的 src
for k in list(sys.modules.keys()):
    if k == "src" or k.startswith("src."):
        del sys.modules[k]
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import src
spec = importlib.util.find_spec("src")
src_dir = (os.path.abspath(spec.submodule_search_locations[0])
           if spec and spec.submodule_search_locations
           else os.path.abspath(getattr(src, "__file__", "") or SRC_PATH))
print("[DEBUG] using package 'src' from:", src_dir)
m = importlib.import_module("src.data_preprocessing.load_pyg_data_and_build_dataset")
try:
    print("[DEBUG] load_pyg_data_and_build_dataset.py:", inspect.getfile(m))
except Exception:
    pass

# ===== 项目内模块 =====
from src.data_preprocessing.load_data import load_graph_from_file
from src.data_preprocessing.prepare_data import prepare_data
from src.data_preprocessing.create_dataloader import create_dataloader
from src.data_preprocessing.load_pyg_data_and_build_dataset import (
    load_prepared_pyg, build_dataset_from_prepared,
)
from src.models.estimator import GraphCardinalityEstimatorMultiSubgraph
from src.utils.earlystop import EarlyStopping
# =====================

USE_AMP = True

# ----------------- 全局配置（按需手动改） -----------------
CONFIG = dict(
    DATASET        = "yeast",
    DATA_GRAPH     = None,
    MATCHES_PATH   = None,
    PREPARED_OUT   = None,
    QUERY_DIR      = None,       # 显式指定查询根目录（优先级更高）
    QUERY_ROOT     = None,       # 缺省使用 data/train_data/{ds}/query_graph
    QUERY_NUM_DIR  = None,       # 若设置，使用 query_root/query_{k}

    SELECTED_QUERY_NUM = 32,     # 仅用于 fine-tune 的查询大小（例如 4/8/12/...）

    DEVICE         = "cuda",
    SEED           = 42,
    BATCH_SIZE     = 32,
    LR             = 1e-4,
    WEIGHT_DECAY   = 1e-4,
    EPOCHS_PRE     = 100,
    EPOCHS_FT      = 100,
    PATIENCE       = 10,
    PRETRAIN_RATIO = 0.20,       # 20% 全部查询做预训练
    K_FOLDS        = 5,
    EXCLUDE_OVERLAP= True,
    NUM_WORKERS    = 0,
    PIN_MEMORY     = False,

    USE_MEDIAN_BIAS = True,      # 用 median(log1p(y)) 设 head.bias
    HEAD_LR_MULT    = 10.0,

    ENABLE_EMBED_SHORTCUT = True,
    EMBED_SHORTCUT_INIT   = 0.0,
    USE_MULTI_SCALE_POOL  = True,
    MULTI_SCALE_FUSION    = "gate",
    MULTI_SCALE_DROPOUT   = None,
    MULTI_SCALE_GATE_INIT = 0.0,
    MULTI_SCALE_ATTN_HIDDEN = None,

    W_MSLE     = 0.2,   # ↓
    W_QERR     = 0.6,   # ↑ 让相对误差成为主损失
    W_LOGHUBER = 0.2,   # ↑ 让鲁棒的方向性惩罚有存在感
    LOG_HUBER_DELTA = 0.25,   # 稍小一些，更像L1，拉开分布
    TRIM_RATIO     = 0.05 ,   # 少量修剪，留出“拉开的”梯度

    # 训练中 QErr 的裁剪上界，避免梯度爆炸；测试阶段不裁剪
    QERR_CLIP_MAX   = 1e4,


    STRATA_NUM_BINS = 5,
    USE_WEIGHTED_SAMPLER = True,
    SAMPLE_REPLACEMENT  = True,

    USE_CALIBRATOR      = False,
    CALIB_LR_MULT       = 15.0,
    WARMUP_HEAD_EPOCHS  = 5,     # 仅训练 head + calibrator 的轮数

    LR_WARMUP_EPOCHS    = 5,

    PRETRAIN_WEIGHTS   = "pretrained.pth",
    BEST_PRE_PATH      = "best_model_pretrain.pth",
    BEST_FOLD_PATH_TPL = "best_model_fold{fold}.pth",
    PRED_TXT_TPL       = "predictions_fold{fold}.txt",
    RESULT_DIR         = None,
)
# ---------------------------------------------------------

def _apply_dataset(C: dict) -> dict:
    ds = C["DATASET"]
    base = os.path.join("..", "data", "train_data", ds)
    C["DATA_GRAPH"]   = os.path.join(base, "data_graph", f"{ds}.graph")
    C["MATCHES_PATH"] = os.path.join(base, "matches_output.txt")
    C["PREPARED_OUT"] = os.path.join(base, "prepared_pyg_data.pt")
    if C.get("QUERY_DIR"):
        C["QUERY_ROOT"] = C["QUERY_DIR"]
    else:
        C["QUERY_ROOT"] = os.path.join(base, "query_graph")
    return C

def _bind_output_paths_to_selected_num(C: dict) -> dict:
    ds = C["DATASET"]
    n  = C["SELECTED_QUERY_NUM"]

    # 新增：从环境变量读取 run tag（每次调参自动传入）
    tag = os.getenv("RUN_TAG", "").strip()
    suffix = f"_{tag}" if tag else ""
    out_dir = os.path.join("result", ds, f"{n}{suffix}")  # <- 带上 suffix
    os.makedirs(out_dir, exist_ok=True)
    C["PRETRAIN_WEIGHTS"]   = os.path.join(out_dir, f"pretrained_q{n}.pth")
    C["BEST_PRE_PATH"]      = os.path.join(out_dir, f"best_model_pretrain_q{n}.pth")
    C["BEST_FOLD_PATH_TPL"] = os.path.join(out_dir, f"best_model_q{n}_fold{{fold}}.pth")
    C["PRED_TXT_TPL"]       = os.path.join(out_dir, f"predictions_q{n}_fold{{fold}}.txt")
    C["RESULT_DIR"]         = out_dir
    return C

CONFIG = _bind_output_paths_to_selected_num(_apply_dataset(CONFIG))

GLOBAL_NUM_VERTICES = None
GLOBAL_NUM_LABELS   = None


def _load_yaml_config(path: str) -> dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files. Please install 'pyyaml' or avoid using --config.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data)!r}")
    return data


def _apply_cli_overrides(args):
    if args is None:
        return
    if getattr(args, "config", None):
        CONFIG.update(_load_yaml_config(args.config))
    if getattr(args, "enable_shortcut", None) is not None:
        CONFIG["ENABLE_EMBED_SHORTCUT"] = bool(args.enable_shortcut)
    if getattr(args, "shortcut_init", None) is not None:
        CONFIG["EMBED_SHORTCUT_INIT"] = float(args.shortcut_init)
    if getattr(args, "enable_multi_scale", None) is not None:
        CONFIG["USE_MULTI_SCALE_POOL"] = bool(args.enable_multi_scale)
    if getattr(args, "multi_scale_fusion", None):
        CONFIG["MULTI_SCALE_FUSION"] = str(args.multi_scale_fusion)
    if getattr(args, "multi_scale_dropout", None) is not None:
        CONFIG["MULTI_SCALE_DROPOUT"] = float(args.multi_scale_dropout)
    if getattr(args, "multi_scale_gate_init", None) is not None:
        CONFIG["MULTI_SCALE_GATE_INIT"] = float(args.multi_scale_gate_init)
    if getattr(args, "multi_scale_attn_hidden", None) is not None:
        CONFIG["MULTI_SCALE_ATTN_HIDDEN"] = int(args.multi_scale_attn_hidden)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Graph Cardinality Estimator")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file to override defaults")
    parser.add_argument("--enable-shortcut", dest="enable_shortcut", action="store_true",
                        help="Force enable embed-to-pool shortcut gating")
    parser.add_argument("--disable-shortcut", dest="enable_shortcut", action="store_false",
                        help="Disable embed-to-pool shortcut gating")
    parser.add_argument("--shortcut-init", type=float, default=None,
                        help="Initial value (pre-sigmoid) for embed shortcut gate")
    parser.add_argument("--enable-multi-scale", dest="enable_multi_scale", action="store_true",
                        help="Enable multi-scale pooling wrapper")
    parser.add_argument("--disable-multi-scale", dest="enable_multi_scale", action="store_false",
                        help="Disable multi-scale pooling wrapper")
    parser.add_argument("--multi-scale-fusion", type=str, choices=["gate", "concat"], default=None,
                        help="Fusion strategy between attention and mean pooling")
    parser.add_argument("--multi-scale-dropout", type=float, default=None,
                        help="Dropout applied to the fused multi-scale representation")
    parser.add_argument("--multi-scale-gate-init", type=float, default=None,
                        help="Initial value (pre-sigmoid) for multi-scale fusion gate")
    parser.add_argument("--multi-scale-attn-hidden", type=int, default=None,
                        help="Hidden size used inside attention pooling when multi-scale is enabled")
    parser.set_defaults(enable_shortcut=None, enable_multi_scale=None)
    return parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])

# ---------------- 工具 ----------------
def resolve_query_dir(query_root: str, query_num_dir: int | None, query_dir: str | None) -> str:
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

def _sanitize_tensor_(t: torch.Tensor, is_degree: bool = False) -> torch.Tensor:
    if t is None or not torch.is_tensor(t): return t
    if is_degree:
        t = t.float()
        t = torch.clamp(t, min=0.0)  # 统一允许 0 度
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

def _get_embed_fields(ge) -> dict:
    return dict(
        vertex = getattr(ge, "vertex_embed", None) or getattr(ge, "vertex_emb", None),
        label  = getattr(ge, "label_embed",  None) or getattr(ge, "label_emb",  None),
        wildcard_idx = getattr(ge, "wildcard_idx", None),
        num_labels   = getattr(ge, "num_labels",   None),
    )

def _infer_bounds_from_model_or_batch(model, subgraphs_batch, queries):
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
            nl = total - 1  # 最后一个用于 wildcard

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

def _maybe_resize_embeddings(model: GraphCardinalityEstimatorMultiSubgraph,
                             num_vertices_needed: int | None,
                             num_labels_needed: int | None):
    ge = getattr(model, "embed", None)
    if ge is None:
        return
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    fields = _get_embed_fields(ge)
    ve = fields["vertex"]; le = fields["label"]

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
        need_n = int(num_labels_needed) + 1  # + wildcard
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
                setattr(ge, "wildcard_idx", int(num_labels_needed))
            except Exception:
                pass
            print(f"[Resize] label_embedding: {cur_n} -> {need_n} (incl wildcard)")

def _resize_to_fit_checkpoint(model, model_keys):
    ge = getattr(model, "embed", None)
    if ge is None:
        return
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    cand_label = ["embed.label_emb.weight", "embed.label_embed.weight"]
    cand_vertex = ["embed.vertex_emb.weight", "embed.vertex_embed.weight"]

    for k in cand_label:
        if k in model_keys:
            need = int(model_keys[k].shape[0])
            emb = getattr(ge, "label_emb", None) or getattr(ge, "label_embed", None)
            if isinstance(emb, nn.Embedding) and emb.num_embeddings < need:
                new = nn.Embedding(need, emb.embedding_dim).to(dev, dtype=dtype)
                with torch.no_grad():
                    new.weight[:emb.num_embeddings].copy_(emb.weight)
                    nn.init.xavier_uniform_(new.weight[emb.num_embeddings:])
                if hasattr(ge, "label_emb"):
                    ge.label_emb = new
                else:
                    ge.label_embed = new
                try:
                    setattr(ge, "num_labels", need - 1)
                    setattr(ge, "wildcard_idx", need - 1)
                except Exception:
                    pass
                print(f"[ResizeFit] label_embedding: {emb.num_embeddings} -> {need} (from ckpt)")
            break

    for k in cand_vertex:
        if k in model_keys:
            need = int(model_keys[k].shape[0])
            emb = getattr(ge, "vertex_emb", None) or getattr(ge, "vertex_embed", None)
            if isinstance(emb, nn.Embedding) and emb.num_embeddings < need:
                new = nn.Embedding(need, emb.embedding_dim).to(dev, dtype=dtype)
                with torch.no_grad():
                    new.weight[:emb.num_embeddings].copy_(emb.weight)
                    nn.init.xavier_uniform_(new.weight[emb.num_embeddings:])
                if hasattr(ge, "vertex_emb"):
                    ge.vertex_emb = new
                else:
                    ge.vertex_embed = new
                print(f"[ResizeFit] vertex_embedding: {emb.num_embeddings} -> {need} (from ckpt)")
            break

def _clamp_ids_inplace(model: GraphCardinalityEstimatorMultiSubgraph, subgraphs_batch, queries):
    num_vertices, num_labels = _infer_bounds_from_model_or_batch(model, subgraphs_batch, queries)
    ge = getattr(model, "embed", None)
    fields = _get_embed_fields(ge) if ge is not None else dict()
    le = fields.get("label", None)

    if isinstance(le, nn.Embedding):
        total = int(le.num_embeddings)
        wildcard_idx = fields.get("wildcard_idx", None)
        if (wildcard_idx is None) or (not isinstance(wildcard_idx, (int, float))) \
           or (int(wildcard_idx) < 0) or (int(wildcard_idx) >= total):
            wildcard_idx = total - 1
            try:
                setattr(ge, "wildcard_idx", int(wildcard_idx))
            except Exception:
                pass
    else:
        wildcard_idx = int(num_labels)

    wildcard_idx = int(wildcard_idx)

    for subs in subgraphs_batch:
        for g in subs:
            if "vertex_ids" in g:
                g["vertex_ids"].clamp_(min=0, max=max(0, num_vertices - 1))
            if "labels" in g:
                lbl = g["labels"]
                bad = (lbl < 0) | (lbl >= num_labels)
                if bad.any():
                    lbl = lbl.clone()
                    lbl[bad] = wildcard_idx
                    g["labels"] = lbl
            if "edge_index" in g and "labels" in g:
                g["edge_index"] = _filter_edge_index(g["edge_index"], int(g["labels"].size(0)))

    for q in queries:
        if "labels" in q:
            lbl = q["labels"]
            bad = (lbl < 0) | (lbl >= num_labels)
            if bad.any():
                lbl = lbl.clone()
                lbl[bad] = wildcard_idx
                q["labels"] = lbl
        if "edge_index" in q and "labels" in q:
            q["edge_index"] = _filter_edge_index(q["edge_index"], int(q["labels"].size(0)))

def _batch_has_non_finite(subgraphs_batch, queries, y) -> bool:
    def bad(x: torch.Tensor) -> bool:
        if not torch.is_tensor(x): return False
        if not x.is_floating_point(): return False
        return (~torch.isfinite(x)).any().item()
    for subs in subgraphs_batch:
        for g in subs:
            for k in ("vertex_ids", "labels", "edge_index", "degree"):
                if k in g and bad(g[k]): return True
    for q in queries:
        for k in ("labels", "edge_index", "degree"):
            if k in q and bad(q[k]): return True
    return bad(y)

def move_batch_to_device(data_graph_batch, query_batch, y, device):
    for subs in data_graph_batch:
        for g in subs:
            if "vertex_ids" in g: g["vertex_ids"] = g["vertex_ids"].long()
            if "labels"     in g: g["labels"]     = g["labels"].long()
            if "edge_index" in g: g["edge_index"] = g["edge_index"].long()
            if "degree"     in g: g["degree"]     = _sanitize_tensor_(g["degree"], is_degree=True)
    for q in query_batch:
        if "labels"     in q: q["labels"]     = q["labels"].long()
        if "edge_index" in q: q["edge_index"] = q["edge_index"].long()
        if "degree"     in q: q["degree"]     = _sanitize_tensor_(q["degree"], is_degree=True)
    y = _sanitize_tensor_(y, is_degree=False).long()

    y = y.to(device, non_blocking=True)
    for subs in data_graph_batch:
        for g in subs:
            g["vertex_ids"] = g["vertex_ids"].to(device, non_blocking=True)
            g["labels"]     = g["labels"].to(device, non_blocking=True)
            g["edge_index"] = g["edge_index"].to(device, non_blocking=True)
            if "degree" in g and g["degree"] is not None:
                g["degree"] = g["degree"].to(device, non_blocking=True)
    for q in query_batch:
        q["labels"]     = q["labels"].to(device, non_blocking=True)
        q["edge_index"] = q["edge_index"].to(device, non_blocking=True)
        if "degree" in q and q["degree"] is not None:
            q["degree"] = q["degree"].to(device, non_blocking=True)
    return data_graph_batch, query_batch, y

# ---------------- 组合损失：MSLE + mean QErr + Huber(|log10(QErr)|) ----------------
def _huber(x: torch.Tensor, delta: float):
    ax = torch.abs(x)
    return torch.where(ax <= delta, 0.5 * (ax ** 2) / delta, ax - 0.5 * delta)

def _trimmed_mean_loss(loss_vec: torch.Tensor, trim_ratio: float):
    if trim_ratio <= 0.0 or loss_vec.numel() < 2:
        return loss_vec.mean()
    k = int((1.0 - trim_ratio) * loss_vec.numel())
    k = max(1, min(k, loss_vec.numel()))
    topk_vals, _ = torch.topk(loss_vec, k=k, largest=False, sorted=False)
    return topk_vals.mean()

def _qerror_tensor_from_log1p(log1p_pred: torch.Tensor, y: torch.Tensor, cap: float | None, eps: float = 1e-6):
    pred  = torch.expm1(log1p_pred).clamp_min(1.0)
    label = y.float().clamp_min(1.0)
    ratio = (pred + eps) / (label + eps)
    inv   = (label + eps) / (pred + eps)
    qerr  = torch.maximum(ratio, inv)
    if cap is not None:
        qerr = torch.clamp(qerr, max=cap)
    return qerr, pred, label

def _signed_log10_qerr(qerr: torch.Tensor, pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-6):
    # qerr >= 1 by definition => log10(qerr) >= 0; 用 “有符号”的定义能稳定惩罚方向
    sign = torch.sign(pred - label)
    q = torch.clamp(qerr, min=1.0)
    return sign * torch.log10(q)

def composite_loss(log1p_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    返回 (total_loss, dict_of_components)
    """
    # MSLE：直接在 log1p 空间的 MSE
    target_log1p = torch.log1p(y.float().clamp_min(0))
    msle = F.mse_loss(log1p_pred, target_log1p)

    # Q-Error（带上限裁剪，避免训练早期爆炸）
    qerr, pred, label = _qerror_tensor_from_log1p(log1p_pred, y, cap=CONFIG["QERR_CLIP_MAX"])
    mean_qerr = qerr.mean()

    # Huber(|log10(QErr)|) 的鲁棒版本
    signed_l10 = _signed_log10_qerr(qerr, pred, label)  # 可能正负
    huber_raw  = _huber(signed_l10, delta=CONFIG["LOG_HUBER_DELTA"]).abs()  # 对称 => 取绝对
    huber_logq = _trimmed_mean_loss(huber_raw, CONFIG["TRIM_RATIO"])

    total = CONFIG["W_MSLE"] * msle + CONFIG["W_QERR"] * mean_qerr + CONFIG["W_LOGHUBER"] * huber_logq
    parts = dict(msle=float(msle.item()),
                 mean_qerr=float(mean_qerr.item()),
                 huber_logq=float(huber_logq.item()),
                 total=float(total.item()))
    return total, parts

# ---------------- 模型与前向（log1p_pred） ----------------
def build_model(device, data_graph, num_subgraphs: int):
    

    cfg = dict(
        gnn_in_ch=8, gnn_hidden_ch=16, gnn_out_ch=32, num_gnn_layers=2,
        transformer_dim=64, transformer_heads=4, transformer_ffn_dim=128,
        transformer_layers=2,
        num_subgraphs=num_subgraphs,
        num_vertices=data_graph.num_vertices,
        num_labels=getattr(data_graph, "num_labels", 64),
        dropout=0.05,
        enable_embed_shortcut=CONFIG.get("ENABLE_EMBED_SHORTCUT", True),
        embed_shortcut_init=CONFIG.get("EMBED_SHORTCUT_INIT", 0.0),
        use_multi_scale_pool=CONFIG.get("USE_MULTI_SCALE_POOL", True),
        multi_scale_fusion=CONFIG.get("MULTI_SCALE_FUSION", "gate"),
        multi_scale_dropout=CONFIG.get("MULTI_SCALE_DROPOUT", None),
        multi_scale_gate_init=CONFIG.get("MULTI_SCALE_GATE_INIT", 0.0),
        multi_scale_attn_hidden=CONFIG.get("MULTI_SCALE_ATTN_HIDDEN", None),
    )
    model = GraphCardinalityEstimatorMultiSubgraph(**cfg).to(device)
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
    model.apply(_init)
    return model

def model_forward_log1p_pred(model: GraphCardinalityEstimatorMultiSubgraph,
                             subgraphs_batch, queries) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    _clamp_ids_inplace(model, subgraphs_batch, queries)

    # === memory tokens: [B, S, D] ===
    mem_tokens = []
    for subs in subgraphs_batch:
        toks = []
        for g in subs:
            if hasattr(model, "forward_memory_token_from_subgraph"):
                ei = _filter_edge_index(g.get("edge_index"), int(g["labels"].size(0)))
                tok = model.forward_memory_token_from_subgraph(
                    g["vertex_ids"], g["labels"], g.get("degree"), ei, None
                )                  # [1,D]
            else:
                x = model.embed.forward_data(g["vertex_ids"], g["labels"], g.get("degree"))
                if hasattr(model, "gnn_encoder_data"):
                    ei = _filter_edge_index(g.get("edge_index"), int(g["labels"].size(0)))
                    x = model.gnn_encoder_data(x, ei)
                pooled = model.pool_data(x) if hasattr(model, "pool_data") else x.mean(0, keepdim=True)
                tok = model.project(pooled)   # [1,D]
            toks.append(tok)
        if not toks:
            toks = [torch.zeros(1, model.cls_token.shape[-1], device=device, dtype=dtype)]
        mem_tokens.append(torch.cat(toks, dim=0))  # [S_i, D]

    S_max = max(t.shape[0] for t in mem_tokens)
    D = mem_tokens[0].shape[-1]
    padded = []
    valid_S = []
    for t in mem_tokens:
        valid_S.append(t.shape[0])
        if t.shape[0] < S_max:
            pad_rows = torch.zeros(S_max - t.shape[0], D, device=device, dtype=dtype)
            t = torch.cat([t, pad_rows], dim=0)
        padded.append(t)
    mem_core = torch.stack(padded, dim=0)  # [B,S_max,D]
    B = mem_core.shape[0]

    if hasattr(model, "cls_token") and isinstance(model.cls_token, torch.Tensor):
        cls_tok = model.cls_token.to(device=device, dtype=dtype).expand(B, 1, D)
    else:
        cls_tok = torch.zeros(B, 1, D, device=device, dtype=dtype)
    mem = torch.cat([cls_tok, mem_core], dim=1)   # [B,1+S_max,D]

    # === key padding mask (True=padding) ===
    valid_lens = torch.tensor([1 + s for s in valid_S], device=device)  # +1 for cls
    S_full = mem.size(1)
    ar = torch.arange(S_full, device=device).unsqueeze(0).expand(B, S_full)
    src_key_padding_mask = ar >= valid_lens.unsqueeze(1)   # [B, S_full] bool

    mem_norm = getattr(model, "mem_norm", nn.Identity())
    mem = mem_norm(mem)
    if hasattr(model, "transformer_encoder"):
        mem = model.transformer_encoder(mem, src_key_padding_mask=src_key_padding_mask)

    # === query token: [B,1,D] ===
    q_toks = []
    for q in queries:
        if hasattr(model, "forward_query_token"):
            ei_q = _filter_edge_index(q.get("edge_index"), int(q["labels"].size(0)))
            tok_q = model.forward_query_token(q["labels"], q.get("degree"), ei_q, None)
        else:
            xq = model.embed.forward_query(q["labels"], q.get("degree"))
            if hasattr(model, "gnn_encoder_query"):
                ei_q = _filter_edge_index(q.get("edge_index"), int(q["labels"].size(0)))
                xq = model.gnn_encoder_query(xq, ei_q)
            pooled_q = model.pool_query(xq) if hasattr(model, "pool_query") else xq.mean(0, keepdim=True)
            tok_q = model.project(pooled_q)
        q_toks.append(tok_q)  # [1,D]
    tgt = torch.stack([t.squeeze(0) for t in q_toks], dim=0).unsqueeze(1)  # [B,1,D]
    if hasattr(model, "query_token") and isinstance(model.query_token, torch.Tensor):
        tgt = tgt + model.query_token.to(device=device, dtype=dtype).expand_as(tgt)
    query_norm = getattr(model, "query_norm", nn.Identity())
    tgt = query_norm(tgt)

    if hasattr(model, "transformer_decoder"):
        tgt = model.transformer_decoder(tgt, mem, memory_key_padding_mask=src_key_padding_mask)  # [B,1,D]

    log1p_pred = model.head(tgt.squeeze(1)).squeeze(-1)  # [B]
    return log1p_pred

# ---------------- 校准层 ----------------
class OutputCalibrator(nn.Module):
    def __init__(self, a_init=1.0, b_init=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))
    def forward(self, x): return self.a * x + self.b

class _PackForSave(nn.Module):
    def __init__(self, model: nn.Module, calibrator: nn.Module | None):
        super().__init__()
        self.model = model
        self.calib = calibrator if calibrator is not None else nn.Identity()

# ---------------- 训练/验证/测试 ----------------
def set_head_bias_to_mu(model: GraphCardinalityEstimatorMultiSubgraph, mu: float):
    with torch.no_grad():
        last = model.head[-1]
        if isinstance(last, nn.Linear):
            last.bias.fill_(float(mu))
    print(f"[Calibrate] head.bias <- {mu:.4f} (log1p-scale)")

def make_param_groups(model, base_lr, weight_decay, head_lr_mult=1.0, calibrator=None, calib_lr_mult=1.0):
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    def is_no_decay(n, p): return (p.dim() == 1) or n.endswith('bias') or ('norm' in n.lower())
    decay_base, no_decay_base = [], []
    for n, p in model.named_parameters():
        if id(p) in head_ids: continue
        (no_decay_base if is_no_decay(n, p) else decay_base).append(p)
    groups = [
        {'params': decay_base,    'lr': base_lr,                'weight_decay': weight_decay, 'base_lr': base_lr},
        {'params': no_decay_base, 'lr': base_lr,                'weight_decay': 0.0,          'base_lr': base_lr},
        {'params': head_params,   'lr': base_lr * head_lr_mult, 'weight_decay': weight_decay, 'base_lr': base_lr * head_lr_mult},
    ]
    if calibrator is not None:
        groups.append({'params': list(calibrator.parameters()),
                       'lr': base_lr * calib_lr_mult, 'weight_decay': 0.0, 'base_lr': base_lr * calib_lr_mult})
    base_params = [p for p in model.parameters() if id(p) not in head_ids]
    return groups, base_params, head_params

def train_one_phase(model, train_loader, val_loader, optimizer, scheduler,
                    epochs: int, patience: int, device: torch.device,
                    best_path: str = "best_model.pth",
                    calibrator: OutputCalibrator | None = None,
                    warmup_head_epochs: int = 0,
                    base_params=None):
    """
    使用 AveragedModel 做 EMA：
    - 训练期间：每个 step 末尾 update EMA
    - 验证：直接用 ema_model.module 验证（不再 clone/load）
    - EarlyStopping：保存 ema_model.module（与验证一致）
    - 训练结束：从 best_path 载入回 model（仍是“普通”模型），统一后续流程
    """
    # ---------- 新增：EMA ----------
    ema_decay = 0.99
    ema_model = AveragedModel(
        model,
        avg_fn=lambda avg, cur, n: ema_decay * avg + (1.0 - ema_decay) * cur
    )
    for p in ema_model.parameters():
        p.requires_grad_(False)
    # --------------------------------

    stopper = EarlyStopping(patience=patience, verbose=True, path=best_path)

    def _set_base_requires_grad(flag: bool):
        if base_params is None: return
        for p in base_params: p.requires_grad = flag

    for epoch in range(1, epochs + 1):
        # 学习率 warmup（相对 base_lr）
        if CONFIG.get('LR_WARMUP_EPOCHS', 0) and epoch <= CONFIG['LR_WARMUP_EPOCHS']:
            factor = float(epoch) / float(max(1, CONFIG['LR_WARMUP_EPOCHS']))
            for pg in optimizer.param_groups:
                base_lr = pg.get('base_lr', pg['lr'])
                pg['lr'] = base_lr * factor

        # 阶段冻结策略
        if epoch == 1 and warmup_head_epochs > 0:
            _set_base_requires_grad(False)
            print(f"[Warmup] Freeze backbone for first {warmup_head_epochs} epoch(s).")
        if epoch == warmup_head_epochs + 1 and warmup_head_epochs > 0:
            _set_base_requires_grad(True)
            if calibrator is not None:
                for p in calibrator.parameters(): p.requires_grad = False
            print("[Warmup] Unfreeze backbone & freeze calibrator.")

        model.train()
        if calibrator is not None: calibrator.train()
        running, start = 0.0, time.perf_counter()

        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as bar:
            for data_graph_batch, query_batch, y in bar:
                data_graph_batch, query_batch, y = move_batch_to_device(data_graph_batch, query_batch, y, device)

                with amp.autocast(device_type='cuda', enabled=USE_AMP and (device.type == 'cuda')):
                    log1p_pred = model_forward_log1p_pred(model, data_graph_batch, query_batch)
                    if calibrator is not None:
                        log1p_pred = calibrator(log1p_pred)

                    # === 联合损失 ===
                    loss, parts = composite_loss(log1p_pred, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if calibrator is not None:
                    torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
                optimizer.step()

                # ---------- 新增：每 step 末尾更新 EMA ----------
                ema_model.update_parameters(model)
                # ------------------------------------------------

                running += float(loss.item())
                bar.set_postfix(
                    loss=f"{running/(bar.n+1):.4f}",
                    msle=f"{parts['msle']:.3f}",
                    qerr=f"{parts['mean_qerr']:.3f}",
                    huber=f"{parts['huber_logq']:.3f}"
                )

        print(f"Epoch {epoch}/{epochs}  TrainLoss={running/max(1,len(train_loader)):.6f}  time={time.perf_counter()-start:.1f}s")

               # ===== Validation：以 Mean Q-Error 作为验证指标（使用 EMA 权重）=====
        model.eval()
        if calibrator is not None: 
            calibrator.eval()

        eval_model = ema_model.module     # 用 EMA 平均后的模型做验证
        sum_qerr, n_samples = 0.0, 0

        with torch.no_grad(), tqdm(val_loader, desc="Validation(QErr)", unit="batch") as bar:
            for dg_batch, q_batch, y in bar:
                dg_batch, q_batch, y = move_batch_to_device(dg_batch, q_batch, y, device)
                with amp.autocast(device_type='cuda', enabled=USE_AMP and (device.type == 'cuda')):
                    log1p_pred = model_forward_log1p_pred(eval_model, dg_batch, q_batch)
                    if calibrator is not None:
                        log1p_pred = calibrator(log1p_pred)
                    # 验证阶段不做裁剪：cap=None
                    qerr, pred, label = _qerror_tensor_from_log1p(log1p_pred, y, cap=None)
                    mean_q = qerr.mean()

                # 以样本数加权平均，避免最后一个小 batch 拉偏
                bs = int(label.numel())
                sum_qerr += float(mean_q.item()) * bs
                n_samples += bs

                bar.set_postfix(mean_qerr=f"{(sum_qerr/max(1,n_samples)):.4f}")

        val_qerr = sum_qerr / max(1, n_samples)
        print(f"Val Mean Q-Error (EMA): {val_qerr:.6f}")

        # 早停 & 学习率调度都用 Q-Error（越小越好）
        stopper(val_qerr, _PackForSave(ema_model.module, calibrator))
        if stopper.early_stop:
            print("Early stopping triggered.")
            break
        scheduler.step(val_qerr)


    # 载入最佳（包含 calibrator）
    device = next(model.parameters()).device
    state = torch.load(best_path, map_location=device)
    model_keys = {k.split("model.",1)[1]: v for k,v in state.items() if k.startswith("model.")}
    calib_keys = {k.split("calib.",1)[1]: v for k,v in state.items() if k.startswith("calib.")}
    model.load_state_dict(model_keys or state, strict=False)
    if calibrator is not None and calib_keys:
        calibrator.load_state_dict(calib_keys, strict=False)
    return model, calibrator


def test(model, loader, device: torch.device, out_path: str = "predictions_and_labels.txt",
         calibrator: OutputCalibrator | None = None):
    """
    测试阶段：不使用任何 Q-Error 裁剪（cap=None）
    """
    model.eval()
    if calibrator is not None: calibrator.eval()
    total_mean_qerr, n = 0.0, 0
    log10_q_list = []     # 收集 logQError
    abs_log10_all = []    # 保留旧指标（可对比）
    start_t = time.perf_counter()
    with open(out_path, "w") as f, tqdm(loader, desc="Testing", unit="batch") as bar:
        f.write("Batch,Prediction,Label,QError\n")
        for dg_batch, q_batch, y in bar:
            dg_batch, q_batch, y = move_batch_to_device(dg_batch, q_batch, y, device)
            with amp.autocast(device_type='cuda', enabled=USE_AMP and (device.type == 'cuda')):
                log1p_pred = model_forward_log1p_pred(model, dg_batch, q_batch)
                if calibrator is not None:
                    log1p_pred = calibrator(log1p_pred)
                qerr, pred, label = _qerror_tensor_from_log1p(log1p_pred, y, cap=None)
            total_mean_qerr += qerr.sum().item()
            n += label.numel()

            # 记录 logQError（无符号）与 |log10Q| 以便对比
            l10 = torch.log10(qerr.clamp_min(1.0)).detach().cpu().numpy()
            log10_q_list.append(l10)
            abs_log10_all.append(np.abs(l10))

            for idx, (p, l, e) in enumerate(zip(pred.tolist(), label.tolist(), qerr.tolist()), 1):
                f.write(f"{idx},{p:.4f},{l:.4f},{e:.4f}\n")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_t
    avg_query_time = total_time / max(1, n)
    avg_q = total_mean_qerr / max(1, n)

    # 统计 logQError 分位数
    log10_q_all = np.concatenate(log10_q_list) if log10_q_list else np.array([np.inf])
    p10 = float(np.percentile(log10_q_all, 10))
    q1  = float(np.percentile(log10_q_all, 25))
    med = float(np.percentile(log10_q_all, 50))
    q3  = float(np.percentile(log10_q_all, 75))
    p90 = float(np.percentile(log10_q_all, 90))
    p95 = float(np.percentile(log10_q_all, 95))
    mean_l10 = float(np.mean(log10_q_all))

    # 旧的 |log10Q| IQR 统计（可留作对比）
    abs_log10_all = np.concatenate(abs_log10_all) if abs_log10_all else np.array([np.inf])
    q1_abs, q3_abs = float(np.percentile(abs_log10_all, 25)), float(np.percentile(abs_log10_all, 75))
    iqr_abs = q3_abs - q1_abs

    print(
        f"Test Mean Q-Error {avg_q:.6f} | "
        f"logQError stats -> mean={mean_l10:.4f}  p10={p10:.4f}  Q1={q1:.4f}  median={med:.4f}  Q3={q3:.4f}  p90={p90:.4f}  p95={p95:.4f} | "
        f"Avg Query Time = {avg_query_time:.6f} s"
    )
    with open(out_path, "a") as f:
        f.write("# logQError_percentiles\n")
        f.write(f"# mean={mean_l10:.6f}, p10={p10:.6f}, q1={q1:.6f}, median={med:.6f}, q3={q3:.6f}, p90={p90:.6f}, p95={p95:.6f}\n")
        f.write(f"# total_queries: {n}\n")
        f.write(f"# total_test_time_sec: {total_time:.6f}\n")
        f.write(f"# average_query_time_sec: {avg_query_time:.6f}\n")
    return avg_q, med, q3, iqr_abs

# ---------------- 主流程 ----------------
def main():
    global CONFIG, GLOBAL_NUM_VERTICES, GLOBAL_NUM_LABELS
    CONFIG.update(_bind_output_paths_to_selected_num(_apply_dataset(CONFIG)))
    C = CONFIG
    torch.manual_seed(C["SEED"]); np.random.seed(C["SEED"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(C["SEED"])
    device = torch.device("cuda:0" if (C["DEVICE"] == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Using", device)

    query_dir = resolve_query_dir(C["QUERY_ROOT"], C["QUERY_NUM_DIR"], C["QUERY_DIR"])
    if not os.path.exists(C["PREPARED_OUT"]):
        print(f"[Info] {C['PREPARED_OUT']} 不存在，调用 prepare_data() 生成 …")
        Path(C["PREPARED_OUT"]).parent.mkdir(parents=True, exist_ok=True)
        prepare_data(
            device=("cuda" if device.type == "cuda" else "cpu"),
            data_graph_filename=C["DATA_GRAPH"],
            query_graph_folder=query_dir,
            matches_path=C["MATCHES_PATH"],
            output_path=C["PREPARED_OUT"]
        )
    else:
        print(f"[Info] 使用已存在的 {C['PREPARED_OUT']}")

    prepared = load_prepared_pyg(C["PREPARED_OUT"])
    dataset  = build_dataset_from_prepared(prepared, repeat_idx=0)
    num_subgraphs = len(prepared['pyg_data_graphs'][0])
    print(f"[Info] num_subgraphs = {num_subgraphs}")

    # 统计 ID / label 范围（用于后续 embedding 扩容）
    vmins, vmaxs = [], []
    for g in prepared['pyg_data_graphs'][0]:
        if 'vertex_ids' in g and g['vertex_ids'].numel() > 0:
            vmins.append(int(g['vertex_ids'].min().item()))
            vmaxs.append(int(g['vertex_ids'].max().item()))
    if vmins and vmaxs:
        GLOBAL_NUM_VERTICES = max(vmaxs) + 1
        print(f"[VID] min(vertex_id) over subgraphs = {min(vmins)}")
        print(f"[VID] max(vertex_id) over subgraphs = {max(vmaxs)}")
        print(f"[EmbedSize] using num_vertices_data = {GLOBAL_NUM_VERTICES}  (from prepared vertex_ids)")

    lab_vals = []
    neg1 = False
    for g in prepared['pyg_data_graphs'][0]:
        if 'labels' in g and g['labels'].numel() > 0:
            lab_vals.append(int(g['labels'].max().item()))
            if int(g['labels'].min().item()) < 0: neg1 = True
    for q in prepared['pyg_query_graphs']:
        if 'labels' in q and q['labels'].numel() > 0:
            lab_vals.append(int(q['labels'].max().item()))
            if int(q['labels'].min().item()) < 0: neg1 = True
    if lab_vals:
        mx_lab = max(lab_vals)
        if neg1: print("[LABEL] min(label) = -1")
        print(f"[LABEL] max(label) = {mx_lab}")
        GLOBAL_NUM_LABELS = mx_lab + 1
        print(f"[LabelSize] using num_labels = {GLOBAL_NUM_LABELS}  (max(label)+1 across data+query)")

    # 边界检查（仅提示）
    print("[Verify] scanning graphs for out-of-range indices ...")
    bad = 0
    for g in prepared['pyg_data_graphs'][0]:
        if 'edge_index' in g and 'labels' in g and g['edge_index'].numel() > 0:
            E = g['edge_index'].size(1); N = g['labels'].size(0)
            if E:
                mx = int(g['edge_index'].max().item()); mn = int(g['edge_index'].min().item())
                if mx >= N or mn < 0: bad += 1
    print("[Verify] OK: no out-of-range indices." if bad == 0 else f"[Verify] FOUND {bad} data subgraphs with out-of-range edge_index! (filtered on-the-fly)")

    print("[Verify] scanning query graphs for out-of-range indices ...")
    bad_q = 0
    for q in prepared['pyg_query_graphs']:
        if 'edge_index' in q and 'labels' in q and q['edge_index'].numel() > 0:
            E = q['edge_index'].size(1); N = q['labels'].size(0)
            if E:
                mx = int(q['edge_index'].max().item()); mn = int(q['edge_index'].min().item())
                if mx >= N or mn < 0: bad_q += 1
    print("[Verify] OK: no out-of-range indices in queries." if bad_q == 0 else f"[Verify] FOUND {bad_q} query graphs out-of-range! (filtered on-the-fly)")

    # ====== 分层划分 ======
    splits = make_splits_indices_stratified(
        prepared,
        selected_query_num=C["SELECTED_QUERY_NUM"],
        pretrain_ratio=C["PRETRAIN_RATIO"],
        k_folds=C["K_FOLDS"],
        seed=C["SEED"],
        exclude_overlap=C["EXCLUDE_OVERLAP"],
        n_bins=C["STRATA_NUM_BINS"],
    )

    def _preview(prepared, splits, k_preview=5):
        qids = prepared['query_ids']
        name = lambda idxs: ', '.join(qids[i] for i in idxs[:k_preview])
        lines = [f"Global queries: {len(qids)}",
                 f"Pretrain: {len(splits['pretrain_idx'])} e.g., [{name(splits['pretrain_idx'])}]"]
        for fi, fold in enumerate(splits['folds']):
            lines.append(f"Fold {fi+1}: train={len(fold['train_idx'])}, val={len(fold['val_idx'])}. "
                         f"val e.g., [{name(fold['val_idx'])}]")
        return "\n".join(lines)
    print("=== Split Preview ===")
    print(_preview(prepared, splits))

    # 预训练集 90/10 切分
    pre_idx = splits['pretrain_idx']
    rng = np.random.default_rng(C["SEED"]); rng.shuffle(pre_idx)
    cut = max(1, int(round(0.9 * len(pre_idx))))
    pre_tr = pre_idx[:cut]; pre_va = pre_idx[cut:]
    print(f"[Pretrain] train={len(pre_tr)}, val={len(pre_va)}")

    dataset  = build_dataset_from_prepared(prepared, repeat_idx=0)
    pre_train_dl = create_dataloader(dataset, C["BATCH_SIZE"], indices=pre_tr,
                                     shuffle=True, seed=C["SEED"], pin_memory=False)
    pre_val_dl   = create_dataloader(dataset, C["BATCH_SIZE"], indices=pre_va,
                                     shuffle=False, seed=C["SEED"], pin_memory=False)

    # ====== 构建模型 ======
    data_graph = load_graph_from_file(C["DATA_GRAPH"])
    model = build_model(device, data_graph, num_subgraphs=num_subgraphs)
    _maybe_resize_embeddings(model, GLOBAL_NUM_VERTICES, GLOBAL_NUM_LABELS)

    # 若存在已完成的预训练权重，直接加载并跳过 Stage 1（连同 calibrator）
    if os.path.exists(C["BEST_PRE_PATH"]):
        print(f"[Info] Found existing best pretrain weights: {C['BEST_PRE_PATH']} (will load & skip Stage 1)")
        state_pre = torch.load(C["BEST_PRE_PATH"], map_location=device)
        model_keys = {k.split("model.",1)[1]: v for k,v in state_pre.items() if k.startswith("model.")}
        calib_keys = {k.split("calib.",1)[1]: v for k,v in state_pre.items() if k.startswith("calib.")}
        if not model_keys: model_keys = state_pre
        _resize_to_fit_checkpoint(model, model_keys)
        model.load_state_dict(model_keys, strict=False)
        calibrator_pre = OutputCalibrator() if CONFIG["USE_CALIBRATOR"] else None
        if calibrator_pre is not None and calib_keys:
            calibrator_pre.load_state_dict(calib_keys, strict=False)
    else:
        # 设定 head.bias 为 log1p(y) 的均值或中位
        y_pre_train = np.array([prepared['true_cardinalities'][i] for i in pre_tr], dtype=np.float64)
        mu_pre = float((np.median if CONFIG["USE_MEDIAN_BIAS"] else np.mean)(np.log1p(y_pre_train))) if y_pre_train.size else 0.0
        set_head_bias_to_mu(model, mu_pre)

        calibrator_pre = OutputCalibrator() if CONFIG["USE_CALIBRATOR"] else None
        groups, base_params, _ = make_param_groups(
            model, CONFIG["LR"], CONFIG["WEIGHT_DECAY"], head_lr_mult=CONFIG["HEAD_LR_MULT"],
            calibrator=calibrator_pre, calib_lr_mult=CONFIG["CALIB_LR_MULT"]
        )
        opt = torch.optim.AdamW(groups)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                         patience=max(1, CONFIG["PATIENCE"] // 2), factor=0.5)

        print("\n========== Stage 1: Pretrain (composite loss) ==========")
        model, calibrator_pre = train_one_phase(
            model, pre_train_dl, pre_val_dl, opt, sch,
            epochs=CONFIG["EPOCHS_PRE"], patience=CONFIG["PATIENCE"], device=device,
            best_path=CONFIG["BEST_PRE_PATH"],
            calibrator=calibrator_pre,
            warmup_head_epochs=0, base_params=base_params
        )
        torch.save(_PackForSave(model, calibrator_pre).state_dict(), CONFIG["PRETRAIN_WEIGHTS"])
        print(f"保存预训练权重 -> {CONFIG['PRETRAIN_WEIGHTS']}")

    # ====== K 折微调（只用 SELECTED_QUERY_NUM） ======
    print("\n========== Stage 2: K-Fold Fine-tuning (composite loss) ==========")
    fold_qerrs, fold_box_q3, fold_box_med = [], [], []

    def _print_fold_stats(name, y_list):
        arr = np.array(y_list, dtype=np.float64)
        if arr.size == 0:
            print(f"[{name}] empty"); return
        print(f"[{name}] count={arr.size}  min={arr.min():.3g}  p50={np.median(arr):.3g}  "
              f"p90={np.percentile(arr,90):.3g}  max={arr.max():.3g}")

    INNER_VAL_RATIO = 0.10  # 9:1 切 inner-val；至少保证 1 个样本

    for i, fold in enumerate(splits['folds'], 1):
        tr_idx_main, test_idx = fold['train_idx'], fold['val_idx']  # test_idx 只用于最终测试
        print(f"\n---- Fold {i}/{len(splits['folds'])}  train(main)={len(tr_idx_main)}  test={len(test_idx)} ----")
        _print_fold_stats("FoldTrainY(main)", [prepared['true_cardinalities'][j] for j in tr_idx_main])
        _print_fold_stats("FoldTestY",        [prepared['true_cardinalities'][j] for j in test_idx])

        # === 在 train(main) 内部分出 inner_train / inner_val ===
        rng_fold = np.random.default_rng(C["SEED"] + i)
        tr_shuf = tr_idx_main.copy(); rng_fold.shuffle(tr_shuf)
        cut = max(1, int(round(INNER_VAL_RATIO * len(tr_shuf))))
        inner_val_idx   = tr_shuf[:cut]
        inner_train_idx = tr_shuf[cut:]
        print(f"[Fold {i}] inner-train={len(inner_train_idx)}  inner-val={len(inner_val_idx)}")

        # 训练加载器：仅 inner-train
        if CONFIG["USE_WEIGHTED_SAMPLER"]:
            ft_train_dl = build_weighted_loader(
                dataset, inner_train_idx, batch_size=CONFIG["BATCH_SIZE"], create_dl_fn=create_dataloader, 
                seed=C["SEED"], num_workers=CONFIG["NUM_WORKERS"], pin_memory=False
            )
        else:
            ft_train_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=inner_train_idx,
                                            shuffle=True, seed=C["SEED"], pin_memory=False)

        # 早停验证加载器：inner-val
        ft_inner_val_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=inner_val_idx,
                                            shuffle=False, seed=C["SEED"], pin_memory=False)

        # 最终测试加载器：该折 val_idx
        ft_test_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=test_idx,
                                       shuffle=False, seed=C["SEED"], pin_memory=False)

        # === 模型与优化器（每折重置） ===
        model_k = build_model(device, load_graph_from_file(CONFIG["DATA_GRAPH"]), num_subgraphs=num_subgraphs)
        _maybe_resize_embeddings(model_k, GLOBAL_NUM_VERTICES, GLOBAL_NUM_LABELS)

        # 从预训练权重加载（包含 calibrator 作为初始化）
        pre_path = CONFIG["BEST_PRE_PATH"] if os.path.exists(CONFIG["BEST_PRE_PATH"]) else CONFIG["PRETRAIN_WEIGHTS"]
        calibrator_k = OutputCalibrator() if CONFIG["USE_CALIBRATOR"] else None
        if os.path.exists(pre_path):
            state_pre = torch.load(pre_path, map_location=device)
            model_keys = {k.split("model.",1)[1]: v for k,v in state_pre.items() if k.startswith("model.")}
            calib_keys = {k.split("calib.",1)[1]: v for k,v in state_pre.items() if k.startswith("calib.")}
            if not model_keys: model_keys = state_pre
            _resize_to_fit_checkpoint(model_k, model_keys)
            model_k.load_state_dict(model_keys, strict=False)
            if calibrator_k is not None and calib_keys:
                calibrator_k.load_state_dict(calib_keys, strict=False)

        y_fold_train = np.array([prepared['true_cardinalities'][j] for j in inner_train_idx], dtype=np.float64)
        # mu_fold = float((np.median if CONFIG["USE_MEDIAN_BIAS"] else np.mean)(np.log1p(y_fold_train))) if y_fold_train.size else 0.0
        # set_head_bias_to_mu(model_k, mu_fold)

        groups, base_params_k, _ = make_param_groups(
            model_k, CONFIG["LR"], CONFIG["WEIGHT_DECAY"], head_lr_mult=CONFIG["HEAD_LR_MULT"],
            calibrator=calibrator_k, calib_lr_mult=CONFIG["CALIB_LR_MULT"]
        )
        opt_k = torch.optim.AdamW(groups)
        sch_k = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_k, mode="min",
                                                           patience=CONFIG["PATIENCE"], factor=0.5)

        best_fold_path = CONFIG["BEST_FOLD_PATH_TPL"].format(fold=i)
        # 训练时只用 inner-val 做早停
        model_k, calibrator_k = train_one_phase(
            model_k, ft_train_dl, ft_inner_val_dl, opt_k, sch_k,
            epochs=CONFIG["EPOCHS_FT"], patience=CONFIG["PATIENCE"], device=device,
            best_path=best_fold_path,
            calibrator=calibrator_k,
            warmup_head_epochs=CONFIG["WARMUP_HEAD_EPOCHS"],
            base_params=base_params_k
        )

        # 仅在该折的真正测试集上评估（不 clip）
        pred_txt = CONFIG["PRED_TXT_TPL"].format(fold=i)
        avg_q, med_log, q3_log, iqr_log = test(model_k, ft_test_dl, device, out_path=pred_txt, calibrator=calibrator_k)
        fold_qerrs.append(avg_q); fold_box_med.append(med_log); fold_box_q3.append(q3_log)

    if fold_qerrs:
        print("\n========== K-Fold Summary ==========")
        for i, (q, medl, q3l) in enumerate(zip(fold_qerrs, fold_box_med, fold_box_q3), 1):
            print(f"Fold {i}: Mean Q-Err = {q:.6f} | logQError median={medl:.4f} Q3={q3l:.4f}")
        print(f"Overall mean Q-Error = {np.mean(fold_qerrs):.6f}")
        print(f"Overall logQError median = {np.mean(fold_box_med):.4f}  Q3 = {np.mean(fold_box_q3):.4f}")

    print("\nAll done.")

if __name__ == "__main__":
    parsed_args = _parse_args()
    _apply_cli_overrides(parsed_args)
    main()
