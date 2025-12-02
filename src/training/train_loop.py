"""Training and evaluation loops shared by the CLI entrypoint."""
import time
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm

from src.config.train_config import CONFIG
from src.models.ablation_factory import make_ablation_model
from src.models.estimator import GraphCardinalityEstimatorMultiSubgraph
from src.utils.earlystop import EarlyStopping
from src.utils.train_data_utils import (
    USE_AMP,
    _clamp_ids_inplace,
    _filter_edge_index,
    check_invalid_data,
    clip_gradients,
    move_batch_to_device,
    sanitize_log1p_pred,
)


def build_model(device: torch.device, data_graph, num_subgraphs: int):
    """Construct the chosen ablation model variant and initialize weights."""
    cfg = dict(
        gnn_in_ch=32,
        gnn_hidden_ch=32,
        gnn_out_ch=32,
        num_gnn_layers=2,
        transformer_dim=32,
        transformer_heads=2,
        transformer_ffn_dim=64,
        transformer_layers=1,
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
        enable_cross_attention=CONFIG.get("ENABLE_CROSS_ATTENTION", True),
        cross_attn_heads=CONFIG.get("CROSS_ATTN_HEADS", 1),
        cross_attn_dropout=CONFIG.get("CROSS_ATTN_DROPOUT", None),
        cross_ffn_hidden=CONFIG.get("CROSS_FFN_HIDDEN", None),
        use_memory_positional_encoding=CONFIG.get("USE_MEMORY_POS_ENCODING", True),
    )
    variant = CONFIG.get("MODEL_VARIANT", "BASE")
    model = make_ablation_model(variant, **cfg).to(device)

    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(_init)
    return model


def model_forward_log1p_pred(model: GraphCardinalityEstimatorMultiSubgraph, subgraphs_batch, queries) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    _clamp_ids_inplace(model, subgraphs_batch, queries)

    mem_tokens = []
    for subs in subgraphs_batch:
        toks = []
        for g in subs:
            if hasattr(model, "forward_memory_token_from_subgraph"):
                ei = _filter_edge_index(g.get("edge_index"), int(g["labels"].size(0)))
                tok = model.forward_memory_token_from_subgraph(g["vertex_ids"], g["labels"], g.get("degree"), ei, None)
            else:
                x = model.embed.forward_data(g["vertex_ids"], g["labels"], g.get("degree"))
                if hasattr(model, "gnn_encoder_data"):
                    ei = _filter_edge_index(g.get("edge_index"), int(g["labels"].size(0)))
                    x = model.gnn_encoder_data(x, ei)
                pooled = model.pool_data(x) if hasattr(model, "pool_data") else x.mean(0, keepdim=True)
                tok = model.project(pooled)
            toks.append(tok)
        if not toks:
            toks = [torch.zeros(1, model.cls_token.shape[-1], device=device, dtype=dtype)]
        seq = torch.cat(toks, dim=0)
        if hasattr(model, "apply_memory_positional_encoding"):
            seq = model.apply_memory_positional_encoding(seq)
        mem_tokens.append(seq)

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
    mem_core = torch.stack(padded, dim=0)
    B = mem_core.shape[0]

    if hasattr(model, "cls_token") and isinstance(model.cls_token, torch.Tensor):
        cls_tok = model.cls_token.to(device=device, dtype=dtype).expand(B, 1, D)
    else:
        cls_tok = torch.zeros(B, 1, D, device=device, dtype=dtype)
    mem = torch.cat([cls_tok, mem_core], dim=1)

    valid_lens = torch.tensor([1 + s for s in valid_S], device=device, dtype=torch.long)
    S_full = mem.size(1)
    ar = torch.arange(S_full, device=device).unsqueeze(0).expand(B, S_full)
    src_key_padding_mask = ar >= valid_lens.unsqueeze(1)

    if hasattr(model, "transformer_encoder"):
        mem = model.transformer_encoder(mem, src_key_padding_mask=src_key_padding_mask)
    mem_norm = getattr(model, "mem_norm", nn.Identity())
    mem = mem_norm(mem)

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

            if check_invalid_data(xq):
                print(f"[WARN] xq contains NaN/Inf in query {q}. Replacing with zeros.")
                xq = torch.where(torch.isfinite(xq), xq, torch.tensor(0.0, device=xq.device, dtype=xq.dtype))

            pooled_q = model.pool_query(xq) if hasattr(model, "pool_query") else xq.mean(0, keepdim=True)
            tok_q = model.project(pooled_q)
        q_toks.append(tok_q)

    tgt = torch.stack([t.squeeze(0) for t in q_toks], dim=0).unsqueeze(1)
    if hasattr(model, "query_token") and isinstance(model.query_token, torch.Tensor):
        tgt = tgt + model.query_token.to(device=device, dtype=dtype).expand_as(tgt)

    query_norm = getattr(model, "query_norm", nn.Identity())
    tgt = query_norm(tgt)
    if hasattr(model, "cross_interact"):
        tgt = model.cross_interact(mem, tgt, memory_key_padding_mask=src_key_padding_mask)

    if check_invalid_data(tgt):
        print("[DEBUG] ❌ tgt contains NaN/Inf BEFORE head:")
        print("tgt =", tgt)
        raise ValueError("tgt contains NaN/Inf before head layer")

    log1p_pred_raw = model.head(tgt.squeeze(1)).squeeze(-1)

    if check_invalid_data(log1p_pred_raw):
        print("[DEBUG] ❌ log1p_pred_raw contains NaN/Inf:")
        print("Raw values:", log1p_pred_raw.detach().cpu().numpy())
        print("Is NaN:", torch.isnan(log1p_pred_raw).cpu().numpy())
        print("Is Inf:", torch.isinf(log1p_pred_raw).cpu().numpy())
        raise ValueError("log1p_pred_raw contains NaN/Inf before sanitize")

    log1p_pred = sanitize_log1p_pred(log1p_pred_raw)

    if check_invalid_data(log1p_pred):
        print("[DEBUG] ❌ log1p_pred contains NaN/Inf AFTER sanitize (should not happen):")
        print("Sanitized values:", log1p_pred.detach().cpu().numpy())
        raise ValueError("log1p_pred contains NaN/Inf after sanitize")

    return log1p_pred


def _qerror_tensor_from_log1p(log1p_pred: torch.Tensor, y: torch.Tensor, cap: float | None, eps: float = 1e-6):
    pred = torch.expm1(log1p_pred).clamp_min(1.0)
    label = y.float().clamp_min(1.0)
    ratio = (pred + eps) / (label + eps)
    inv = (label + eps) / (pred + eps)
    qerr = torch.maximum(ratio, inv)
    if cap is not None:
        qerr = torch.clamp(qerr, max=cap)
    return qerr, pred, label


def _signed_log10_qerr(qerr: torch.Tensor, pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-6):
    sign = torch.sign(pred - label)
    q = torch.clamp(qerr, min=1.0)
    return sign * torch.log10(q)


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


def make_splits_indices_stratified(prepared, selected_query_num, pretrain_ratio, k_folds, seed, exclude_overlap, n_bins):
    rng = np.random.RandomState(seed)
    n = len(prepared['query_ids'])
    all_indices = np.arange(n)

    pre_n = max(1, int(round(pretrain_ratio * n)))
    pretrain_idx = rng.permutation(all_indices)[:pre_n].tolist()

    qids = prepared['query_ids']
    qk_list = prepared['query_k_list']
    pool = [i for i in all_indices if qk_list[i] == int(selected_query_num)]

    if exclude_overlap:
        s = set(pretrain_idx)
        pool = [i for i in pool if i not in s]
    if len(pool) < k_folds:
        raise ValueError("Not enough samples for K-fold after excluding overlap.")

    y_all = np.array(prepared['true_cardinalities'], dtype=np.float64)
    logy = np.log1p(y_all[pool])
    bin_ids = _digitize_by_quantiles(logy, n_bins)
    dens_ids = np.array([_parse_density_from_qid(qids[i]) for i in pool], dtype=int)

    strata = {}
    for li, gi in enumerate(pool):
        key = (int(bin_ids[li]), int(dens_ids[li]))
        strata.setdefault(key, []).append(gi)

    folds = [{'train_idx': [], 'val_idx': []} for _ in range(k_folds)]
    for _, idxs in strata.items():
        idxs = rng.permutation(idxs).tolist()
        base = len(idxs) // k_folds
        rem = len(idxs) % k_folds
        sizes = [base + (1 if i < rem else 0) for i in range(k_folds)]
        cur = 0
        slices = []
        for sz in sizes:
            slices.append(idxs[cur:cur + sz])
            cur += sz
        for f in range(k_folds):
            folds[f]['val_idx'].extend(slices[f])
            folds[f]['train_idx'].extend([x for j, sl in enumerate(slices) if j != f for x in sl])
    return {'pretrain_idx': pretrain_idx, 'folds': folds}


def _parse_density_from_qid(qid: str) -> int:
    low = qid.lower()
    if "dense" in low:
        return 1
    if "sparse" in low:
        return 0
    return -1


def _digitize_by_quantiles(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    cuts = np.quantile(values, qs)
    cuts[0], cuts[-1] = -np.inf, np.inf
    bins = np.digitize(values, cuts[1:-1], right=True)
    return bins.astype(int)


class OutputCalibrator(nn.Module):
    def __init__(self, a_init=1.0, b_init=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, x):
        return self.a * x + self.b


class _PackForSave(nn.Module):
    def __init__(self, model: nn.Module, calibrator: nn.Module | None):
        super().__init__()
        self.model = model
        self.calib = calibrator if calibrator is not None else nn.Identity()


def set_head_bias_to_mu(model: GraphCardinalityEstimatorMultiSubgraph, mu: float):
    with torch.no_grad():
        last = model.head[-1]
        if isinstance(last, nn.Linear):
            last.bias.fill_(float(mu))
    print(f"[Calibrate] head.bias <- {mu:.4f} (log1p-scale)")


def make_param_groups(model, base_lr, weight_decay, head_lr_mult=1.0, calibrator=None, calib_lr_mult=1.0):
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}

    def is_no_decay(n, p):
        return (p.dim() == 1) or n.endswith('bias') or ('norm' in n.lower())

    decay_base, no_decay_base = [], []
    for n, p in model.named_parameters():
        if id(p) in head_ids:
            continue
        (no_decay_base if is_no_decay(n, p) else decay_base).append(p)
    groups = [
        {'params': decay_base, 'lr': base_lr, 'weight_decay': weight_decay, 'base_lr': base_lr},
        {'params': no_decay_base, 'lr': base_lr, 'weight_decay': 0.0, 'base_lr': base_lr},
        {'params': head_params, 'lr': base_lr * head_lr_mult, 'weight_decay': weight_decay, 'base_lr': base_lr * head_lr_mult},
    ]
    if calibrator is not None:
        groups.append({'params': list(calibrator.parameters()), 'lr': base_lr * calib_lr_mult, 'weight_decay': 0.0, 'base_lr': base_lr * calib_lr_mult})
    base_params = [p for p in model.parameters() if id(p) not in head_ids]
    return groups, base_params, head_params


def train_one_phase(model, train_loader, val_loader, optimizer, scheduler, epochs: int, patience: int,
                    device: torch.device, best_path: str = "best_model.pth",
                    calibrator: OutputCalibrator | None = None, warmup_head_epochs: int = 0, base_params=None):
    stopper = EarlyStopping(patience=patience, verbose=True, path=best_path)
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema_decay = 0.99

    def _set_base_requires_grad(flag: bool):
        if base_params is None:
            return
        for p in base_params:
            p.requires_grad = flag

    for epoch in range(1, epochs + 1):
        if CONFIG.get('LR_WARMUP_EPOCHS', 0) and epoch <= CONFIG['LR_WARMUP_EPOCHS']:
            factor = float(epoch) / float(max(1, CONFIG['LR_WARMUP_EPOCHS']))
            for pg in optimizer.param_groups:
                base_lr = pg.get('base_lr', pg['lr'])
                pg['lr'] = base_lr * factor

        if epoch == 1 and warmup_head_epochs > 0:
            _set_base_requires_grad(False)
            print(f"[Warmup] Freeze backbone for first {warmup_head_epochs} epoch(s).")
        if epoch == warmup_head_epochs + 1 and warmup_head_epochs > 0:
            _set_base_requires_grad(True)
            if calibrator is not None:
                for p in calibrator.parameters():
                    p.requires_grad = False
            print("[Warmup] Unfreeze backbone & freeze calibrator.")

        model.train()
        if calibrator is not None:
            calibrator.train()
        running, start = 0.0, time.perf_counter()

        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as bar:
            for data_graph_batch, query_batch, y in bar:
                data_graph_batch, query_batch, y = move_batch_to_device(data_graph_batch, query_batch, y, device)

                with amp.autocast(device_type=device.type, enabled=USE_AMP):
                    log1p_pred = model_forward_log1p_pred(model, data_graph_batch, query_batch)
                    if calibrator is not None:
                        log1p_pred = calibrator(log1p_pred)

                    target_log1p = torch.log1p(y.float().clamp_min(0.0))
                    loss_msle = torch.mean((log1p_pred - target_log1p) ** 2)

                    qerr, pred, label = _qerror_tensor_from_log1p(log1p_pred, y, cap=CONFIG["QERR_CLIP_MAX"])
                    loss_qerr = qerr.mean()

                    s_signed = _signed_log10_qerr(qerr, pred, label)
                    loss_signed = _trimmed_mean_loss(_huber(s_signed, delta=CONFIG["LOG_HUBER_DELTA"]), CONFIG["TRIM_RATIO"])

                    loss = (CONFIG["LAMBDA_MSLE"] * loss_msle
                            + CONFIG["LAMBDA_QERR_MEAN"] * loss_qerr
                            + CONFIG["LAMBDA_SIGNED_LOGQ"] * loss_signed)

                    tail_weight = float(CONFIG.get("TAIL_QERR_PENALTY_WEIGHT", 0.0))
                    tail_penalty = torch.zeros((), device=log1p_pred.device, dtype=log1p_pred.dtype)
                    if tail_weight > 0:
                        log10_q = torch.log10(qerr.clamp_min(1.0))
                        thresh = float(CONFIG.get("TAIL_QERR_LOG10_THRESH", 1.0))
                        excess = torch.clamp(log10_q - thresh, min=0.0)
                        power = float(CONFIG.get("TAIL_QERR_PENALTY_POWER", 1.0))
                        if power != 1.0:
                            excess = excess.pow(power)
                        tail_penalty = excess.mean()
                        loss = loss + tail_weight * tail_penalty

                    if calibrator is not None and any(p.requires_grad for p in calibrator.parameters()):
                        loss = loss + CONFIG.get("CALIB_REG", 0.0) * ((calibrator.a - 1.0) ** 2 + (calibrator.b - 0.0) ** 2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                clip_gradients(model, max_norm=10.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if calibrator is not None:
                    torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        v_det = v.detach()
                        if k not in ema_state:
                            ema_state[k] = v_det.clone()
                        else:
                            ema_state[k].mul_(ema_decay).add_(v_det, alpha=1.0 - ema_decay)

                running += float(loss.item())
                postfix = dict(
                    loss=f"{running/(bar.n+1):.4f}",
                    msle=f"{loss_msle.item():.3f}",
                    qerr=f"{loss_qerr.item():.3f}",
                    slogq=f"{loss_signed.item():.3f}",
                )
                if tail_weight > 0:
                    postfix["tail"] = f"{tail_penalty.item():.3f}"
                bar.set_postfix(**postfix)

        print(f"Epoch {epoch}/{epochs}  TrainLoss={running/max(1,len(train_loader)):.6f}  time={time.perf_counter()-start:.1f}s")

        model.eval()
        if calibrator is not None:
            calibrator.eval()

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state, strict=False)

        qerr_sum, n_cnt = 0.0, 0
        abs_log10_list = []

        with torch.no_grad(), tqdm(val_loader, desc="Validation", unit="batch") as bar:
            for dg_batch, q_batch, y in bar:
                dg_batch, q_batch, y = move_batch_to_device(dg_batch, q_batch, y, device)
                log1p_pred = model_forward_log1p_pred(model, dg_batch, q_batch)
                if calibrator is not None:
                    log1p_pred = calibrator(log1p_pred)
                qerr, pred, label = _qerror_tensor_from_log1p(log1p_pred, y, cap=None)
                qerr_sum += qerr.sum().item()
                n_cnt += qerr.numel()

                s = _signed_log10_qerr(qerr, pred, label)
                abs_log10_list.append(torch.abs(s).detach().cpu().numpy())

        val_mean_qerr = qerr_sum / max(1, n_cnt)
        abs_log10_all = np.concatenate(abs_log10_list) if abs_log10_list else np.array([np.inf])
        med_abs = float(np.median(abs_log10_all))
        q3_abs = float(np.percentile(abs_log10_all, 75))

        model.load_state_dict(backup, strict=False)

        print(f"Val mean Q-Error = {val_mean_qerr:.6f}  |  |log10Q| median={med_abs:.4f}  Q3={q3_abs:.4f}")

        stopper(val_mean_qerr, _PackForSave(model, calibrator))
        if stopper.early_stop:
            print("Early stopping triggered.")
            break
        scheduler.step(val_mean_qerr)

    state = torch.load(best_path, map_location=device)
    model_keys = {k.split("model.", 1)[1]: v for k, v in state.items() if k.startswith("model.")}

    model.load_state_dict(model_keys or state, strict=False)
    return model, calibrator


def test(model, loader, device: torch.device, out_path: str = "predictions_and_labels.txt", calibrator: OutputCalibrator | None = None):
    """Run evaluation on the provided loader without Q-Error clipping."""
    model.eval()
    if calibrator is not None:
        calibrator.eval()
    total_mean_qerr, n = 0.0, 0
    abs_log10_all = []
    start_t = time.perf_counter()
    with open(out_path, "w") as f, tqdm(loader, desc="Testing", unit="batch") as bar:
        f.write("Batch,Prediction,Label,QError\n")
        for dg_batch, q_batch, y in bar:
            dg_batch, q_batch, y = move_batch_to_device(dg_batch, q_batch, y, device)
            with amp.autocast(device_type=device.type, enabled=USE_AMP):
                log1p_pred = model_forward_log1p_pred(model, dg_batch, q_batch)
                if calibrator is not None:
                    log1p_pred = calibrator(log1p_pred)
                qerr, pred, label = _qerror_tensor_from_log1p(log1p_pred, y, cap=None)
            total_mean_qerr += qerr.sum().item()
            n += label.numel()
            s = _signed_log10_qerr(qerr, pred, label)
            abs_log10_all.append(torch.abs(s).detach().cpu().numpy())
            for idx, (p, l, e) in enumerate(zip(pred.tolist(), label.tolist(), qerr.tolist()), 1):
                f.write(f"{idx},{p:.4f},{l:.4f},{e:.4f}\n")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_t
    avg_query_time = total_time / max(1, n)
    avg_q = total_mean_qerr / max(1, n)
    abs_log10_all = np.concatenate(abs_log10_all) if abs_log10_all else np.array([np.inf])
    val_med = float(np.median(abs_log10_all))
    q1, q3 = float(np.percentile(abs_log10_all, 25)), float(np.percentile(abs_log10_all, 75))
    val_iqr = q3 - q1
    print(f"Test Mean Q-Error {avg_q:.6f} | |log10Q| median={val_med:.4f} Q1={q1:.4f}  Q3={q3:.4f}  IQR={val_iqr:.4f} | Avg Query Time = {avg_query_time:.6f} s")
    with open(out_path, "a") as f:
        f.write(f"# total_queries: {n}\n")
        f.write(f"# total_test_time_sec: {total_time:.6f}\n")
        f.write(f"# average_query_time_sec: {avg_query_time:.6f}\n")
    return avg_q, val_med, q3, val_iqr
