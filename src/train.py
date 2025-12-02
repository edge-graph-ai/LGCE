"""Command-line entrypoint orchestrating training, ablations, and evaluation."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from src.config.train_config import CONFIG, apply_cli_overrides, apply_dataset_paths, bind_output_paths_to_selected_num
from src.data_preprocessing.load_data import load_graph_from_file
from src.data_preprocessing.prepare_data import prepare_data
from src.data_preprocessing.create_dataloader import create_dataloader
from src.data_preprocessing.load_pyg_data_and_build_dataset import load_prepared_pyg, build_dataset_from_prepared
from src.training.train_loop import (
    OutputCalibrator,
    build_model,
    make_param_groups,
    make_splits_indices_stratified,
    set_head_bias_to_mu,
    test,
    train_one_phase,
)
from src.utils.train_data_utils import (
    _maybe_resize_embeddings,
    _resize_to_fit_checkpoint,
    build_weighted_loader,
    resolve_query_dir,
)


def _preview_splits(prepared, splits, k_preview: int = 5) -> None:
    """Print a compact preview of the chosen pretrain/train/test splits."""
    qids = prepared['query_ids']
    name = lambda idxs: ', '.join(qids[i] for i in idxs[:k_preview])
    lines = [f"Global queries: {len(qids)}",
             f"Pretrain: {len(splits['pretrain_idx'])} e.g., [{name(splits['pretrain_idx'])}]"]
    for fi, fold in enumerate(splits['folds']):
        lines.append(
            f"Fold {fi + 1}: train={len(fold['train_idx'])} (e.g., [{name(fold['train_idx'])}]) | "
            f"test={len(fold['val_idx'])} (e.g., [{name(fold['val_idx'])}])"
        )
    print('\n'.join(lines))


def main():
    """Parse CLI flags, prepare data, then run pretraining and K-fold fine-tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file overriding defaults")
    parser.add_argument("--enable-shortcut", dest="enable_shortcut", action="store_true",
                        help="Force enable embed shortcut gating")
    parser.add_argument("--disable-shortcut", dest="enable_shortcut", action="store_false",
                        help="Disable embed shortcut gating")
    parser.add_argument("--shortcut-init", type=float, default=None, help="Initial logit for embed shortcut gate")
    parser.add_argument("--enable-multi-scale", dest="enable_multi_scale", action="store_true",
                        help="Enable multi-scale pooling wrapper")
    parser.add_argument("--disable-multi-scale", dest="enable_multi_scale", action="store_false",
                        help="Disable multi-scale pooling wrapper")
    parser.add_argument("--multi-scale-fusion", type=str, choices=["gate", "concat"], default=None,
                        help="Fusion strategy between pooling branches")
    parser.add_argument("--multi-scale-dropout", type=float, default=None,
                        help="Dropout applied on fused multi-scale features")
    parser.add_argument("--multi-scale-gate-init", type=float, default=None,
                        help="Initial logit for multi-scale fusion gate")
    parser.add_argument("--multi-scale-attn-hidden", type=int, default=None,
                        help="Hidden size for attention pooling within multi-scale module")
    parser.add_argument("--enable-cross-attn", dest="enable_cross_attn", action="store_true",
                        help="Enable the lightweight cross-attention head between query and memory tokens")
    parser.add_argument("--disable-cross-attn", dest="enable_cross_attn", action="store_false",
                        help="Disable the lightweight cross-attention head")
    parser.add_argument("--cross-attn-heads", type=int, default=None, help="Number of heads used by the lightweight cross-attention")
    parser.add_argument("--cross-attn-dropout", type=float, default=None, help="Dropout applied inside the lightweight cross-attention")
    parser.add_argument("--cross-ffn-hidden", type=int, default=None, help="Hidden size of the feed-forward fusion layer after cross-attention")
    parser.add_argument("--enable-mem-pos", dest="enable_mem_pos", action="store_true",
                        help="Inject learnable positional encodings into memory tokens")
    parser.add_argument("--disable-mem-pos", dest="enable_mem_pos", action="store_false",
                        help="Disable positional encodings for memory tokens")
    parser.add_argument("--variant", type=str, default=None,
                        help="选择消融：BASE | NO_GIN | NO_ATTENTION | NO_GIN_NO_ATTENTION （兼容：NO_ENCODER | NO_DECODER）")
    parser.add_argument("--selected-query-num", type=int, default=None,
                        help="Override CONFIG['SELECTED_QUERY_NUM'] (默认 24，例如 4、8、12 等)")
    parser.set_defaults(enable_shortcut=None, enable_multi_scale=None, enable_cross_attn=None, enable_mem_pos=None)
    args = parser.parse_args()

    apply_cli_overrides(args)
    if args.variant:
        CONFIG["MODEL_VARIANT"] = args.variant
    CONFIG.update(bind_output_paths_to_selected_num(apply_dataset_paths(CONFIG)))

    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["SEED"])
    device = torch.device("cuda:0" if (CONFIG["DEVICE"] == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Using", device, "| Variant =", CONFIG["MODEL_VARIANT"], "| SELECTED_QUERY_NUM =", CONFIG["SELECTED_QUERY_NUM"])

    query_dir = resolve_query_dir(CONFIG["QUERY_ROOT"], CONFIG["QUERY_NUM_DIR"], CONFIG["QUERY_DIR"])
    if not os.path.exists(CONFIG["PREPARED_OUT"]):
        print(f"[Info] {CONFIG['PREPARED_OUT']} 不存在，调用 prepare_data() 生成 …")
        Path(CONFIG["PREPARED_OUT"]).parent.mkdir(parents=True, exist_ok=True)
        prepare_data(device=("cuda" if device.type == "cuda" else "cpu"),
                     data_graph_filename=CONFIG["DATA_GRAPH"],
                     query_graph_folder=query_dir,
                     matches_path=CONFIG["MATCHES_PATH"],
                     output_path=CONFIG["PREPARED_OUT"])
    else:
        print(f"[Info] 使用已存在的 {CONFIG['PREPARED_OUT']}")

    prepared = load_prepared_pyg(CONFIG["PREPARED_OUT"])
    dataset = build_dataset_from_prepared(prepared, repeat_idx=0)
    num_subgraphs = len(prepared['pyg_data_graphs'][0])
    print(f"[Info] num_subgraphs = {num_subgraphs}")

    # Inspect bounds to pre-resize embeddings when possible.
    global_num_vertices, global_num_labels = None, None
    vmins, vmaxs = [], []
    for g in prepared['pyg_data_graphs'][0]:
        if 'vertex_ids' in g and g['vertex_ids'].numel() > 0:
            vmins.append(int(g['vertex_ids'].min().item()))
            vmaxs.append(int(g['vertex_ids'].max().item()))
    if vmins and vmaxs:
        global_num_vertices = max(vmaxs) + 1
        print(f"[VID] min(vertex_id) over subgraphs = {min(vmins)}")
        print(f"[VID] max(vertex_id) over subgraphs = {max(vmaxs)}")
        print(f"[EmbedSize] using num_vertices_data = {global_num_vertices}  (from prepared vertex_ids)")

    lab_vals = []
    neg1 = False
    for g in prepared['pyg_data_graphs'][0]:
        if 'labels' in g and g['labels'].numel() > 0:
            lab_vals.append(int(g['labels'].max().item()))
            if int(g['labels'].min().item()) < 0:
                neg1 = True
    for q in prepared['pyg_query_graphs']:
        if 'labels' in q and q['labels'].numel() > 0:
            lab_vals.append(int(q['labels'].max().item()))
            if int(q['labels'].min().item()) < 0:
                neg1 = True
    if lab_vals:
        mx_lab = max(lab_vals)
        if neg1:
            print("[LABEL] min(label) = -1")
        print(f"[LABEL] max(label) = {mx_lab}")
        global_num_labels = mx_lab + 1
        print(f"[LabelSize] using num_labels = {global_num_labels}  (max(label)+1 across data+query)")

    print("[Verify] scanning graphs for out-of-range indices ...")
    bad = 0
    for g in prepared['pyg_data_graphs'][0]:
        if 'edge_index' in g and 'labels' in g and g['edge_index'].numel() > 0:
            E = g['edge_index'].size(1)
            N = g['labels'].size(0)
            if E:
                mx = int(g['edge_index'].max().item())
                mn = int(g['edge_index'].min().item())
                if mx >= N or mn < 0:
                    bad += 1
    print("[Verify] OK: no out-of-range indices." if bad == 0 else f"[Verify] FOUND {bad} data subgraphs with out-of-range edge_index! (filtered on-the-fly)")

    print("[Verify] scanning query graphs for out-of-range indices ...")
    bad_q = 0
    for q in prepared['pyg_query_graphs']:
        if 'edge_index' in q and 'labels' in q and q['edge_index'].numel() > 0:
            E = q['edge_index'].size(1)
            N = q['labels'].size(0)
            if E:
                mx = int(q['edge_index'].max().item())
                mn = int(q['edge_index'].min().item())
                if mx >= N or mn < 0:
                    bad_q += 1
    print("[Verify] OK: no out-of-range indices in queries." if bad_q == 0 else f"[Verify] FOUND {bad_q} query graphs out-of-range! (filtered on-the-fly)")

    splits = make_splits_indices_stratified(prepared, selected_query_num=CONFIG["SELECTED_QUERY_NUM"],
                                            pretrain_ratio=CONFIG["PRETRAIN_RATIO"], k_folds=CONFIG["K_FOLDS"],
                                            seed=CONFIG["SEED"], exclude_overlap=CONFIG["EXCLUDE_OVERLAP"],
                                            n_bins=CONFIG["STRATA_NUM_BINS"])
    _preview_splits(prepared, splits)

    pre_tr = splits['pretrain_idx']
    pre_va = splits['folds'][0]['val_idx']

    pre_train_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=pre_tr, shuffle=True,
                                     seed=CONFIG["SEED"], num_workers=CONFIG["NUM_WORKERS"],
                                     pin_memory=CONFIG["PIN_MEMORY"])
    pre_val_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=pre_va, shuffle=False,
                                   seed=CONFIG["SEED"], num_workers=CONFIG["NUM_WORKERS"],
                                   pin_memory=CONFIG["PIN_MEMORY"])

    data_graph = load_graph_from_file(CONFIG["DATA_GRAPH"])
    model = build_model(device, data_graph, num_subgraphs=num_subgraphs)
    _maybe_resize_embeddings(model, global_num_vertices, global_num_labels)

    if os.path.exists(CONFIG["PRETRAIN_WEIGHTS"]):
        state = torch.load(CONFIG["PRETRAIN_WEIGHTS"], map_location=device)
        model_keys = {k.split("model.", 1)[1]: v for k, v in state.items() if k.startswith("model.")}
        if not model_keys:
            model_keys = state
        _resize_to_fit_checkpoint(model, model_keys)
        model.load_state_dict(model_keys, strict=False)

    pre_calibrator = OutputCalibrator() if CONFIG["USE_CALIBRATOR"] else None
    y_pre_train = np.array([prepared['true_cardinalities'][i] for i in pre_tr], dtype=np.float64)
    mu_pre = float((np.median if CONFIG["USE_MEDIAN_BIAS"] else np.mean)(np.log1p(y_pre_train))) if y_pre_train.size else 0.0
    set_head_bias_to_mu(model, mu_pre)

    groups, base_params, _ = make_param_groups(model, CONFIG["LR"], CONFIG["WEIGHT_DECAY"],
                                               head_lr_mult=CONFIG["HEAD_LR_MULT"], calibrator=pre_calibrator,
                                               calib_lr_mult=CONFIG["CALIB_LR_MULT"])
    opt = torch.optim.AdamW(groups)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=max(1, CONFIG["PATIENCE"] // 2), factor=0.5)

    print("\n========== Stage 1: Pretrain (MSLE + meanQ + robust signed log10Q, 20% of ALL queries) ==========")
    model, pre_calibrator = train_one_phase(model, pre_train_dl, pre_val_dl, opt, sch, epochs=CONFIG["EPOCHS_PRE"],
                                            patience=CONFIG["PATIENCE"], device=device, best_path=CONFIG["BEST_PRE_PATH"],
                                            calibrator=pre_calibrator, warmup_head_epochs=0, base_params=base_params)
    torch.save(torch.nn.ModuleDict(dict(model=model, calibrator=pre_calibrator if pre_calibrator else torch.nn.Identity())).state_dict(), CONFIG["PRETRAIN_WEIGHTS"])
    print(f"保存预训练权重 -> {CONFIG['PRETRAIN_WEIGHTS']}")

    print("\n========== Stage 2: K-Fold Fine-tuning (only SELECTED_QUERY_NUM) ==========")
    fold_qerrs, fold_box_q3, fold_box_med = [], [], []
    INNER_VAL_RATIO = 0.10

    for i, fold in enumerate(splits['folds'], 1):
        tr_idx_main, test_idx = fold['train_idx'], fold['val_idx']
        print(f"\n---- Fold {i}/{len(splits['folds'])}  train(main)={len(tr_idx_main)}  test={len(test_idx)} ----")

        rng_fold = np.random.default_rng(CONFIG["SEED"] + i)
        tr_shuf = tr_idx_main.copy()
        rng_fold.shuffle(tr_shuf)
        cut = max(1, int(round(INNER_VAL_RATIO * len(tr_shuf))))
        inner_val_idx = tr_shuf[:cut]
        inner_train_idx = tr_shuf[cut:]
        print(f"[Fold {i}] inner-train={len(inner_train_idx)}  inner-val={len(inner_val_idx)}")

        if CONFIG["USE_WEIGHTED_SAMPLER"]:
            ft_train_dl = build_weighted_loader(dataset, inner_train_idx, batch_size=CONFIG["BATCH_SIZE"],
                                                seed=CONFIG["SEED"], strata_bins=CONFIG["STRATA_NUM_BINS"],
                                                replacement=CONFIG["SAMPLE_REPLACEMENT"], num_workers=CONFIG["NUM_WORKERS"],
                                                pin_memory=False)
        else:
            ft_train_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=inner_train_idx, shuffle=True,
                                            seed=CONFIG["SEED"], pin_memory=False)

        ft_inner_val_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=inner_val_idx, shuffle=False,
                                            seed=CONFIG["SEED"], pin_memory=False)
        ft_test_dl = create_dataloader(dataset, CONFIG["BATCH_SIZE"], indices=test_idx, shuffle=False,
                                       seed=CONFIG["SEED"], pin_memory=False)

        model_k = build_model(device, data_graph, num_subgraphs=num_subgraphs)
        _maybe_resize_embeddings(model_k, global_num_vertices, global_num_labels)

        pre_path = CONFIG["BEST_PRE_PATH"] if os.path.exists(CONFIG["BEST_PRE_PATH"]) else CONFIG["PRETRAIN_WEIGHTS"]
        if os.path.exists(pre_path):
            state_pre = torch.load(pre_path, map_location=device)
            model_keys = {k.split("model.", 1)[1]: v for k, v in state_pre.items() if k.startswith("model.")}
            if not model_keys:
                model_keys = state_pre
            _resize_to_fit_checkpoint(model_k, model_keys)
            model_k.load_state_dict(model_keys, strict=False)

        calibrator_k = OutputCalibrator() if CONFIG["USE_CALIBRATOR"] else None
        groups, base_params_k, _ = make_param_groups(model_k, CONFIG["LR"], CONFIG["WEIGHT_DECAY"],
                                                     head_lr_mult=CONFIG["HEAD_LR_MULT"], calibrator=calibrator_k,
                                                     calib_lr_mult=CONFIG["CALIB_LR_MULT"])
        opt_k = torch.optim.AdamW(groups)
        sch_k = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_k, mode="min", patience=CONFIG["PATIENCE"], factor=0.5)

        best_fold_path = CONFIG["BEST_FOLD_PATH_TPL"].format(fold=i)
        model_k, calibrator_k = train_one_phase(model_k, ft_train_dl, ft_inner_val_dl, opt_k, sch_k,
                                                epochs=CONFIG["EPOCHS_FT"], patience=CONFIG["PATIENCE"], device=device,
                                                best_path=best_fold_path, calibrator=calibrator_k,
                                                warmup_head_epochs=CONFIG["WARMUP_HEAD_EPOCHS"], base_params=base_params_k)

        pred_txt = CONFIG["PRED_TXT_TPL"].format(fold=i)
        avg_q, med_log, q3_log, iqr_log = test(model_k, ft_test_dl, device, out_path=pred_txt, calibrator=calibrator_k)
        fold_qerrs.append(avg_q)
        fold_box_med.append(med_log)
        fold_box_q3.append(q3_log)

    if fold_qerrs:
        print("\n========== K-Fold Summary ==========")
        for i, (q, medl, q3l) in enumerate(zip(fold_qerrs, fold_box_med, fold_box_q3), 1):
            print(f"Fold {i}: Mean Q-Err = {q:.6f} | |log10Q| median={medl:.4f} Q3={q3l:.4f}")
        print(f"Overall mean Q-Error = {np.mean(fold_qerrs):.6f}")
        print(f"Overall |log10Q| median = {np.mean(fold_box_med):.4f}  Q3 = {np.mean(fold_box_q3):.4f}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
