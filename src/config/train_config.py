"""Training configuration helpers and defaults."""
import os
from typing import Any, Dict

try:  # Optional dependency used only when loading external YAML configs
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Default configuration used by the training script.
CONFIG: Dict[str, Any] = dict(
    DATASET="yeast",
    DATA_GRAPH=None,
    MATCHES_PATH=None,
    PREPARED_OUT=None,
    QUERY_DIR=None,
    QUERY_ROOT=None,
    QUERY_NUM_DIR=None,

    SELECTED_QUERY_NUM=24,

    DEVICE="cuda",
    SEED=42,
    BATCH_SIZE=64,
    LR=1e-3,
    WEIGHT_DECAY=1e-4,
    EPOCHS_PRE=50,
    EPOCHS_FT=50,
    PATIENCE=10,
    PRETRAIN_RATIO=0.20,
    PRETRAIN_VAL_RATIO=0.10,
    K_FOLDS=5,
    EXCLUDE_OVERLAP=True,
    NUM_WORKERS=0,
    PIN_MEMORY=False,

    USE_MEDIAN_BIAS=True,
    HEAD_LR_MULT=10.0,
    QERR_CLIP_MAX=1e4,

    ENABLE_EMBED_SHORTCUT=True,
    EMBED_SHORTCUT_INIT=0.0,
    USE_MULTI_SCALE_POOL=True,
    MULTI_SCALE_FUSION="gate",
    MULTI_SCALE_DROPOUT=None,
    MULTI_SCALE_GATE_INIT=0.0,
    MULTI_SCALE_ATTN_HIDDEN=None,

    ENABLE_CROSS_ATTENTION=True,
    CROSS_ATTN_HEADS=1,
    CROSS_ATTN_DROPOUT=None,
    CROSS_FFN_HIDDEN=None,
    USE_MEMORY_POS_ENCODING=True,

    LAMBDA_MSLE=0.2,
    LAMBDA_QERR_MEAN=0.6,
    LAMBDA_SIGNED_LOGQ=0.2,

    LOG_HUBER_DELTA=0.25,
    TRIM_RATIO=0.05,

    TAIL_QERR_LOG10_THRESH=4.0,
    TAIL_QERR_PENALTY_WEIGHT=0.3,
    TAIL_QERR_PENALTY_POWER=2.0,

    STRATA_NUM_BINS=5,
    USE_WEIGHTED_SAMPLER=True,
    SAMPLE_REPLACEMENT=True,

    USE_CALIBRATOR=False,
    CALIB_LR_MULT=15.0,
    CALIB_REG=1e-4,
    WARMUP_HEAD_EPOCHS=5,

    LR_WARMUP_EPOCHS=5,

    PRETRAIN_WEIGHTS="pretrained.pth",
    BEST_PRE_PATH="best_model_pretrain.pth",
    BEST_FOLD_PATH_TPL="best_model_fold{fold}.pth",
    PRED_TXT_TPL="predictions_fold{fold}.txt",
    RESULT_DIR=None,

    MODEL_VARIANT="BASE",
)


def apply_dataset_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Populate dataset-dependent paths onto the config mapping."""
    dataset = config["DATASET"]
    base = os.path.join("..", "data", "train_data", dataset)
    config["DATA_GRAPH"] = os.path.join(base, "data_graph", f"{dataset}.graph")
    config["MATCHES_PATH"] = os.path.join(base, "matches_output.txt")
    config["PREPARED_OUT"] = os.path.join(base, "prepared_pyg_data.pt")
    config["QUERY_ROOT"] = config["QUERY_DIR"] or os.path.join(base, "query_graph")
    return config


def bind_output_paths_to_selected_num(config: Dict[str, Any]) -> Dict[str, Any]:
    """Bind output file names to the currently selected query count."""
    dataset = config["DATASET"]
    num = config["SELECTED_QUERY_NUM"]
    variant = (config.get("MODEL_VARIANT") or "BASE").strip().lower().replace(" ", "_")
    out_dir = os.path.join("ablation", dataset, variant, str(num))
    os.makedirs(out_dir, exist_ok=True)

    config["PRETRAIN_WEIGHTS"] = os.path.join(out_dir, f"pretrained_q{num}.pth")
    config["BEST_PRE_PATH"] = os.path.join(out_dir, f"best_model_pretrain_q{num}.pth")
    config["BEST_FOLD_PATH_TPL"] = os.path.join(out_dir, f"best_model_q{num}_fold{{fold}}.pth")
    config["PRED_TXT_TPL"] = os.path.join(out_dir, f"predictions_q{num}_fold{{fold}}.txt")
    config["RESULT_DIR"] = out_dir
    return config


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load configuration overrides from a YAML file if provided."""
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files. Please install 'pyyaml'.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data)!r}")
    return data


def apply_cli_overrides(args, config: Dict[str, Any] = CONFIG) -> None:
    """Apply CLI-provided overrides directly onto the shared config dict."""
    if args is None:
        return
    if getattr(args, "config", None):
        config.update(load_yaml_config(args.config))
    if getattr(args, "dataset", None):
        config["DATASET"] = str(args.dataset)
    if getattr(args, "enable_shortcut", None) is not None:
        config["ENABLE_EMBED_SHORTCUT"] = bool(args.enable_shortcut)
    if getattr(args, "shortcut_init", None) is not None:
        config["EMBED_SHORTCUT_INIT"] = float(args.shortcut_init)
    if getattr(args, "enable_multi_scale", None) is not None:
        config["USE_MULTI_SCALE_POOL"] = bool(args.enable_multi_scale)
    if getattr(args, "multi_scale_fusion", None):
        config["MULTI_SCALE_FUSION"] = str(args.multi_scale_fusion)
    if getattr(args, "multi_scale_dropout", None) is not None:
        config["MULTI_SCALE_DROPOUT"] = float(args.multi_scale_dropout)
    if getattr(args, "multi_scale_gate_init", None) is not None:
        config["MULTI_SCALE_GATE_INIT"] = float(args.multi_scale_gate_init)
    if getattr(args, "multi_scale_attn_hidden", None) is not None:
        config["MULTI_SCALE_ATTN_HIDDEN"] = int(args.multi_scale_attn_hidden)
    if getattr(args, "enable_cross_attn", None) is not None:
        config["ENABLE_CROSS_ATTENTION"] = bool(args.enable_cross_attn)
    if getattr(args, "cross_attn_heads", None) is not None:
        config["CROSS_ATTN_HEADS"] = int(args.cross_attn_heads)
    if getattr(args, "cross_attn_dropout", None) is not None:
        config["CROSS_ATTN_DROPOUT"] = float(args.cross_attn_dropout)
    if getattr(args, "cross_ffn_hidden", None) is not None:
        config["CROSS_FFN_HIDDEN"] = int(args.cross_ffn_hidden)
    if getattr(args, "enable_mem_pos", None) is not None:
        config["USE_MEMORY_POS_ENCODING"] = bool(args.enable_mem_pos)
    if getattr(args, "selected_query_num", None) is not None:
        config["SELECTED_QUERY_NUM"] = int(args.selected_query_num)
