# -*- coding: utf-8 -*-
import os
import re
import sys
import csv
import time
import shutil
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 解析标准输出中的关键指标（尽量鲁棒）
PAT_QERR_OVERALL = re.compile(r"Overall mean Q-Error\s*=\s*([0-9.]+)")
PAT_TEST_MEAN_Q  = re.compile(r"Test Mean Q-Error\s+([0-9.]+)")
PAT_VAL_LOSS     = re.compile(r"Val Loss:\s*([0-9.]+)")
PAT_LOGQ_MED     = re.compile(r"logQError median=([0-9.]+)")
PAT_LOGQ_Q3      = re.compile(r"Q3=([0-9.]+)")

def run_one_trial(
    train_py: Path,
    transformer_layers: int,
    python_exec: str = sys.executable,
    extra_env: dict | None = None,
    log_dir: Path | None = None,
    timeout_hours: float | None = None,
) -> dict:
    """
    启动一次 train.py 训练，返回解析得到的指标字典：
    {
        'transformer_layers': int,
        'overall_mean_qerr': float | None,
        'test_mean_qerr': float | None,
        'val_loss': float | None,
        'logq_median': float | None,
        'logq_q3': float | None,
        'elapsed_sec': float,
        'ok': bool,
        'log_file': str
    }
    """
    tag = f"tl{transformer_layers}"
    env = os.environ.copy()
    env["TRANSFORMER_LAYERS"] = str(transformer_layers)
    env["RUN_TAG"] = tag
    if extra_env:
        env.update(extra_env)

    # 日志目录
    log_dir = log_dir or (HERE / "tuning_logs" / "transformer_layers")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{tag}.log"

    cmd = [python_exec, str(train_py)]
    print(f"[TUNE] Running: TRANSFORMER_LAYERS={transformer_layers}  (log: {log_file})")

    start = time.perf_counter()
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            # 同时把 stdout/stderr 写入文件，也回显到控制台
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, cwd=HERE, text=True, bufsize=1
            )
            # 逐行读取，边写边回显
            for line in proc.stdout:
                lf.write(line)
                print(line, end="")
            proc.wait(timeout=None if timeout_hours is None else timeout_hours * 3600)
            rc = proc.returncode
    except Exception as e:
        print(f"[TUNE][{tag}] ERROR: {e}")
        rc = -1
    elapsed = time.perf_counter() - start

    # 解析日志
    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""

    overall_mean_qerr = None
    test_mean_qerr    = None
    val_loss          = None
    logq_median       = None
    logq_q3           = None

    m = PAT_QERR_OVERALL.search(text)
    if m: overall_mean_qerr = float(m.group(1))

    m = PAT_TEST_MEAN_Q.search(text)
    if m: test_mean_qerr = float(m.group(1))

    # 取最后一次出现的 Val Loss
    vals = PAT_VAL_LOSS.findall(text)
    if vals:
        try:
            val_loss = float(vals[-1])
        except Exception:
            pass

    m = PAT_LOGQ_MED.search(text)
    if m: logq_median = float(m.group(1))

    m = PAT_LOGQ_Q3.search(text)
    if m: logq_q3 = float(m.group(1))

    ok = (rc == 0)
    return dict(
        transformer_layers=transformer_layers,
        overall_mean_qerr=overall_mean_qerr,
        test_mean_qerr=test_mean_qerr,
        val_loss=val_loss,
        logq_median=logq_median,
        logq_q3=logq_q3,
        elapsed_sec=elapsed,
        ok=ok,
        log_file=str(log_file),
    )

def pick_score(row: dict, prefer: str = "overall_mean_qerr") -> float | None:
    """
    选用一个标量得分用于排序（越小越好）。
    优先级：
      1) overall_mean_qerr
      2) test_mean_qerr
      3) val_loss
    """
    keys = {
        "overall_mean_qerr": row.get("overall_mean_qerr"),
        "test_mean_qerr":    row.get("test_mean_qerr"),
        "val_loss":          row.get("val_loss"),
    }
    for k in [prefer, "overall_mean_qerr", "test_mean_qerr", "val_loss"]:
        v = keys.get(k)
        if v is not None:
            return float(v)
    return None

def main():
    ap = argparse.ArgumentParser("Auto-tune transformer_layers")
    ap.add_argument("--train_py", type=str, default=str(HERE / "train.py"),
                    help="train.py 的路径")
    ap.add_argument("--grid", type=str, default="1,2,3,4,6,8",
                    help="要搜索的层数组合，用逗号分隔，如 1,2,3,4")
    ap.add_argument("--prefer_metric", type=str, default="overall_mean_qerr",
                    choices=["overall_mean_qerr","test_mean_qerr","val_loss"],
                    help="选最优的主指标（越小越好）")
    ap.add_argument("--timeout_hours", type=float, default=None,
                    help="单次 trial 的超时时间（小时），默认不限")
    ap.add_argument("--csv", type=str, default=str(HERE / "tuning_results.csv"),
                    help="结果 CSV 输出路径")
    args = ap.parse_args()

    train_py = Path(args.train_py).resolve()
    if not train_py.exists():
        print(f"[ERROR] train.py not found: {train_py}")
        sys.exit(1)

    grid = [int(x) for x in args.grid.split(",") if x.strip()]
    results = []

    for tl in grid:
        row = run_one_trial(
            train_py=train_py,
            transformer_layers=tl,
            python_exec=sys.executable,
            extra_env=None,
            log_dir=HERE / "tuning_logs" / "transformer_layers",
            timeout_hours=args.timeout_hours,
        )
        # 计算可排序的分数
        row["_score"] = pick_score(row, prefer=args.prefer_metric)
        results.append(row)

    # 写 CSV
    header = ["transformer_layers", "overall_mean_qerr", "test_mean_qerr",
              "val_loss", "logq_median", "logq_q3", "elapsed_sec", "ok", "_score", "log_file"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in header})
    print(f"[TUNE] Saved summary -> {args.csv}")

    # 选择最优
    ok_rows = [r for r in results if r.get("ok") and (r.get("_score") is not None)]
    if not ok_rows:
        print("[TUNE] No successful runs with valid score.")
        sys.exit(2)

    best = min(ok_rows, key=lambda r: r["_score"])
    print("\n====== Auto Tune Result ======")
    print(f"Best transformer_layers = {best['transformer_layers']}")
    print(f"Score ({args.prefer_metric}) = {best['_score']:.6f}")
    print(f"Log file: {best['log_file']}")
    print("==============================")

if __name__ == "__main__":
    main()
