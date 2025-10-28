import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# 配置（按你的目录约定）
# -----------------------------
dataset = "eu2005"  # ← 原来的 'youtube' 改成变量
BASE_DIR = os.path.join("compare", dataset)
EXPERIMENTS = ["4", "8"]
MODELS = ["FlowSC", "LGCE"]
OUTPUT_PNG = f"{dataset}_avg_query_time_flowsc_vs_lgce.png"

# LGCE 结果文件：文件尾部有三行注释
#   # total_queries: 20
#   # total_test_time_sec: 2.939422
#   # average_query_time_sec: 0.146971
LGCE_TOTAL_QUERIES_PAT = re.compile(r"#\s*total_queries:\s*([0-9]+)")
LGCE_TOTAL_TIME_PAT    = re.compile(r"#\s*total_test_time_sec:\s*([0-9]*\.?[0-9]+)")
LGCE_AVG_TIME_PAT      = re.compile(r"#\s*average_query_time_sec:\s*([0-9]*\.?[0-9]+)")

# FlowSC：fold 块内
#   多条： dataset/... .graph: <qerror> <under/over>
#   末尾有： average query time: X
FLOW_GRAPH_LINE_PAT = re.compile(r""".*?\.graph:\s*[0-9]*\.?[0-9]+\s+(?:underestimate|overestimate)\s*$""", re.IGNORECASE)
FLOW_AVG_TIME_PAT   = re.compile(r"""average\s+query\s+time:\s*([0-9]*\.?[0-9]+)""", re.IGNORECASE)
FLOW_FOLD_PAT       = re.compile(r"""^fold:\s*\d+""", re.IGNORECASE)

def _glob_unique(patterns, dirpath):
    """按顺序尝试多个通配模式并去重，返回文件列表（不含目录）。"""
    seen = set()
    files = []
    for pat in patterns:
        for fp in sorted(glob.glob(os.path.join(dirpath, pat))):
            if not os.path.isdir(fp) and fp not in seen:
                seen.add(fp)
                files.append(fp)
    return files

def parse_lgce_overall_avg(dirpath: str) -> float | None:
    """overall_avg = (sum total_test_time_sec) / (sum total_queries)"""
    total_queries_sum = 0
    total_time_sum = 0.0

    patterns = [
        "predictions_q*_fold*.txt",
        "predictions_*fold*.txt",
        "predictions_fold*.txt",
        "*.txt"
    ]
    files = _glob_unique(patterns, dirpath)

    for fp in files:
        tq, tt = None, None
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m1 = LGCE_TOTAL_QUERIES_PAT.search(line)
                    if m1:
                        tq = int(m1.group(1))
                        continue
                    m2 = LGCE_TOTAL_TIME_PAT.search(line)
                    if m2:
                        tt = float(m2.group(1))
                        continue
            if tq is not None and tt is not None and tq > 0:
                total_queries_sum += tq
                total_time_sum += tt
        except Exception:
            pass

    if total_queries_sum > 0:
        return total_time_sum / total_queries_sum
    return None

def parse_flowsc_overall_avg(dirpath: str) -> float | None:
    """
    对每个 fold：
      m = 该 fold 的 .graph 行数（查询数）
      X = 'average query time'（秒/查询）
      fold_total_time ≈ X * m
    overall_avg = (sum fold_total_time) / (sum m)
    """
    total_queries_sum = 0
    total_time_sum = 0.0
    files = sorted(glob.glob(os.path.join(dirpath, "*")))
    for fp in files:
        if os.path.isdir(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                m_count = 0
                avg_time = None

                def flush_fold():
                    nonlocal total_queries_sum, total_time_sum, m_count, avg_time
                    if avg_time is not None and m_count > 0:
                        total_queries_sum += m_count
                        total_time_sum += avg_time * m_count
                    m_count = 0

                for line in f:
                    if FLOW_FOLD_PAT.search(line):
                        flush_fold()
                        avg_time = None
                        continue
                    if FLOW_GRAPH_LINE_PAT.search(line):
                        m_count += 1
                        continue
                    m = FLOW_AVG_TIME_PAT.search(line)
                    if m:
                        try:
                            avg_time = float(m.group(1))
                        except:
                            avg_time = None
                        continue

                # 文件末尾最后一个 fold
                if m_count > 0 and avg_time is not None:
                    total_queries_sum += m_count
                    total_time_sum += avg_time * m_count
        except Exception:
            pass

    if total_queries_sum > 0:
        return total_time_sum / total_queries_sum
    return None

def main():
    # 收集每个 #nodes 的两模型总体平均查询时间
    avg_time = defaultdict(dict)  # avg_time[exp][model] = value
    for exp in EXPERIMENTS:
        flowsc_dir = os.path.join(BASE_DIR, exp, "FlowSC")
        lgce_dir   = os.path.join(BASE_DIR, exp, "LGCE")
        flowsc_avg = parse_flowsc_overall_avg(flowsc_dir) if os.path.exists(flowsc_dir) else None
        lgce_avg   = parse_lgce_overall_avg(lgce_dir)     if os.path.exists(lgce_dir)   else None
        avg_time[exp]["FlowSC"] = flowsc_avg
        avg_time[exp]["LGCE"]   = lgce_avg
        print(f"#nodes={exp} -> FlowSC avg(s): {flowsc_avg},  LGCE avg(s): {lgce_avg}")

    # 画分组柱状图
    xs = np.arange(len(EXPERIMENTS), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    flowsc_vals = [avg_time[exp].get("FlowSC") for exp in EXPERIMENTS]
    lgce_vals   = [avg_time[exp].get("LGCE")   for exp in EXPERIMENTS]
    flowsc_vals = [np.nan if v is None else v for v in flowsc_vals]
    lgce_vals   = [np.nan if v is None else v for v in lgce_vals]

    # 指定颜色
    COLOR_FLOW = "#8FB3D9"  # FlowSC
    COLOR_LGCE = "#90C4B8"  # LGCE

    # 画柱
    ax.bar(xs - width/2, flowsc_vals, width, label="FlowSC",
           color=COLOR_FLOW, edgecolor=COLOR_FLOW, alpha=0.95)
    ax.bar(xs + width/2, lgce_vals,   width, label="LGCE",
           color=COLOR_LGCE, edgecolor=COLOR_LGCE, alpha=0.95)

    ax.set_xticks(xs)
    ax.set_xticklabels(EXPERIMENTS)
    ax.set_xlabel("#nodes")
    ax.set_ylabel("Average query time (seconds)")
    ax.set_title(f"{dataset}: Average query time by #nodes (FlowSC vs LGCE)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 自定义图例（挪到图外右侧并缩小）
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_FLOW, lw=10, alpha=0.9),
        plt.Line2D([0], [0], color=COLOR_LGCE, lw=10, alpha=0.9),
    ]
    ax.legend(
        legend_handles, ["FlowSC", "LGCE"],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        handleheight=1.0
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
