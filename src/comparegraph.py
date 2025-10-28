import os
import re
import math
import glob
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

# -----------------------------
# 配置
# -----------------------------
dataset = "eu2005"  # ← 把原来的 'youtube' 改成变量
BASE_DIR = os.path.join("compare", dataset)
EXPERIMENTS = ["4", "8"]      # 五个实验子目录（表示节点数）
MODELS = ["FlowSC", "LGCE"]                  # 两个模型
OUTPUT_PNG = f"{dataset}_flowsc_lgce_signed_log_qerror_boxplot.png"

# -----------------------------
# 工具函数
# -----------------------------
GRAPH_LINE_PAT = re.compile(
    r""".*?\.graph:\s*([0-9]*\.?[0-9]+)\s+(underestimate|overestimate)\s*$""",
    re.IGNORECASE
)

def parse_flowsc_file(filepath):
    """解析 FlowSC 日志文件中每条 .graph 记录的 qerror 与 under/over 标记"""
    results = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GRAPH_LINE_PAT.match(line.strip())
            if m:
                qerr = float(m.group(1))
                est = m.group(2).lower()
                sign = 1 if est == "overestimate" else -1
                results.append((qerr, sign))
    return results

def _glob_unique(patterns, dirpath):
    """按顺序用多种模式匹配并去重"""
    seen, files = set(), []
    for pat in patterns:
        for fp in sorted(glob.glob(os.path.join(dirpath, pat))):
            if not os.path.isdir(fp) and fp not in seen:
                seen.add(fp)
                files.append(fp)
    return files

def parse_lgce_dir(dirpath):
    """
    解析 LGCE 目录：仅匹配 fold 预测文件（支持新旧命名）
    - 新：predictions_q20_fold1.txt
    - 宽松：predictions_*fold*.txt
    - 旧：predictions_fold1.txt
    - 兜底：*.txt（最后手段）
    提取每行 Prediction/Label/QError，依据 Prediction vs Label 判 under/over。
    """
    results = []

    patterns = [
        "predictions_q*_fold*.txt",   # 新命名
        "predictions_*fold*.txt",     # 更宽松
        "predictions_fold*.txt",      # 旧命名
        "*.txt",                      # 兜底（放最后，避免误读）
    ]
    files = _glob_unique(patterns, dirpath)

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8-sig", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)

                def find_idx(hdr, colname):
                    if hdr is None:
                        return None
                    for i, name in enumerate(hdr):
                        if str(name).strip().lower() == colname.lower():
                            return i
                    return None

                # 如果第一行不是合法表头（或不是 CSV），回退到手动解析
                if header is None or len(header) < 3 or "QError" not in ",".join([str(h) for h in header]):
                    f.seek(0)
                    lines = f.read().strip().splitlines()
                    if not lines:
                        continue
                    header = [h.strip() for h in lines[0].split(",")]
                    i_pred = find_idx(header, "Prediction")
                    i_label = find_idx(header, "Label")
                    i_qerr = find_idx(header, "QError")
                    if None in (i_pred, i_label, i_qerr):
                        continue
                    for line in lines[1:]:
                        if line.strip().startswith("#"):
                            continue  # 跳过末尾注释统计行
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) <= max(i_pred, i_label, i_qerr):
                            continue
                        try:
                            pred = float(parts[i_pred]); lab = float(parts[i_label]); qerr = float(parts[i_qerr])
                        except:
                            continue
                        if qerr <= 0:
                            continue
                        sign = 1 if pred > lab else -1 if pred < lab else 0
                        if sign != 0:
                            results.append((qerr, sign))
                else:
                    # 正常 CSV 路径
                    header = [h.strip() for h in header]
                    i_pred = find_idx(header, "Prediction")
                    i_label = find_idx(header, "Label")
                    i_qerr = find_idx(header, "QError")
                    if None in (i_pred, i_label, i_qerr):
                        continue
                    for row in reader:
                        if not row or len(row) <= max(i_pred, i_label, i_qerr):
                            continue
                        if isinstance(row[0], str) and row[0].strip().startswith("#"):
                            continue  # 跳过注释行
                        try:
                            pred = float(row[i_pred]); lab = float(row[i_label]); qerr = float(row[i_qerr])
                        except:
                            continue
                        if qerr <= 0:
                            continue
                        sign = 1 if pred > lab else -1 if pred < lab else 0
                        if sign != 0:
                            results.append((qerr, sign))
        except:
            pass
    return results

def signed_log10(qerr, sign):
    # 仍然用 log10 计算，只是坐标轴写成 log
    val = math.log10(qerr)
    return val if sign > 0 else -val

# -----------------------------
# 数据收集 & 绘图
# -----------------------------
if __name__ == "__main__":
    data = defaultdict(list)

    for exp in EXPERIMENTS:
        for model in MODELS:
            dirpath = os.path.join(BASE_DIR, exp, model)
            if not os.path.exists(dirpath):
                continue

            if model == "FlowSC":
                files = sorted(glob.glob(os.path.join(dirpath, "*")))
                all_pairs = []
                for fp in files:
                    if os.path.isdir(fp):
                        continue
                    all_pairs.extend(parse_flowsc_file(fp))
                data[(exp, model)].extend(signed_log10(q, s) for q, s in all_pairs if q > 0)

            elif model == "LGCE":
                pairs = parse_lgce_dir(dirpath)
                data[(exp, model)].extend(signed_log10(q, s) for q, s in pairs if q > 0)

    # 画图：分组箱线图（加大间距+变窄+适当左右偏移，避免重叠）
    plot_values = []
    positions = []

    group_gap = 2.6
    box_width = 0.35
    offset = 0.2

    x = 1.0
    for exp in EXPERIMENTS:
        v_flowsc = data.get((exp, "FlowSC"), [])
        v_lgce   = data.get((exp, "LGCE"), [])

        plot_values.append(v_flowsc); positions.append(x - offset)
        plot_values.append(v_lgce);   positions.append(x + offset)
        x += group_gap

    fig, ax = plt.subplots(figsize=(13, 6))

    bp = ax.boxplot(
        plot_values,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False
    )

    # 颜色
    COLOR_FLOW = "#8FB3D9"  # FlowSC
    COLOR_LGCE = "#90C4B8"  # LGCE

    for i, box in enumerate(bp['boxes']):
        face = COLOR_FLOW if i % 2 == 0 else COLOR_LGCE
        box.set_facecolor(face)
        box.set_edgecolor(face)
        box.set_alpha(0.9)
        box.set_linewidth(1.4)

    # whiskers/caps/median/fliers 统一配色
    for i, (w1, w2) in enumerate(zip(bp['whiskers'][0::2], bp['whiskers'][1::2])):
        color = COLOR_FLOW if i % 2 == 0 else COLOR_LGCE
        w1.set_color(color); w2.set_color(color)
        w1.set_linewidth(1.2); w2.set_linewidth(1.2)

    for i, (c1, c2) in enumerate(zip(bp['caps'][0::2], bp['caps'][1::2])):
        color = COLOR_FLOW if i % 2 == 0 else COLOR_LGCE
        c1.set_color(color); c2.set_color(color)
        c1.set_linewidth(1.2); c2.set_linewidth(1.2)

    for i, med in enumerate(bp['medians']):
        med.set_color("#000000")   
        med.set_linewidth(1.0)

    for i, fl in enumerate(bp['fliers']):
        color = COLOR_FLOW if i % 2 == 0 else COLOR_LGCE
        fl.set_marker('o')
        fl.set_markerfacecolor(color)
        fl.set_markeredgecolor(color)
        fl.set_alpha(0.6)

    # X 轴：组中心位置与标签（明确代表节点数）
    group_centers = []
    x = 1.0
    for _ in EXPERIMENTS:
        group_centers.append(x)
        x += group_gap

    ax.set_xticks(group_centers)
    ax.set_xticklabels(EXPERIMENTS)
    ax.set_xlabel("Experiment (#nodes )")

    # Y 轴刻度与标签（显示为 12 8 4 0 4 8 12）
    ticks = [-8,  -4 , 0 , 4 , 8]
    tick_labels = [str(abs(t)) for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("log(q-error)  (underestimate below 0, overestimate above 0)")

    # 图例：移到图外右侧并缩小
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

    ax.set_title(f"{dataset} (#nodes): Signed log(q-error) Boxplots (FlowSC vs LGCE)")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PNG}")
