# plot_ablation_signed_logq_1x1_wordnet.py
# -*- coding: utf-8 -*-
"""
只绘制 WordNet 的消融箱型图为 1×1 矢量图，便于直接插入 LNCS 模板。
目录结构约定：
  ablation/wordnet/<variant>/<qnum>/predictions_*.txt

每个 predictions_* 文件需包含列：Prediction, Label, QError
signed log q-error = sign * log10(QError)，sign=+1(估大) / -1(估小)
箱线图参数：whis=(0,100)（须到 min/max），showfliers=False（不单独画离群点）
"""

import os
import re
import math
import glob
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ——— 矢量输出（文字保持为文本） ———
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42
plt.rcParams['svg.fonttype'] = 'none'
# 小字号（与前文风格一致）
plt.rcParams.update({
    "font.size": 7.0,
    "axes.labelsize": 6.6,
    "xtick.labelsize": 5.4,
    "ytick.labelsize": 5.4,
    "legend.fontsize": 7.0,
})

# =========================
# 配置
# =========================
BASE_DIR = "ablation"          # 根目录：ablation/<dataset>/
DATASET_NAME = "WordNet"       # 展示名
DATASET_DIR  = "wordnet"       # 目录名

# 若各 q 列表不同，会自动扫描数字子目录作为实验 q（如：4/8/12/16/20）
VARIANTS = [                   # (变体目录名, 图例名)
    ("base",        "BASE"),
    ("no_decoder",  "NO_DECODER"),
    ("no_encoder",  "NO_ENCODER"),
    ("no_gin",      "NO_GIN"),
]

# 颜色（与先前风格一致）
COLOR_BASE       = "#4C78A8"  # 蓝
COLOR_NO_DECODER = "#F2A97E"  # 橙
COLOR_NO_ENCODER = "#59A14F"  # 绿
COLOR_NO_GIN     = "#A17DB8"  # 紫
COLOR_MAP = {
    "base": COLOR_BASE,
    "no_decoder": COLOR_NO_DECODER,
    "no_encoder": COLOR_NO_ENCODER,
    "no_gin": COLOR_NO_GIN,
}

# ——— 布局参数（组间距、组内左右偏移、箱宽）———
GROUP_GAP     = 2.8
BOX_WIDTH     = 0.26
OFFSETS_4     = [-0.45, -0.15, 0.15, 0.45]  # 4 个变体在一组内的相对位置

# 图尺寸（适合 \textwidth 使用；可按需要微调）
FIG_W_IN, FIG_H_IN = 7.2, 2.9
OUT_PDF = "ablation_wordnet_1x1_signed_logq_boxplots.pdf"
OUT_SVG = "ablation_wordnet_1x1_signed_logq_boxplots.svg"

# =========================
# 解析工具
# =========================
PRED_PATTERNS = [
    "predictions_q*_fold*.txt",   # 新命名
    "predictions_*fold*.txt",     # 更宽松
    "predictions_fold*.txt",      # 旧命名
    "*.txt",                      # 兜底
]

def _glob_unique(patterns, dirpath):
    seen, files = set(), []
    for pat in patterns:
        for fp in sorted(glob.glob(os.path.join(dirpath, pat))):
            if not os.path.isdir(fp) and fp not in seen:
                seen.add(fp); files.append(fp)
    return files

def parse_variant_dir(dirpath):
    """
    读取一个变体在某个 q 目录下的所有 predictions_* 文件，返回
    [(qerror, sign), ...]，其中 sign=+1(估大) / -1(估小)。
    """
    results = []
    files = _glob_unique(PRED_PATTERNS, dirpath)
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8-sig", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)

                def find_idx(hdr, name):
                    if hdr is None: return None
                    for i, h in enumerate(hdr):
                        if str(h).strip().lower() == name.lower():
                            return i
                    return None

                if header is None or len(header) < 3 or "QError" not in ",".join([str(h) for h in header]):
                    # 回退手动解析
                    f.seek(0)
                    lines = f.read().strip().splitlines()
                    if not lines: continue
                    header = [h.strip() for h in lines[0].split(",")]
                    i_pred = find_idx(header, "Prediction"); i_label = find_idx(header, "Label"); i_qerr = find_idx(header, "QError")
                    if None in (i_pred, i_label, i_qerr): continue
                    for line in lines[1:]:
                        if line.strip().startswith("#"): continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) <= max(i_pred, i_label, i_qerr): continue
                        try:
                            pred = float(parts[i_pred]); lab = float(parts[i_label]); qerr = float(parts[i_qerr])
                        except: continue
                        if qerr <= 0: continue
                        sign = 1 if pred > lab else -1 if pred < lab else 0
                        if sign != 0: results.append((qerr, sign))
                else:
                    header = [h.strip() for h in header]
                    i_pred = find_idx(header, "Prediction"); i_label = find_idx(header, "Label"); i_qerr = find_idx(header, "QError")
                    if None in (i_pred, i_label, i_qerr): continue
                    for row in reader:
                        if not row or len(row) <= max(i_pred, i_label, i_qerr): continue
                        if isinstance(row[0], str) and row[0].strip().startswith("#"): continue
                        try:
                            pred = float(row[i_pred]); lab = float(row[i_label]); qerr = float(row[i_qerr])
                        except: continue
                        if qerr <= 0: continue
                        sign = 1 if pred > lab else -1 if pred < lab else 0
                        if sign != 0: results.append((qerr, sign))
        except:
            pass
    return results

def signed_log10(qerr, sign):
    val = math.log10(qerr)
    return val if sign > 0 else -val

def scan_numeric_subdirs(path):
    """扫描数字子目录（作为 q 值），按数值排序返回 ['4','8',...]"""
    if not os.path.isdir(path): return []
    exps = [n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n)) and n.isdigit()]
    exps.sort(key=lambda s: int(s))
    return exps

# =========================
# 绘制 WordNet 的消融箱型图
# =========================
def plot_wordnet(ax):
    ds_root = os.path.join(BASE_DIR, DATASET_DIR)

    # 全量 q（union）
    all_q = set()
    for vdir, _ in VARIANTS:
        vroot = os.path.join(ds_root, vdir)
        all_q.update(scan_numeric_subdirs(vroot))
    exps = sorted(all_q, key=lambda s: int(s))

    # 收集数据：data[(q, variant_dir)] = [signed log qerr, ...]
    data = defaultdict(list)
    for vdir, _ in VARIANTS:
        for q in exps:
            dpath = os.path.join(ds_root, vdir, q)
            if not os.path.exists(dpath): continue
            pairs = parse_variant_dir(dpath)
            data[(q, vdir)].extend(signed_log10(qe, s) for qe, s in pairs if qe > 0)

    # 组装：每个 q 一组，组内最多 4 箱；同时记录变体索引，保证配色正确
    plot_values, positions, box_variant_idx = [], [], []
    x = 1.0
    for q in exps:
        for j, (vdir, _) in enumerate(VARIANTS):
            vals = data.get((q, vdir), [])
            if not vals:
                continue
            plot_values.append(vals)
            positions.append(x + OFFSETS_4[j])
            box_variant_idx.append(j)
        x += GROUP_GAP

    # 坐标网格在底层
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=(0, (2, 3)), color="#e6e6e6", linewidth=0.35)

    if plot_values:
        bp = ax.boxplot(
            plot_values,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            whis=(0, 100),
            showfliers=False,
            zorder=2
        )
        # 上色
        for i, box in enumerate(bp["boxes"]):
            vdir = VARIANTS[box_variant_idx[i]][0]
            color = COLOR_MAP.get(vdir, "#888888")
            box.set_facecolor(color); box.set_edgecolor(color)
            box.set_alpha(1.0); box.set_linewidth(0.9); box.set_zorder(2)

        # whiskers / caps：每 2 条对应一个箱
        for i, w in enumerate(bp["whiskers"]):
            vdir = VARIANTS[box_variant_idx[i // 2]][0]
            color = COLOR_MAP.get(vdir, "#888888")
            w.set_color(color); w.set_linewidth(0.8); w.set_zorder(2)
        for i, cap in enumerate(bp["caps"]):
            vdir = VARIANTS[box_variant_idx[i // 2]][0]
            color = COLOR_MAP.get(vdir, "#888888")
            cap.set_color(color); cap.set_linewidth(0.8); cap.set_zorder(2)

        # medians
        for med in bp["medians"]:
            med.set_color("#222222"); med.set_linewidth(0.9); med.set_zorder(3)

        # X 轴刻度：各组中心
        group_centers = []
        x = 1.0
        for _ in exps:
            group_centers.append(x); x += GROUP_GAP
        ax.set_xticks(group_centers)
        ax.set_xticklabels(exps)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # y 轴：固定范围与刻度（便于与其他图可比）
    ax.set_ylim(-6, 6)
    ax.set_yticks([-6,-4, -2, 0, 2, 4,6])

    # 轴标签与数据集名
    ax.set_xlabel("#nodes")
    ax.text(0.02, 0.035, DATASET_NAME,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8))

    # 角落文字区分 over/under
    ax.text(0.985, 0.965, "over",  transform=ax.transAxes, ha="right", va="top",    fontsize=7.0, color="#4d4d4d")
    ax.text(0.985, 0.035, "under", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color="#4d4d4d")

    # 轴脊线
    for s in ax.spines.values():
        s.set_color("#b5b5b5"); s.set_linewidth(0.6)

# =========================
# 主流程：1×1（矢量）
# =========================
def main():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W_IN, FIG_H_IN), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.18, top=0.86, wspace=0.30, hspace=0.0)

    plot_wordnet(ax)

    # 顶部居中图例（全局）
    handles = [
        Patch(facecolor=COLOR_BASE,       edgecolor=COLOR_BASE,       label="BASE"),
        Patch(facecolor=COLOR_NO_DECODER, edgecolor=COLOR_NO_DECODER, label="NO_DECODER"),
        Patch(facecolor=COLOR_NO_ENCODER, edgecolor=COLOR_NO_ENCODER, label="NO_ENCODER"),
        Patch(facecolor=COLOR_NO_GIN,     edgecolor=COLOR_NO_GIN,     label="NO_GIN"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.985), borderaxespad=0.0,
               handlelength=1.6, columnspacing=1.2)

    # 全局 y 轴标题（左侧）
    fig.supylabel("signed log q-error", x=0.01, fontsize=6.6)

    # 矢量导出
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    print(f"[OK] Saved vector figures: {OUT_PDF} , {OUT_SVG}")

if __name__ == "__main__":
    main()
