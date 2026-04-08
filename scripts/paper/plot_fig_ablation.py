"""Figure: modality ablation summary.

Panel A: modality composition matrix (DNA / RNA / ESM)
Panel B: overall Pearson performance with delta vs. full model
Panel C: 14-tissue Pearson heatmap
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd

from eval_utils import TISSUE_NAMES

RDIR = Path("/home/hjd/RNAi/results/paper")
FDIR = RDIR / "figures"
FDIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RDIR / "ablation_20kb_summary.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DNA_COLOR = "#2563eb"
RNA_COLOR = "#16a34a"
ESM_COLOR = "#ea580c"
FULL_COLOR = "#dc2626"
OTHER_COLOR = "#4c78a8"
TEXT_DARK = "#1f2937"
TEXT_MUTED = "#6b7280"
GRID = "#e5e7eb"
MISSING_FILL = "#f8fafc"
MISSING_EDGE = "#cbd5e1"
PANEL_BG = "#f8fafc"

MODALITIES = [
    ("DNA", "dna", DNA_COLOR),
    ("RNA", "rna", RNA_COLOR),
    ("ESM", "esm", ESM_COLOR),
]
DISPLAY_ORDER = ["Full model", "RNA+ESM only", "w/o ESM", "w/o RNA", "DNA only"]


def rounded_cell(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, radius=0.09, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def short_tissue_label(tissue):
    prefix = "A." if tissue.startswith("Adult_") else "L."
    body = tissue.split("_", 1)[1]
    body = body.replace("MalpighianTubule", "MalTub")
    return f"{prefix}\n{body}"


def main():
    df = pd.read_csv(CSV_PATH)
    df["variant"] = pd.Categorical(df["variant"], categories=DISPLAY_ORDER, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)

    tissue_cols = [f"{tissue}_pearson" for tissue in TISSUE_NAMES]
    missing = [col for col in tissue_cols if col not in df.columns]
    if missing:
        missing_names = ", ".join(col.replace("_pearson", "") for col in missing)
        raise ValueError(
            "Missing tissue Pearson columns in ablation_20kb_summary.csv: "
            f"{missing_names}. Rerun scripts/paper/run_ablation_20kb.py first."
        )

    full_pearson = float(df.loc[df["variant"] == "Full model", "pearson"].iloc[0])
    variants = df["variant"].astype(str).tolist()
    n = len(df)
    y_positions = np.arange(n)

    fig = plt.figure(figsize=(15.2, 8.0), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.28],
        width_ratios=[1.05, 2.25],
        hspace=0.42,
        wspace=0.18,
    )
    ax_mat = fig.add_subplot(gs[0, 0])
    ax_perf = fig.add_subplot(gs[0, 1], sharey=ax_mat)
    ax_heat = fig.add_subplot(gs[1, :])

    for ax in (ax_mat, ax_perf):
        ax.set_ylim(-0.92, n - 0.45)
        ax.invert_yaxis()
        ax.set_facecolor("white")
        for row in y_positions:
            stripe = "#fcfcfd" if row % 2 == 0 else PANEL_BG
            ax.axhspan(row - 0.5, row + 0.5, color=stripe, zorder=0)
        ax.axhspan(-0.5, 0.5, color="#fef2f2", zorder=0.3)

    # Panel A
    ax_mat.set_title("A  Modality Inputs", loc="left", fontweight="bold", pad=12)
    cell_w, cell_h = 0.56, 0.56
    x_centers = [0.82, 1.72, 2.62]
    header_y = -0.72
    line_y = -0.59
    for j, (label, _, color) in enumerate(MODALITIES):
        ax_mat.text(
            x_centers[j], header_y, label,
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=color,
            clip_on=False,
        )
        ax_mat.plot(
            [x_centers[j] - 0.19, x_centers[j] + 0.19],
            [line_y, line_y],
            color=color,
            lw=2.5,
            solid_capstyle="round",
            clip_on=False,
        )

    for i, row in df.iterrows():
        y = float(y_positions[i])
        is_full = row["variant"] == "Full model"
        label_color = FULL_COLOR if is_full else TEXT_DARK
        label_weight = "bold" if is_full else "normal"
        ax_mat.text(
            -0.10, y, row["variant"],
            ha="right", va="center",
            fontsize=11.1, color=label_color, fontweight=label_weight,
        )
        for j, (_, key, color) in enumerate(MODALITIES):
            val = int(row[key])
            x0 = x_centers[j] - cell_w / 2
            y0 = y - cell_h / 2
            if val:
                rounded_cell(ax_mat, x0, y0, cell_w, cell_h, color, color, lw=1.1, radius=0.06)
            else:
                rounded_cell(ax_mat, x0, y0, cell_w, cell_h, MISSING_FILL, MISSING_EDGE, lw=1.1, radius=0.06)

    ax_mat.text(1.72, n - 0.05, "filled = modality included", ha="center", va="top", fontsize=8.6, color=TEXT_MUTED)
    ax_mat.set_xlim(-0.22, 3.02)
    ax_mat.axis("off")

    # Panel B
    ax_perf.set_title("B  Overall Pearson", loc="left", fontweight="bold", pad=12)
    min_x = max(0.18, float(df["pearson"].min()) - 0.04)
    max_x = min(0.82, float(df["pearson"].max()) + 0.09)
    ax_perf.set_xlim(min_x, max_x)
    ax_perf.set_yticks(y_positions)
    ax_perf.set_yticklabels([""] * n)
    ax_perf.spines["left"].set_visible(False)
    ax_perf.tick_params(axis="y", length=0)
    ax_perf.grid(axis="x", color=GRID, alpha=0.85, lw=0.9)
    ax_perf.axvline(full_pearson, color=FULL_COLOR, ls="--", lw=1.2, alpha=0.18, zorder=1)

    stem_start = min_x + 0.01
    for i, row in df.iterrows():
        y = float(y_positions[i])
        pearson = float(row["pearson"])
        is_full = row["variant"] == "Full model"
        color = FULL_COLOR if is_full else OTHER_COLOR
        lw = 2.6 if is_full else 2.0
        size = 135 if is_full else 92
        alpha = 0.92 if is_full else 0.58
        ax_perf.hlines(y, stem_start, pearson, color=color, lw=lw, alpha=alpha, zorder=2)
        ax_perf.scatter(pearson, y, s=size, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax_perf.text(
            pearson + 0.011, y - 0.08, f"{pearson:.3f}",
            ha="left", va="center",
            fontsize=10.8, color=color,
            fontweight="bold" if is_full else "normal",
        )
        if is_full:
            ax_perf.text(
                pearson + 0.011, y + 0.18, "Best",
                ha="left", va="center",
                fontsize=8.9, color=FULL_COLOR,
            )
        else:
            delta = pearson - full_pearson
            ax_perf.text(
                pearson + 0.011, y + 0.18, f"Delta {delta:+.3f}",
                ha="left", va="center",
                fontsize=8.9, color=TEXT_MUTED,
            )
    ax_perf.set_xlabel("Pearson correlation")

    # Panel C
    ax_heat.set_title("C  Tissue-Specific Pearson Across 14 Tissues", loc="left", fontweight="bold", pad=18)
    heat_values = df[tissue_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    heat_masked = np.ma.masked_invalid(heat_values)
    vmin = float(np.nanmin(heat_values))
    vmax = float(np.nanmax(heat_values))
    cmap = colors.LinearSegmentedColormap.from_list(
        "ablation_heat",
        ["#fff7ed", "#fed7aa", "#fde68a", "#86efac", "#16a34a"],
    )
    cmap.set_bad(MISSING_FILL)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax_heat.imshow(heat_masked, aspect="auto", cmap=cmap, norm=norm)

    ax_heat.set_xticks(np.arange(len(TISSUE_NAMES)))
    ax_heat.set_xticklabels([short_tissue_label(tissue) for tissue in TISSUE_NAMES])
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_yticklabels(variants)
    ax_heat.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=9, length=0)
    ax_heat.tick_params(axis="y", length=0, pad=8)
    plt.setp(ax_heat.get_xticklabels(), rotation=0, ha="center", va="bottom", linespacing=0.96)
    for label, variant in zip(ax_heat.get_yticklabels(), variants):
        if variant == "Full model":
            label.set_color(FULL_COLOR)
            label.set_fontweight("bold")
        else:
            label.set_color(TEXT_DARK)

    ax_heat.set_xticks(np.arange(-0.5, len(TISSUE_NAMES), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.15)
    ax_heat.tick_params(which="minor", left=False, bottom=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    ax_heat.axvline(8.5, color="#94a3b8", lw=1.2)
    ax_heat.add_patch(Rectangle((-0.5, -0.5), len(TISSUE_NAMES), 1.0, fill=False, ec=FULL_COLOR, lw=1.5))

    for i in range(n):
        for j in range(len(TISSUE_NAMES)):
            val = heat_values[i, j]
            if np.isnan(val):
                ax_heat.text(j, i, "-", ha="center", va="center", fontsize=8.3, color=TEXT_MUTED)
                continue
            text_color = "white" if norm(val) > 0.60 else TEXT_DARK
            ax_heat.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=8.4,
                color=text_color,
                fontweight="bold" if i == 0 else "normal",
            )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.022, pad=0.014)
    cbar.outline.set_visible(False)
    cbar.set_label("Pearson", fontsize=10.0)
    cbar.ax.tick_params(labelsize=9.0)

    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.07, right=0.965)
    fig.suptitle("Modality Ablation of InsectExpress (20 kb)", fontsize=15.8, fontweight="bold", y=0.972)
    fig.text(
        0.5, 0.035,
        "Panel C reports per-tissue Pearson across all 14 tissues; rows follow the same ablation order as panels A and B.",
        ha="center", va="center", fontsize=8.9, color=TEXT_MUTED,
    )

    for ext in ("png", "pdf"):
        fig.savefig(FDIR / f"fig_ablation.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {FDIR / 'fig_ablation.png'}")


if __name__ == "__main__":
    main()
