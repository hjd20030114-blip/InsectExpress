#!/usr/bin/env python3
"""Figure 6: ISM motif interpretability summary across all available tissues."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


INTERP_DIR = Path("/home/hjd/RNAi/results/interpretability_v2")
OUT_DIR = Path("/home/hjd/RNAi/results/paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARED_OUT = OUT_DIR / "fig6_ism_shared_motifs.csv"
SIGNATURE_OUT = OUT_DIR / "fig6_ism_signature_motifs.csv"
EXPORT_DPI = 400

ALL_TISSUES = [
    "Adult_Brain",
    "Adult_Head",
    "Adult_Midgut",
    "Adult_Hindgut",
    "Adult_FatBody",
    "Adult_MalpighianTubule",
    "Adult_Carcass",
    "Adult_Ovary",
    "Adult_Testis",
    "Larval_Hindgut",
    "Larval_FatBody",
    "Larval_MalpighianTubule",
    "Larval_Midgut",
    "Larval_Carcass",
]
TISSUE_COLORS = {
    "Adult_Brain": "#1f5f8b",
    "Adult_Head": "#2573b5",
    "Adult_Midgut": "#e67e22",
    "Adult_Hindgut": "#f39c12",
    "Adult_FatBody": "#6b8f23",
    "Adult_MalpighianTubule": "#16a085",
    "Adult_Carcass": "#4d908e",
    "Adult_Ovary": "#d1495b",
    "Adult_Testis": "#9c27b0",
    "Larval_Hindgut": "#ffb703",
    "Larval_FatBody": "#90be6d",
    "Larval_MalpighianTubule": "#43aa8b",
    "Larval_Midgut": "#c73e64",
    "Larval_Carcass": "#2d7d5a",
}
REGION_ORDER = [
    "Distal upstream",
    "Proximal upstream",
    "Core promoter",
    "Proximal downstream",
    "Distal downstream",
]
REGION_COLORS = {
    "Distal upstream": "#7c8aa5",
    "Proximal upstream": "#73b3d8",
    "Core promoter": "#f3c95d",
    "Proximal downstream": "#e88d67",
    "Distal downstream": "#ad6ea8",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.8,
    "axes.titlesize": 13.0,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def blend(color: str, target: str = "#ffffff", amount: float = 0.7) -> tuple[float, float, float]:
    src = np.array(mcolors.to_rgb(color))
    dst = np.array(mcolors.to_rgb(target))
    return tuple(src * (1 - amount) + dst * amount)


def tissue_label(tissue: str) -> str:
    stage, organ = tissue.split("_", 1)
    organ = organ.replace("MalpighianTubule", "Malpighian tubule")
    organ = organ.replace("FatBody", "fat body")
    return f"{stage} {organ.lower()}"


def short_label(tissue: str) -> str:
    stage, organ = tissue.split("_", 1)
    stage_short = "A." if stage == "Adult" else "L."
    organ_map = {
        "Brain": "Brain",
        "Head": "Head",
        "Midgut": "Midgut",
        "Hindgut": "Hindgut",
        "FatBody": "FatBody",
        "MalpighianTubule": "MalTub",
        "Carcass": "Carcass",
        "Ovary": "Ovary",
        "Testis": "Testis",
    }
    return f"{stage_short}\n{organ_map.get(organ, organ)}"


def compact_label(tissue: str) -> str:
    stage, organ = tissue.split("_", 1)
    stage_short = "A." if stage == "Adult" else "L."
    organ_map = {
        "Brain": "Brain",
        "Head": "Head",
        "Midgut": "Midgut",
        "Hindgut": "Hindgut",
        "FatBody": "FatBody",
        "MalpighianTubule": "MalTub",
        "Carcass": "Carcass",
        "Ovary": "Ovary",
        "Testis": "Testis",
    }
    return f"{stage_short}{organ_map.get(organ, organ)}"


def classify_region(position: float) -> str:
    if position < -2000:
        return "Distal upstream"
    if position < -200:
        return "Proximal upstream"
    if position <= 200:
        return "Core promoter"
    if position <= 2000:
        return "Proximal downstream"
    return "Distal downstream"


def format_bp(position: float) -> str:
    if abs(position) >= 1000:
        return f"{position / 1000:+.1f} kb"
    return f"{int(round(position)):+d} bp"


def gaussian_density(positions: np.ndarray, weights: np.ndarray, grid: np.ndarray, sigma: float = 140.0) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros_like(grid)
    diffs = grid[None, :] - positions[:, None]
    kernels = np.exp(-0.5 * (diffs / sigma) ** 2)
    return np.sum(kernels * weights[:, None], axis=0)


def available_tissues() -> list[str]:
    tissues = [t for t in ALL_TISSUES if (INTERP_DIR / t / "top_motifs.json").exists()]
    if not tissues:
        raise FileNotFoundError(f"No top_motifs.json found under {INTERP_DIR}")
    return tissues


def load_motif_table(tissues: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cards = []
    for tissue in tissues:
        path = INTERP_DIR / tissue / "top_motifs.json"
        obj = json.loads(path.read_text())
        motifs = pd.DataFrame(obj["motifs"])
        motifs["tissue"] = tissue
        motifs["n_genes"] = int(obj["n_genes"])
        motifs["region"] = motifs["position"].apply(classify_region)
        motifs = motifs.drop_duplicates(subset=["sequence", "position"])
        rows.append(motifs)

        strongest = motifs.sort_values("score", ascending=False).iloc[0]
        cards.append({
            "tissue": tissue,
            "n_motifs": int(len(motifs)),
            "n_genes": int(obj["n_genes"]),
            "top_sequence": strongest["sequence"],
            "top_score": float(strongest["score"]),
            "top_position": int(strongest["position"]),
            "dominant_region": classify_region(float(strongest["position"])),
            "logo_path": INTERP_DIR / tissue / f"motif_logo_{tissue}.png",
        })

    motif_df = pd.concat(rows, ignore_index=True)
    card_df = pd.DataFrame(cards)
    return motif_df, card_df


def build_signature_table(motif_df: pd.DataFrame, tissues: list[str]) -> pd.DataFrame:
    best_hits = (
        motif_df.sort_values("score", ascending=False)
        .drop_duplicates(subset=["tissue", "sequence"], keep="first")
        .loc[:, ["tissue", "sequence", "score", "position", "region"]]
        .rename(columns={"score": "best_score", "position": "best_position", "region": "best_region"})
    )

    score_matrix = (
        best_hits.pivot(index="sequence", columns="tissue", values="best_score")
        .reindex(columns=tissues)
        .fillna(0.0)
    )
    prevalence = (score_matrix > 0).sum(axis=1)
    n_tissues = len(tissues)

    candidate_tables: dict[str, pd.DataFrame] = {}
    best_metric: dict[str, float] = {}
    for tissue in tissues:
        own = score_matrix[tissue]
        max_other = score_matrix.drop(columns=[tissue]).max(axis=1)
        metric = own * (n_tissues - prevalence + 1) / n_tissues
        cand = pd.DataFrame(
            {
                "sequence": score_matrix.index,
                "own_score": own.to_numpy(),
                "max_other": max_other.to_numpy(),
                "prevalence": prevalence.to_numpy(),
                "metric": metric.to_numpy(),
            }
        )
        # Prefer motifs that peak in this tissue and are not broadly shared.
        cand = cand[(cand["own_score"] >= 0.8) & (cand["own_score"] >= cand["max_other"])]
        if cand.empty:
            cand = pd.DataFrame(
                {
                    "sequence": score_matrix.index,
                    "own_score": own.to_numpy(),
                    "max_other": max_other.to_numpy(),
                    "prevalence": prevalence.to_numpy(),
                    "metric": metric.to_numpy(),
                }
            )
            cand = cand[cand["own_score"] >= 0.8]
        cand = cand.sort_values(["metric", "own_score"], ascending=[False, False]).reset_index(drop=True)
        candidate_tables[tissue] = cand
        best_metric[tissue] = float(cand["metric"].iloc[0]) if not cand.empty else 0.0

    assigned: dict[str, str] = {}
    used_sequences: set[str] = set()
    tissue_order = sorted(tissues, key=lambda t: best_metric[t], reverse=True)
    for tissue in tissue_order:
        cand = candidate_tables[tissue]
        chosen = None
        for _, row in cand.iterrows():
            if row["sequence"] not in used_sequences:
                chosen = row["sequence"]
                break
        if chosen is None:
            chosen = str(cand["sequence"].iloc[0])
        assigned[tissue] = chosen
        used_sequences.add(chosen)

    rows = []
    for tissue in tissues:
        sequence = assigned[tissue]
        hit = best_hits[(best_hits["tissue"] == tissue) & (best_hits["sequence"] == sequence)].iloc[0]
        footprint = score_matrix.loc[sequence, tissues]
        other_scores = footprint.drop(labels=tissue)
        rows.append(
            {
                "tissue": tissue,
                "sequence": sequence,
                "own_score": float(hit["best_score"]),
                "position": float(hit["best_position"]),
                "region": str(hit["best_region"]),
                "prevalence": int((footprint > 0).sum()),
                "runner_up_score": float(other_scores.max()),
                "runner_up_tissue": str(other_scores.idxmax()) if float(other_scores.max()) > 0 else "",
                "footprint": footprint.to_dict(),
            }
        )

    return pd.DataFrame(rows)


def build_shared_motif_table(motif_df: pd.DataFrame, tissues: list[str]) -> pd.DataFrame:
    shared_rows = []
    grouped = motif_df.sort_values("score", ascending=False).groupby("sequence", sort=False)

    for sequence, group in grouped:
        best_per_tissue = group.drop_duplicates(subset=["tissue"], keep="first")
        n_tissues = best_per_tissue["tissue"].nunique()
        if n_tissues <= 1:
            continue

        row = {
            "sequence": sequence,
            "n_tissues": int(n_tissues),
            "sum_score": float(best_per_tissue["score"].sum()),
            "mean_score": float(best_per_tissue["score"].mean()),
            "mean_position": float(best_per_tissue["position"].mean()),
        }
        for tissue in tissues:
            sub = best_per_tissue.loc[best_per_tissue["tissue"] == tissue]
            if sub.empty:
                row[f"{tissue}_score"] = np.nan
                row[f"{tissue}_position"] = np.nan
            else:
                row[f"{tissue}_score"] = float(sub["score"].iloc[0])
                row[f"{tissue}_position"] = float(sub["position"].iloc[0])
        shared_rows.append(row)

    shared_df = pd.DataFrame(shared_rows)
    if shared_df.empty:
        raise ValueError("No shared motifs found across tissues.")

    shared_df = shared_df.sort_values(
        ["n_tissues", "sum_score", "mean_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    shared_df.to_csv(SHARED_OUT, index=False)
    return shared_df


def plot_panel_a(ax: plt.Axes, motif_df: pd.DataFrame, tissues: list[str]) -> None:
    ax.set_title("A  Position-Resolved ISM Hotspot Landscape", loc="left", fontweight="bold", pad=8)

    grid = np.linspace(-4000, 5000, 1600)
    densities = {}
    ymax = 0.0
    for tissue in tissues:
        sub = motif_df.loc[motif_df["tissue"] == tissue]
        density = gaussian_density(sub["position"].to_numpy(), sub["score"].to_numpy(), grid)
        densities[tissue] = density
        ymax = max(ymax, float(density.max()))

    y_step = 0.72 if len(tissues) > 10 else 0.95
    y_levels = np.arange(len(tissues))[::-1] * y_step

    ax.axvspan(-200, 200, color="#fff4cc", alpha=0.92, zorder=0)
    ax.axvline(0, color="#b91c1c", lw=1.2, ls=(0, (3, 2)), alpha=0.9, zorder=1)
    ax.text(0, y_levels[0] + 0.86, "TSS", ha="center", va="bottom", fontsize=9.2, color="#991b1b")
    ax.text(-1500, y_levels[0] + 0.86, "upstream", ha="center", va="bottom", fontsize=8.8, color="#64748b")
    ax.text(2500, y_levels[0] + 0.86, "downstream", ha="center", va="bottom", fontsize=8.8, color="#64748b")

    score_min = float(motif_df["score"].min())
    score_max = float(motif_df["score"].max())

    for idx, tissue in enumerate(tissues):
        y0 = y_levels[idx]
        color = TISSUE_COLORS[tissue]
        light = blend(color, amount=0.8)
        density = densities[tissue] / ymax
        ridge = density * 0.65

        ax.hlines(y0, grid[0], grid[-1], color=blend(color, amount=0.88), lw=0.9, zorder=1)
        ax.fill_between(grid, y0, y0 + ridge, color=light, alpha=0.98, zorder=2)
        ax.plot(grid, y0 + ridge, color=color, lw=1.9, zorder=3)

        top = motif_df.loc[motif_df["tissue"] == tissue].sort_values("score", ascending=False).head(4)
        marker_y = np.interp(top["position"].to_numpy(), grid, y0 + ridge)
        sizes = np.interp(top["score"].to_numpy(), [score_min, score_max], [24, 120])
        ax.scatter(
            top["position"],
            marker_y,
            s=sizes,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

        ax.text(
            -4300,
            y0 + 0.26,
            tissue_label(tissue),
            ha="left",
            va="center",
            fontsize=8.9 if len(tissues) > 10 else 10.0,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(-4350, 5100)
    ax.set_ylim(-0.15, y_levels[0] + 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Position relative to TSS (bp)")
    ax.set_ylabel("Weighted motif density")
    ax.grid(axis="x", color="#e2e8f0", lw=0.8, alpha=0.7)
    ax.spines["left"].set_visible(False)


def plot_panel_b(ax: plt.Axes, motif_df: pd.DataFrame, tissues: list[str]) -> None:
    ax.set_title("B  Score-Weighted Genomic Zone Composition", loc="left", fontweight="bold", pad=16)

    grouped = motif_df.groupby(["tissue", "region"])["score"].sum().unstack(fill_value=0.0)
    grouped = grouped.reindex(index=tissues, columns=REGION_ORDER, fill_value=0.0)
    grouped = grouped.div(grouped.sum(axis=1), axis=0)
    weighted_center = (
        motif_df.groupby("tissue")[["position", "score"]]
        .apply(lambda x: np.average(x["position"], weights=x["score"]))
        .reindex(tissues)
    )

    y = np.arange(len(tissues))[::-1]
    left = np.zeros(len(tissues))
    for region in REGION_ORDER:
        values = grouped[region].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=REGION_COLORS[region],
            edgecolor="white",
            linewidth=0.8,
        )
        left += values

    position_norm = mcolors.TwoSlopeNorm(vmin=-2500, vcenter=0, vmax=3000)
    position_cmap = mcolors.LinearSegmentedColormap.from_list(
        "position_bias",
        ["#225ea8", "#f8fafc", "#c2410c"],
    )
    ax.scatter(
        np.full(len(tissues), 1.06),
        y,
        s=76,
        c=weighted_center.to_numpy(),
        cmap=position_cmap,
        norm=position_norm,
        edgecolor="#475569",
        linewidth=0.75,
        clip_on=False,
        zorder=4,
    )
    for yi, value in zip(y, weighted_center.to_numpy()):
        ax.text(1.095, yi, format_bp(value), ha="left", va="center", fontsize=7.9, color="#334155")

    ax.text(1.06, y.max() + 0.86, "weighted\ncentroid", ha="center", va="bottom", fontsize=8.0, color="#475569")
    ax.set_xlim(0, 1.27)
    ax.set_yticks(y)
    ax.set_yticklabels([tissue_label(t) for t in tissues])
    ax.set_xlabel("Fraction of motif score")
    ax.grid(axis="x", color="#e2e8f0", lw=0.8, alpha=0.7)
    ax.tick_params(axis="y", length=0)

    handles = [Line2D([0], [0], color=REGION_COLORS[r], lw=6, label=r) for r in REGION_ORDER]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.12),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        handlelength=1.5,
        columnspacing=0.8,
    )


def plot_panel_c(ax: plt.Axes, shared_df: pd.DataFrame, tissues: list[str]) -> None:
    n_rows = min(12, len(shared_df))
    shared_df = shared_df.head(n_rows).copy()
    ax.set_title("C  Cross-Tissue Shared Motif Atlas", loc="left", fontweight="bold", pad=20)

    y = np.arange(n_rows)[::-1]
    x = np.arange(len(tissues))
    pos_cmap = mcolors.LinearSegmentedColormap.from_list(
        "shared_position",
        ["#1d4ed8", "#f8fafc", "#dc2626"],
    )
    pos_norm = mcolors.TwoSlopeNorm(vmin=-2500, vcenter=0, vmax=3000)

    score_values = []
    for tissue in tissues:
        score_values.extend(shared_df[f"{tissue}_score"].dropna().tolist())
    score_values = np.asarray(score_values, dtype=float)
    min_score = float(score_values.min())
    max_score = float(score_values.max())

    for yi in y:
        ax.hlines(yi, -0.45, len(tissues) - 0.55, color="#e2e8f0", lw=0.75, zorder=0)

    for row_idx, (_, row) in enumerate(shared_df.iterrows()):
        yi = y[row_idx]
        ax.text(
            -0.6,
            yi,
            row["sequence"],
            ha="right",
            va="center",
            fontsize=8.9,
            color="#1f2937",
            fontfamily="DejaVu Sans Mono",
        )
        ax.text(
            len(tissues) - 0.42,
            yi,
            f"{int(row['n_tissues'])} tissues",
            ha="left",
            va="center",
            fontsize=7.8,
            color="#64748b",
        )
        for xi, tissue in enumerate(tissues):
            score = row[f"{tissue}_score"]
            position = row[f"{tissue}_position"]
            if pd.isna(score):
                ax.scatter(xi, yi, s=16, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.6, zorder=1)
                continue
            size = np.interp(score, [min_score, max_score], [34, 180])
            ax.scatter(
                xi,
                yi,
                s=size,
                c=[position],
                cmap=pos_cmap,
                norm=pos_norm,
                edgecolor=TISSUE_COLORS[tissue],
                linewidth=1.0,
                alpha=0.97,
                zorder=3,
            )

    ax.set_xlim(-0.78, len(tissues) - 0.2)
    ax.set_ylim(-1.2, n_rows - 0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(t) for t in tissues])
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    size_vals = np.linspace(min_score, max_score, 3)
    size_handles = [
        plt.scatter([], [], s=np.interp(v, [min_score, max_score], [34, 180]), color="#94a3b8", edgecolor="none")
        for v in size_vals
    ]
    legend1 = ax.legend(
        size_handles,
        [f"{v:.1f}" for v in size_vals],
        title="Score",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.28),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        title_fontsize=8.3,
        columnspacing=1.1,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)

    cax = ax.inset_axes([0.38, 1.22, 0.34, 0.055])
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=pos_norm, cmap=pos_cmap),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label("Relative position (blue = upstream, red = downstream)", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.6)


def draw_signature_card(ax: plt.Axes, row: pd.Series, tissues: list[str]) -> None:
    tissue = row["tissue"]
    color = TISSUE_COLORS[tissue]
    footprint: dict[str, float] = row["footprint"]
    adult_cut = sum(t.startswith("Adult_") for t in tissues)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bg = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.92,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor=blend(color, amount=0.87),
        edgecolor=blend(color, amount=0.25),
        linewidth=1.2,
    )
    band = FancyBboxPatch(
        (0.03, 0.79),
        0.94,
        0.15,
        boxstyle="round,pad=0.01,rounding_size=0.028",
        facecolor=color,
        edgecolor=color,
        linewidth=0,
    )
    ax.add_patch(bg)
    ax.add_patch(band)

    ax.text(0.05, 0.865, tissue_label(tissue), ha="left", va="center", fontsize=8.3, color="white", fontweight="bold")
    ax.text(0.05, 0.70, "Signature exact 12-mer", ha="left", va="center", fontsize=7.2, color="#475569")
    ax.text(0.05, 0.58, row["sequence"], ha="left", va="center", fontsize=8.9, color="#0f172a", fontfamily="DejaVu Sans Mono", fontweight="bold")
    ax.text(
        0.05,
        0.47,
        f"{format_bp(row['position'])} | {row['region']}",
        ha="left",
        va="center",
        fontsize=7.1,
        color="#334155",
    )
    ax.text(
        0.05,
        0.38,
        f"ISM {row['own_score']:.2f} | seen in {int(row['prevalence'])}/{len(tissues)} tissues",
        ha="left",
        va="center",
        fontsize=7.1,
        color="#64748b",
    )
    if row["runner_up_score"] > 0:
        runner_text = f"next: {compact_label(row['runner_up_tissue'])} {row['runner_up_score']:.2f}"
    else:
        runner_text = "next: none"
    ax.text(0.05, 0.30, runner_text, ha="left", va="center", fontsize=6.9, color="#64748b")

    ax.text(0.05, 0.215, "14-tissue footprint", ha="left", va="center", fontsize=7.0, color="#475569", fontweight="bold")
    ax.text(0.90, 0.215, "A | L", ha="right", va="center", fontsize=6.7, color="#94a3b8")

    x0, y0, total_w, h = 0.05, 0.075, 0.90, 0.10
    gap = 0.006
    cell_w = (total_w - gap * (len(tissues) - 1)) / len(tissues)
    vmax = max(float(v) for v in footprint.values()) if footprint else 1.0
    for idx, footprint_tissue in enumerate(tissues):
        xx = x0 + idx * (cell_w + gap)
        value = float(footprint.get(footprint_tissue, 0.0))
        if value <= 0:
            face = "#f8fafc"
            edge = "#d7dee8"
            lw = 0.55
        else:
            alpha = 0.20 + 0.80 * (value / max(vmax, 1e-6))
            face = mcolors.to_rgba(TISSUE_COLORS[footprint_tissue], alpha=alpha)
            edge = TISSUE_COLORS[footprint_tissue]
            lw = 0.75
        if footprint_tissue == tissue:
            edge = "#0f172a"
            lw = 1.35
        patch = FancyBboxPatch(
            (xx, y0),
            cell_w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
        )
        ax.add_patch(patch)

    divider_x = x0 + adult_cut * (cell_w + gap) - gap / 2
    ax.plot([divider_x, divider_x], [y0 - 0.012, y0 + h + 0.012], color="#cbd5e1", lw=0.9, ls=(0, (2, 2)))


def main() -> None:
    tissues = available_tissues()
    motif_df, card_df = load_motif_table(tissues)
    shared_df = build_shared_motif_table(motif_df, tissues)
    signature_df = build_signature_table(motif_df, tissues)
    signature_df.drop(columns=["footprint"]).to_csv(SIGNATURE_OUT, index=False)

    n_tissues = len(tissues)
    card_cols = min(7, n_tissues)
    card_rows = int(math.ceil(n_tissues / card_cols))

    fig = plt.figure(figsize=(22.0, 12.0 + 0.62 * n_tissues), facecolor="white")
    gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.0 + 0.042 * n_tissues, 1.00, 0.95 + 0.55 * card_rows],
        hspace=0.44,
    )

    top = gs[0].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.24)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(gs[1])
    cards_gs = gs[2].subgridspec(card_rows, card_cols, wspace=0.08, hspace=0.12)

    plot_panel_a(ax_a, motif_df, tissues)
    plot_panel_b(ax_b, motif_df, tissues)
    plot_panel_c(ax_c, shared_df, tissues)

    for idx, tissue in enumerate(tissues):
        row = signature_df.loc[signature_df["tissue"] == tissue].iloc[0]
        card_ax = fig.add_subplot(cards_gs[idx // card_cols, idx % card_cols])
        draw_signature_card(card_ax, row, tissues)

    total_slots = card_rows * card_cols
    for idx in range(len(tissues), total_slots):
        empty_ax = fig.add_subplot(cards_gs[idx // card_cols, idx % card_cols])
        empty_ax.axis("off")

    panel_title_y = 0.986
    fig.text(
        0.012,
        panel_title_y,
        f"ISM interpretability summary across {n_tissues} tissues",
        ha="left",
        va="top",
        fontsize=15.0,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.012,
        panel_title_y - 0.018,
        "Panel A shows score-weighted hotspot landscapes, Panel B summarizes regional allocation, "
        "Panel C highlights recurrent exact 12-mers, and Panel D selects one tissue-enriched exact 12-mer per tissue.",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#475569",
    )
    card_bottoms, card_tops, card_lefts, card_rights = cards_gs.get_grid_positions(fig)
    card_top = card_tops[0]
    card_center = 0.5 * (card_lefts[0] + card_rights[-1])
    fig.text(
        card_center,
        card_top + 0.03,
        "D  Tissue-Enriched Exact-Motif Signatures",
        ha="center",
        va="bottom",
        fontsize=13.0,
        fontweight="bold",
        color="#0f172a",
    )

    for fmt in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig6_ism_interpretability.{fmt}", dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Using tissues: {', '.join(tissues)}")
    print(f"Saved {OUT_DIR / 'fig6_ism_interpretability.png'}")
    print(f"Saved {OUT_DIR / 'fig6_ism_interpretability.pdf'}")
    print(f"Saved {SHARED_OUT}")
    print(f"Saved {SIGNATURE_OUT}")


if __name__ == "__main__":
    main()
