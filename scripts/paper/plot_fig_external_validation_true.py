#!/usr/bin/env python3
"""Main-paper figure for true external validation on two unseen species."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch


BASE_DIR = Path("/home/hjd/RNAi/results/external_validation_true")
OUT_DIR = Path("/home/hjd/RNAi/results/paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = BASE_DIR / "true_external_validation_summary.json"
FIG_BASENAME = "fig_external_validation_true"
EXPORT_DPI = 400

SELECTED_SPECIES = [
    "spodoptera_frugiperda",
    "helicoverpa_armigera",
]

SPECIES_STYLE = {
    "spodoptera_frugiperda": {
        "label": "S. frugiperda",
        "color": "#d55e00",
        "fill": "#fbe3cf",
        "source_short": "Independent control RNA-seq aggregation",
        "provenance": "Untreated/control RNA-seq runs after harmonization.",
    },
    "helicoverpa_armigera": {
        "label": "H. armigera",
        "color": "#0f8a70",
        "fill": "#d8f1ea",
        "source_short": "Independent GEO-derived tissue atlas",
        "provenance": "Local GEO-derived atlas from GSE190405 and GSE86914.",
    },
}

TISSUE_ORDER = [
    "Adult_Brain",
    "Adult_Head",
    "Adult_Midgut",
    "Adult_FatBody",
    "Adult_MalpighianTubule",
    "Adult_Carcass",
    "Adult_Ovary",
    "Larval_FatBody",
    "Larval_Midgut",
]

TISSUE_LABELS = {
    "Adult_Brain": "A. Brain",
    "Adult_Head": "A. Head",
    "Adult_Midgut": "A. Midgut",
    "Adult_FatBody": "A. Fat body",
    "Adult_MalpighianTubule": "A. MalTub",
    "Adult_Carcass": "A. Carcass",
    "Adult_Ovary": "A. Ovary",
    "Larval_FatBody": "L. Fat body",
    "Larval_Midgut": "L. Midgut",
}

AUDIT_STAGES = [
    ("n_expression_genes", "Expression genes"),
    ("n_genes_in_gff", "Genes with GFF"),
    ("n_samples_built", "Evaluable genes"),
]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "axes.titlesize": 14.5,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.8,
        "legend.fontsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def blend(color: str, amount: float = 0.70, target: str = "#ffffff") -> tuple[float, float, float]:
    src = np.array(mcolors.to_rgb(color))
    dst = np.array(mcolors.to_rgb(target))
    return tuple(src * (1.0 - amount) + dst * amount)


def fmt_k(value: int) -> str:
    return f"{value / 1000:.1f}k"


def pretty_tissue_name(tissue: str) -> str:
    stage, organ = tissue.split("_", 1)
    organ = organ.replace("FatBody", "fat body")
    organ = organ.replace("MalpighianTubule", "Malpighian tubule")
    return f"{stage.lower()} {organ.lower()}"


def short_card_tissue_name(tissue: str) -> str:
    stage, organ = tissue.split("_", 1)
    organ_map = {
        "Brain": "brain",
        "Head": "head",
        "Midgut": "midgut",
        "FatBody": "fat body",
        "MalpighianTubule": "MalTub",
        "Carcass": "carcass",
        "Ovary": "ovary",
    }
    prefix = "A." if stage == "Adult" else "L."
    return f"{prefix} {organ_map.get(organ, organ)}"


def card_tissue_lines(tissues: list[str]) -> list[str]:
    grouped = {"Adult": [], "Larval": []}
    for tissue in tissues:
        stage, organ = tissue.split("_", 1)
        organ_map = {
            "Brain": "brain",
            "Head": "head",
            "Midgut": "midgut",
            "FatBody": "fat body",
            "MalpighianTubule": "MalTub",
            "Carcass": "carcass",
            "Ovary": "ovary",
        }
        grouped[stage].append(organ_map.get(organ, organ))

    lines: list[str] = []
    if grouped["Adult"]:
        adult = grouped["Adult"]
        if len(adult) <= 4:
            lines.append("Adult: " + ", ".join(adult))
        else:
            lines.append("Adult: " + ", ".join(adult[:4]))
            lines.append("       " + ", ".join(adult[4:]))
    if grouped["Larval"]:
        larval = grouped["Larval"]
        lines.append("Larva: " + ", ".join(larval))
    return lines


def load_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as handle:
        obj = json.load(handle)

    metrics_rows = []
    tissue_rows = []
    audit_rows = []

    for species in SELECTED_SPECIES:
        result = obj["results"][species]
        primary = result["primary_metrics"]
        audit = result["audit"]
        style = SPECIES_STYLE[species]

        metrics_rows.append(
            {
                "species": species,
                "label": style["label"],
                "display_name": result["display_name"],
                "color": style["color"],
                "fill": style["fill"],
                "source_short": style["source_short"],
                "provenance": style["provenance"],
                "source_note": result["source_note"],
                "expression_mode": result["expression_mode"],
                "overall_pearson": primary["overall_pearson"],
                "overall_spearman": primary["overall_spearman"],
                "gene_profile_pearson_mean": primary["gene_profile_pearson_mean"],
                "top1_tissue_match": primary["top1_tissue_match"],
                "primary_tissues": result["primary_tissues"],
                "n_primary_tissues": len(result["primary_tissues"]),
                "n_samples_built": audit["n_samples_built"],
            }
        )

        for tissue, vals in primary["per_tissue"].items():
            tissue_rows.append(
                {
                    "species": species,
                    "label": style["label"],
                    "tissue": tissue,
                    "pearson": vals["pearson"],
                    "spearman": vals["spearman"],
                    "n_genes": vals["n_genes"],
                }
            )

        for key, stage in AUDIT_STAGES:
            audit_rows.append(
                {
                    "species": species,
                    "label": style["label"],
                    "stage_key": key,
                    "stage": stage,
                    "count": audit[key],
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    tissue_df = pd.DataFrame(tissue_rows)
    audit_df = pd.DataFrame(audit_rows)
    return metrics_df, tissue_df, audit_df


def export_source_tables(metrics_df: pd.DataFrame, tissue_df: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    metrics_df.to_csv(OUT_DIR / f"{FIG_BASENAME}_metrics.csv", index=False)
    tissue_df.to_csv(OUT_DIR / f"{FIG_BASENAME}_per_tissue.csv", index=False)
    audit_df.to_csv(OUT_DIR / f"{FIG_BASENAME}_audit.csv", index=False)


def add_card(ax: plt.Axes, x: float, y: float, w: float, h: float, row: pd.Series) -> None:
    box_y = y + 0.05
    box_h = h - 0.10
    box_x = x + w - 0.27
    box_w = 0.22

    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=blend(row["color"], amount=0.35),
        facecolor=row["fill"],
    )
    ax.add_patch(rect)

    metrics_box = FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=blend(row["color"], amount=0.35),
        facecolor="white",
    )
    ax.add_patch(metrics_box)

    left_x = x + 0.03
    ax.text(left_x, y + h - 0.075, row["label"], fontsize=14.8, fontweight="bold", color=row["color"])
    ax.text(
        left_x,
        y + h - 0.145,
        row["source_short"],
        fontsize=10.4,
        color="#334155",
        fontweight="bold",
    )
    ax.text(
        left_x,
        y + h - 0.205,
        "Unseen species; independent source; absolute expression.",
        fontsize=8.6,
        color="#475569",
        va="top",
    )

    ax.text(left_x, y + h - 0.255, "Mapped tissues", fontsize=9.1, color="#475569", va="top", fontweight="bold")
    tissue_lines = card_tissue_lines(row["primary_tissues"])
    for idx, line in enumerate(tissue_lines):
        ax.text(
            left_x,
            y + h - 0.315 - idx * 0.050,
            line,
            fontsize=8.9,
            color="#334155",
            va="top",
        )

    metric_specs = [
        ("Pearson", f"{row['overall_pearson']:.3f}"),
        ("Spearman", f"{row['overall_spearman']:.3f}"),
        ("Genes", f"{row['n_samples_built']:,}"),
        ("Tissues", f"{row['n_primary_tissues']}"),
    ]
    start_y = box_y + box_h - 0.05
    for idx, (label, value) in enumerate(metric_specs):
        yy = start_y - idx * 0.060
        ax.text(box_x, yy, label, fontsize=8.6, color="#64748b", va="center")
        ax.text(box_x + 0.105, yy, value, fontsize=10.6, color=row["color"], va="center", fontweight="bold")


def panel_a(ax: plt.Axes, metrics_df: pd.DataFrame) -> None:
    ax.set_title("A  True External Validation Design", loc="left", fontweight="bold", pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_card(ax, 0.02, 0.53, 0.96, 0.39, metrics_df.iloc[0])
    add_card(ax, 0.02, 0.06, 0.96, 0.41, metrics_df.iloc[1])


def panel_b(ax: plt.Axes, metrics_df: pd.DataFrame) -> None:
    ax.set_title("B  Species-level External Accuracy", loc="left", fontweight="bold", pad=8)
    y_pos = np.arange(len(metrics_df))[::-1]
    pearson_x = metrics_df["overall_pearson"].to_numpy()
    spearman_x = metrics_df["overall_spearman"].to_numpy()

    for idx, row in metrics_df.iterrows():
        y = y_pos[idx]
        color = row["color"]
        ax.plot(
            [row["overall_pearson"], row["overall_spearman"]],
            [y, y],
            color=blend(color, amount=0.50),
            lw=2.4,
            zorder=1,
        )
        ax.scatter(row["overall_pearson"], y, s=115, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(
            row["overall_spearman"],
            y,
            s=95,
            color=blend(color, amount=0.18),
            marker="s",
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.text(row["overall_pearson"] - 0.004, y + 0.17, f"{row['overall_pearson']:.3f}", ha="right", color=color, fontsize=10.2)
        ax.text(row["overall_spearman"] + 0.004, y - 0.20, f"{row['overall_spearman']:.3f}", ha="left", color=color, fontsize=10.2)
        ax.text(
            0.684,
            y,
            f"{row['n_samples_built']:,} genes | {row['n_primary_tissues']} tissues",
            va="center",
            ha="left",
            fontsize=9.3,
            color="#475569",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_df["label"].tolist(), fontweight="bold")
    ax.set_xlabel("Correlation on external absolute-expression profiles")
    ax.set_xlim(0.50, 0.705)
    ax.set_ylim(-0.6, len(metrics_df) - 0.4)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#374151", markeredgecolor="white", markersize=9, label="Pearson"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#9ca3af", markeredgecolor="white", markersize=8, label="Spearman"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")


def panel_c(ax: plt.Axes, tissue_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    ax.set_title("C  Tissue-resolved Transferability", loc="left", fontweight="bold", pad=8)

    heatmap = (
        tissue_df.pivot(index="label", columns="tissue", values="pearson")
        .reindex(index=metrics_df["label"], columns=TISSUE_ORDER)
    )
    annot = heatmap.copy().astype(object)
    annot = annot.apply(lambda col: col.map(lambda x: "" if pd.isna(x) else f"{x:.3f}"))

    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap=cmap,
        vmin=0.45,
        vmax=0.70,
        mask=heatmap.isna(),
        annot=annot,
        fmt="",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Per-tissue Pearson", "shrink": 0.74},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([TISSUE_LABELS[t] for t in heatmap.columns], rotation=30, ha="right")
    ax.tick_params(axis="y", length=0)
    ax.axvline(7, color="#94a3b8", lw=1.0, ls=(0, (3, 3)))


def panel_d(ax: plt.Axes, audit_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    ax.set_title("D  Mapping and QC Retention Audit", loc="left", fontweight="bold", pad=8)

    stage_positions = np.arange(len(AUDIT_STAGES))
    max_count = audit_df["count"].max()
    bar_width = 0.32
    offsets = {
        metrics_df.iloc[0]["species"]: -bar_width / 2,
        metrics_df.iloc[1]["species"]: bar_width / 2,
    }

    for _, row in metrics_df.iterrows():
        sub = audit_df[audit_df["species"] == row["species"]].copy()
        y = sub["count"].to_numpy()
        x = stage_positions + offsets[row["species"]]
        bars = ax.bar(
            x,
            y,
            width=bar_width,
            color=row["color"],
            alpha=0.88,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
            label=row["label"],
        )
        for idx, (bar, value) in enumerate(zip(bars, y)):
            xx = bar.get_x() + bar.get_width() / 2
            text = fmt_k(int(value))
            if idx == len(y) - 1:
                retention = y[-1] / y[0]
                text = f"{fmt_k(int(value))}\n{retention:.1%}"
            ax.text(
                xx,
                value + max_count * 0.02,
                text,
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=row["color"],
                fontweight="bold" if idx == len(y) - 1 else None,
                linespacing=1.1,
            )

    ax.set_xticks(stage_positions)
    ax.set_xticklabels([stage for _, stage in AUDIT_STAGES])
    ax.set_ylabel("Genes retained after each audit step")
    ax.set_ylim(0, max_count * 1.24)
    ax.set_xlim(-0.55, stage_positions[-1] + 0.55)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.02, 1.02))


def main() -> None:
    metrics_df, tissue_df, audit_df = load_summary()
    export_source_tables(metrics_df, tissue_df, audit_df)

    fig = plt.figure(figsize=(20.4, 12.6))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.10, 1.42],
        height_ratios=[0.98, 1.14],
        wspace=0.24,
        hspace=0.22,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, metrics_df)
    panel_b(ax_b, metrics_df)
    panel_c(ax_c, tissue_df, metrics_df)
    panel_d(ax_d, audit_df, metrics_df)

    fig.suptitle(
        "True external validation of InsectExpress on two unseen insect species",
        fontsize=16.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.016,
        "Evaluation uses independent expression resources with explicit tissue harmonization and unknown-species inference.",
        ha="center",
        fontsize=10.2,
        color="#475569",
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_BASENAME}.{ext}", dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {OUT_DIR / (FIG_BASENAME + '.png')}")


if __name__ == "__main__":
    main()
