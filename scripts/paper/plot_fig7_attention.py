#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the revised Fig. 7 from summary attention statistics.

Figure layout:
  A. Species-balanced TSS attention meta-profile
  B. Layer-by-distance attention enrichment heatmap
  C. Species-wise forest plot of TSS focus differences
  D. Representative broad and tissue-specific gene cases
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 1.1,
    "axes.titleweight": "bold",
    "figure.dpi": 200,
})

DATA = Path("/home/hjd/RNAi/results/paper/attention_summary_v2.npz")
OUT = Path("/home/hjd/RNAi/results/paper/figures")
GROUP_COLORS = {
    "broad": "#2C7FB8",
    "tissue_specific": "#D95F0E",
}
GROUP_LABELS = {
    "broad": "Broadly expressed",
    "tissue_specific": "Tissue-specific",
}
PANEL_ORDER = ["broad", "tissue_specific"]
N_BOOT = 2000


def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return y
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(y_pad, kernel, mode="valid")


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx].mean(axis=1)
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a_idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    b_idx = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = a[a_idx].mean(axis=1) - b[b_idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def load() -> dict:
    d = np.load(DATA, allow_pickle=True)
    return {k: d[k] for k in d.files}


def species_balanced_profiles(data: dict, group: str) -> np.ndarray:
    mask = data["groups"] == group
    species = np.unique(data["species"][mask])
    rows = []
    for sp in species:
        sp_mask = mask & (data["species"] == sp)
        if sp_mask.sum() == 0:
            continue
        sp_mean = data["profiles"][sp_mask].mean(axis=0)
        rows.append(moving_average(sp_mean, window=5))
    return np.stack(rows)


def build_panel_a(ax, data: dict) -> None:
    x = data["positions_kb"]
    for group in PANEL_ORDER:
        sp_profiles = species_balanced_profiles(data, group)
        mean_profile = sp_profiles.mean(axis=0)
        ci_lo, ci_hi = bootstrap_mean_ci(sp_profiles)
        ax.plot(x, mean_profile, lw=2.4, color=GROUP_COLORS[group], label=GROUP_LABELS[group])
        ax.fill_between(x, ci_lo, ci_hi, color=GROUP_COLORS[group], alpha=0.18)
    ax.axvspan(-1.0, 1.0, color="#D9D9D9", alpha=0.25, zorder=0)
    ax.axvline(0.0, color="#4D4D4D", lw=1.0, ls="--", alpha=0.75)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("Distance from TSS (kb)")
    ax.set_ylabel("Normalized attention received")
    ax.set_title("A  Species-Balanced TSS Attention Meta-profile", loc="left")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_panel_b(ax, data: dict) -> None:
    species = np.unique(data["species"])
    sp_means = []
    for sp in species:
        sp_mask = data["species"] == sp
        if sp_mask.sum() == 0:
            continue
        sp_means.append(data["layer_bin_enrich"][sp_mask].mean(axis=0))
    heat = np.stack(sp_means).mean(axis=0)

    vmin = min(0.9, float(heat.min()))
    vmax = max(1.2, float(heat.max()))
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    im = ax.imshow(heat, cmap="RdYlBu_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(data["distance_labels"])))
    ax.set_xticklabels(data["distance_labels"], rotation=35, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels([f"L{i+1}" for i in range(heat.shape[0])])
    ax.set_xlabel("Attention distance bin")
    ax.set_ylabel("Transformer layer")
    ax.set_title("B  Layer-by-Distance Attention Enrichment", loc="left")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val - 1.0) > 0.18 else "black", fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fold enrichment over uniform")


def species_effects(data: dict) -> list[dict]:
    effects = []
    for sp in np.unique(data["species"]):
        broad = data["tss_focus"][(data["species"] == sp) & (data["groups"] == "broad")]
        specific = data["tss_focus"][(data["species"] == sp) & (data["groups"] == "tissue_specific")]
        if len(broad) < 8 or len(specific) < 8:
            continue
        effect = float(specific.mean() - broad.mean())
        lo, hi = bootstrap_diff_ci(specific, broad, seed=42 + len(effects))
        effects.append({
            "species": str(sp),
            "effect": effect,
            "lo": lo,
            "hi": hi,
            "n_broad": int(len(broad)),
            "n_specific": int(len(specific)),
        })
    effects.sort(key=lambda x: x["effect"], reverse=True)
    return effects


def build_panel_c(ax, data: dict) -> None:
    effects = species_effects(data)
    if not effects:
        ax.text(0.5, 0.5, "Insufficient species-level samples", ha="center", va="center")
        ax.set_axis_off()
        return

    overall_vals = np.array([e["effect"] for e in effects], dtype=np.float32)
    overall_mean = float(overall_vals.mean())
    overall_lo, overall_hi = bootstrap_mean_ci(overall_vals[:, None], seed=123)

    labels = ["Overall"] + [e["species"] for e in effects]
    effects_full = [overall_mean] + [e["effect"] for e in effects]
    lo_full = [float(overall_lo[0])] + [e["lo"] for e in effects]
    hi_full = [float(overall_hi[0])] + [e["hi"] for e in effects]

    y = np.arange(len(labels))
    ax.axvline(0.0, color="#808080", lw=1.0, ls="--")
    for yi, eff, lo, hi in zip(y, effects_full, lo_full, hi_full):
        color = "#C51B7D" if yi == 0 else "#4D4D4D"
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.8, solid_capstyle="round")
        ax.scatter(eff, yi, s=42 if yi == 0 else 34, color=color, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Δ TSS focus (tissue-specific - broad)")
    ax.set_title("C  Species-wise TSS Focus Effect", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def abbreviate_tissues(names: np.ndarray) -> list[str]:
    out = []
    for name in names:
        parts = str(name).split("_")
        if len(parts) >= 2:
            out.append(f"{parts[0][0]}.{parts[1]}")
        else:
            out.append(str(name))
    return out


def plot_case_profile(ax, x: np.ndarray, profile: np.ndarray, color: str, title: str, panel_title: str | None = None) -> None:
    ax.plot(x, moving_average(profile, window=5), color=color, lw=2.0)
    ax.axvspan(-1.0, 1.0, color="#D9D9D9", alpha=0.25, zorder=0)
    ax.axvline(0.0, color="#4D4D4D", lw=1.0, ls="--", alpha=0.75)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylabel("Attention")
    ax.set_title(f"{panel_title}{title}" if panel_title else title, loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_case_heatmap(ax, obs: np.ndarray, pred: np.ndarray, mask: np.ndarray, tissue_labels: list[str],
                      vmin: float, vmax: float) -> matplotlib.image.AxesImage:
    mat = np.vstack([obs, pred]).astype(np.float32)
    valid = np.vstack([mask, mask]) > 0.5
    mat[~valid] = np.nan
    cmap = plt.cm.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#F2F2F2")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Obs", "Pred"])
    ax.set_xticks(np.arange(len(tissue_labels)))
    ax.set_xticklabels(tissue_labels, rotation=45, ha="right", fontsize=8)
    return im


def build_panel_d(fig: plt.Figure, spec, data: dict) -> None:
    case_indices = data["case_indices"]
    if case_indices.size == 0:
        ax = fig.add_subplot(spec)
        ax.text(0.5, 0.5, "Representative cases unavailable", ha="center", va="center")
        ax.set_axis_off()
        return

    tissue_labels = abbreviate_tissues(data["tissue_names"])
    case_obs = data["observed"][case_indices]
    case_pred = data["predicted"][case_indices]
    case_mask = data["expression_mask"][case_indices] > 0.5
    valid_vals = np.concatenate([
        case_obs[case_mask],
        case_pred[case_mask],
    ])
    vmin = float(np.nanmin(valid_vals))
    vmax = float(np.nanmax(valid_vals))

    gs = GridSpecFromSubplotSpec(len(case_indices), 2, subplot_spec=spec,
                                 width_ratios=[1.35, 1.0], hspace=0.55, wspace=0.30)
    x = data["positions_kb"]
    heat_axes = []
    im = None

    for row_i, idx in enumerate(case_indices):
        group = str(data["groups"][idx])
        species = str(data["species"][idx])
        gene_id = str(data["gene_ids"][idx])
        mae = float(data["mae"][idx])
        tau = float(data["tau"][idx])

        ax_profile = fig.add_subplot(gs[row_i, 0])
        panel_title = "D  Representative genes: " if row_i == 0 else None
        title = f"{GROUP_LABELS[group]} | {species} | {gene_id} (tau={tau:.2f}, MAE={mae:.2f})"
        plot_case_profile(
            ax_profile,
            x,
            data["profiles"][idx],
            GROUP_COLORS[group],
            title=title,
            panel_title=panel_title,
        )
        if row_i == len(case_indices) - 1:
            ax_profile.set_xlabel("Distance from TSS (kb)")
        else:
            ax_profile.set_xlabel("")
            ax_profile.set_xticklabels([])

        ax_heat = fig.add_subplot(gs[row_i, 1])
        im = plot_case_heatmap(
            ax_heat,
            data["observed"][idx],
            data["predicted"][idx],
            data["expression_mask"][idx],
            tissue_labels,
            vmin=vmin,
            vmax=vmax,
        )
        if row_i == 0:
            ax_heat.set_title("Observed vs predicted", fontsize=11)
        heat_axes.append(ax_heat)

    if im is not None:
        fig.colorbar(im, ax=heat_axes, fraction=0.018, pad=0.02, label="log2(TPM+1)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()

    fig = plt.figure(figsize=(17.5, 13.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    build_panel_a(ax_a, data)

    ax_b = fig.add_subplot(gs[0, 1])
    build_panel_b(ax_b, data)

    ax_c = fig.add_subplot(gs[1, 0])
    build_panel_c(ax_c, data)

    build_panel_d(fig, gs[1, 1], data)

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT / f"fig7_attention.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig7_attention.png'} and .pdf")


if __name__ == "__main__":
    main()
