#!/usr/bin/env python3
"""
Figure 1C: InsectExpress model architecture diagram
Draws a publication-quality architecture figure using matplotlib.
Style reference: Enformer Extended Data Fig 1 (Avsec et al., Nat Methods 2021)

Output: results/paper/figures/fig1c_architecture.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / 'results' / 'paper' / 'figures'

# color palette
C_DNA = '#3498db'
C_RNA = '#e67e22'
C_ESM = '#e74c3c'
C_CONV = '#2980b9'
C_TRANS = '#8e44ad'
C_FUSION = '#27ae60'
C_OUT = '#2c3e50'
C_BG = '#f8f9fa'


def draw_block(ax, x, y, w, h, label, sublabel, color, alpha=0.85):
    """draw a rounded rectangle block with label"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         facecolor=color, alpha=alpha, edgecolor='#333',
                         linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.6, label, ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    if sublabel:
        ax.text(x + w/2, y + h*0.25, sublabel, ha='center', va='center',
                fontsize=6, color='#eee', style='italic')


def draw_arrow(ax, x1, y1, x2, y2, color='#555'):
    """draw arrow between blocks"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    bw, bh = 1.8, 0.6  # block width, height

    # === Tower 1: DNA Sequence (left) ===
    x_dna = 0.5
    # input
    draw_block(ax, x_dna, 7.5, bw, bh, '50kb DNA', '4 x 50,000', C_DNA)
    # conv stem
    draw_block(ax, x_dna, 6.3, bw, bh, 'MultiScale Conv', 'k=7,11,15,21', C_CONV)
    draw_arrow(ax, x_dna+bw/2, 7.5, x_dna+bw/2, 6.3+bh)
    draw_block(ax, x_dna, 5.3, bw, bh, '5x Dilated Res', 'd=1,2,4,8,16', C_CONV)
    draw_arrow(ax, x_dna+bw/2, 6.3, x_dna+bw/2, 5.3+bh)
    draw_block(ax, x_dna, 4.3, bw, bh, '4x Conv+Pool', '64x downsample', C_CONV)
    draw_arrow(ax, x_dna+bw/2, 5.3, x_dna+bw/2, 4.3+bh)
    # transformer
    draw_block(ax, x_dna, 3.1, bw, bh+0.15, 'Transformer', '4L x 256d x 8H', C_TRANS)
    draw_arrow(ax, x_dna+bw/2, 4.3, x_dna+bw/2, 3.1+bh+0.15)
    # global pool
    draw_block(ax, x_dna, 2.0, bw, 0.5, 'Global AvgPool', '256-d', '#7f8c8d')
    draw_arrow(ax, x_dna+bw/2, 3.1, x_dna+bw/2, 2.0+0.5)

    # dim annotations (right side of DNA tower)
    dims = [('50,000 x 4', 7.7), ('25,000 x 64', 6.5), ('25,000 x 128', 5.5),
            ('781 x 256', 4.5), ('781 x 256', 3.35), ('256', 2.15)]
    for dim_text, yy in dims:
        ax.text(x_dna+bw+0.15, yy, dim_text, fontsize=6, color='#888',
                va='center', fontstyle='italic')

    # === Tower 2: RNA Features (center) ===
    x_rna = 4.0
    draw_block(ax, x_rna, 7.5, bw, bh, 'RNA Features', '16-dim', C_RNA)
    draw_block(ax, x_rna, 5.5, bw, bh, 'MLP Encoder', '16→128→64', C_RNA, alpha=0.7)
    draw_arrow(ax, x_rna+bw/2, 7.5, x_rna+bw/2, 5.5+bh)
    ax.text(x_rna+bw+0.15, 5.7, '64', fontsize=6, color='#888',
            va='center', fontstyle='italic')

    # === Tower 3: ESM-2 (right) ===
    x_esm = 7.2
    draw_block(ax, x_esm, 7.5, bw, bh, 'ESM-2 Embed', '320-dim', C_ESM)
    draw_block(ax, x_esm, 5.5, bw, bh, 'Protein Encoder', '320→128→64', C_ESM, alpha=0.7)
    draw_arrow(ax, x_esm+bw/2, 7.5, x_esm+bw/2, 5.5+bh)
    # fallback annotation
    ax.text(x_esm+bw+0.15, 7.7, 'or species\nembedding', fontsize=6,
            color='#999', va='center')
    ax.text(x_esm+bw+0.15, 5.7, '64', fontsize=6, color='#888',
            va='center', fontstyle='italic')

    # === Fusion ===
    x_fuse = 3.5
    fw = 3.5
    draw_block(ax, x_fuse, 1.0, fw, bh, 'Concatenation + Fusion MLP',
               '384 → 512 → 256', C_FUSION)
    # arrows into fusion
    draw_arrow(ax, x_dna+bw/2, 2.0, x_fuse+0.5, 1.0+bh)
    draw_arrow(ax, x_rna+bw/2, 5.5, x_fuse+fw/2, 1.0+bh)
    draw_arrow(ax, x_esm+bw/2, 5.5, x_fuse+fw-0.5, 1.0+bh)

    # === Output ===
    draw_block(ax, x_fuse+0.5, -0.2, fw-1, bh, '14-Tissue Expression Output',
               '+ species×tissue bias', C_OUT)
    draw_arrow(ax, x_fuse+fw/2, 1.0, x_fuse+fw/2, -0.2+bh)

    # === Title & annotations ===
    ax.text(5.25, 8.3, 'InsectExpress Architecture (~5.2M parameters)',
            ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

    # tower labels
    ax.text(x_dna+bw/2, 8.0, 'Sequence Tower', ha='center', fontsize=9,
            fontweight='bold', color=C_DNA)
    ax.text(x_rna+bw/2, 8.0, 'RNA Stability\nTower', ha='center', fontsize=9,
            fontweight='bold', color=C_RNA)
    ax.text(x_esm+bw/2, 8.0, 'Protein\nTower', ha='center', fontsize=9,
            fontweight='bold', color=C_ESM)

    # loss annotation
    ax.text(x_fuse+fw+0.3, -0.0, 'Loss = MSE + 0.3 * (1-r)\n+ 0.05 * ortholog',
            fontsize=7, color='#555', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0',
                      edgecolor='#f39c12', alpha=0.8))

    # Conv stem bracket
    ax.annotate('ConvStem\n64x down', xy=(x_dna-0.15, 4.3), xytext=(x_dna-0.4, 5.8),
                fontsize=7, color='#2980b9', fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='-[, widthB=2.8', color='#2980b9', lw=1.5))

    out_pdf = OUT_DIR / 'fig1c_architecture.pdf'
    out_png = OUT_DIR / 'fig1c_architecture.png'
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300, facecolor='white')
    fig.savefig(out_png, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
