"""Render the JOSS paper architecture figure.

Usage:
    uvx --with matplotlib python scripts/make_paper_figure.py

Outputs:
    paper/figures/architecture.pdf  — vector, embedded by paper/paper.md
    paper/figures/architecture.png  — preview
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"

INK = "#1a1a2e"
BOX_INPUT = "#dbeafe"
BOX_A = "#fae9d9"
BOX_B = "#dcf0e4"
BOX_OUT = "#efe6f7"
EDGE = {"edgecolor": INK, "linewidth": 1.0}


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    face: str,
    lines: list[str],
    title: str | None = None,
    title_size: float = 8.5,
    size: float = 7.6,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=face,
        **EDGE,
    )
    ax.add_patch(patch)
    cx, cy = x + w / 2, y + h / 2
    if title:
        ax.text(
            cx,
            y + h - 0.16,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=INK,
        )
        body = "\n".join(lines)
        ax.text(
            cx,
            y + (h - 0.3) / 2,
            body,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            linespacing=1.45,
        )
    else:
        ax.text(
            cx,
            cy,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            linespacing=1.45,
        )


def arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=INK,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.4), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    # pipeline spine
    box(ax, 0.05, 1.45, 1.30, 0.55, BOX_INPUT, ["interview answer\n(wav / mp3 / m4a)"])
    box(ax, 1.70, 1.45, 1.30, 0.55, BOX_INPUT, ["preprocess\n16 kHz mono"])
    box(ax, 6.95, 1.45, 1.35, 0.55, BOX_OUT, ["median aggregation\nper-segment scores"])
    box(
        ax,
        8.60,
        1.45,
        1.35,
        0.55,
        BOX_OUT,
        [
            "$P(\\mathrm{read}\\,|\\,\\mathrm{audio})$",
            "SNR warning\nfeature attribution",
        ],
    )

    # tracks
    box(
        ax,
        3.45,
        2.30,
        3.10,
        0.95,
        BOX_A,
        [
            "energy VAD · 34 prosodic features",
            "(F0, jitter/shimmer, pause",
            "distribution, nPVI rhythm) →",
            "LightGBM + isotonic calibration",
        ],
        title="Track A — interpretable",
    )
    box(
        ax,
        3.45,
        0.20,
        3.10,
        0.95,
        BOX_B,
        [
            "frozen wav2vec 2.0 backbone",
            "mean+std pooled embeddings →",
            "MLP head (BCE)",
            "chunk-level scoring",
        ],
        title="Track B — performance",
    )

    # arrows: spine -> split
    arrow(ax, 1.35, 1.72, 1.70, 1.72)
    arrow(ax, 3.00, 1.72, 3.45, 2.72, rad=-0.12)
    arrow(ax, 3.00, 1.72, 3.45, 0.72, rad=0.12)
    # tracks -> aggregate
    arrow(ax, 6.55, 2.72, 6.95, 1.75, rad=0.12)
    arrow(ax, 6.55, 0.72, 6.95, 1.72, rad=-0.12)
    arrow(ax, 8.30, 1.72, 8.60, 1.72)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "architecture.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"figure written to {OUT}")


if __name__ == "__main__":
    main()
