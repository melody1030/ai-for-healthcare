"""Coverage of the mood space (timeline task: "coverage plot").

The proposal notes that DEAM songs are uneven across the valence-arousal
space. Here we measure, for every possible target point, how far the closest
song is. Large gaps tell us where the planner will be less accurate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config


def nearest_gap(songs: pd.DataFrame, points: np.ndarray) -> np.ndarray:
    """Distance from each (valence, arousal) point to its closest song."""
    nn = NearestNeighbors(n_neighbors=1, metric=config.DISTANCE_METRIC)
    nn.fit(songs[["valence", "arousal"]].to_numpy())
    dist, _ = nn.kneighbors(np.asarray(points, dtype=float))
    return dist[:, 0]


def coverage_table(songs: pd.DataFrame, step: float = 1.0) -> pd.DataFrame:
    """Gap to the closest song for every point on a grid over 1-9 x 1-9.

    With ``step=1`` this is the 81 integer slider settings.
    """
    axis = np.arange(config.SCALE_MIN, config.SCALE_MAX + 1e-9, step)
    vv, aa = np.meshgrid(axis, axis)
    pts = np.column_stack([vv.ravel(), aa.ravel()])
    gaps = nearest_gap(songs, pts)
    return pd.DataFrame(
        {
            "valence": pts[:, 0],
            "arousal": pts[:, 1],
            "gap": gaps,
            "covered": gaps <= config.GAP_WARNING,
        }
    )


def summarise(table: pd.DataFrame) -> str:
    covered = table["covered"].mean() * 100
    worst = table.sort_values("gap", ascending=False).head(5)
    lines = [
        f"Grid points: {len(table)}",
        f"Median gap to closest song: {table['gap'].median():.2f}",
        f"Max gap: {table['gap'].max():.2f}",
        f"Points within {config.GAP_WARNING} of a song: {covered:.1f}%",
        "Worst-covered targets (valence, arousal -> gap):",
    ]
    lines += [f"  ({r.valence:.0f}, {r.arousal:.0f}) -> {r.gap:.2f}" for r in worst.itertuples()]
    return "\n".join(lines)


def plot_coverage(songs: pd.DataFrame, out_path: str | Path, title_note: str = "") -> Path:
    """Two panels: where the songs are, and how far each target is from a song."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = config.SCALE_MIN, config.SCALE_MAX
    fine = coverage_table(songs, step=0.1)
    n = int(round(np.sqrt(len(fine))))
    grid = fine["gap"].to_numpy().reshape(n, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.scatter(songs["valence"], songs["arousal"], s=8, alpha=0.5, color="#2f6f6a")
    ax.set_title(f"DEAM songs in mood space (n = {len(songs)})")

    ax = axes[1]
    im = ax.imshow(
        grid, origin="lower", extent=(lo, hi, lo, hi), cmap="magma_r", aspect="equal"
    )
    cs = ax.contour(
        np.linspace(lo, hi, n), np.linspace(lo, hi, n), grid,
        levels=[config.GAP_WARNING], colors="white", linewidths=1.2,
    )
    ax.clabel(cs, fmt={config.GAP_WARNING: f"gap = {config.GAP_WARNING}"}, fontsize=8)
    fig.colorbar(im, ax=ax, label="distance to closest song")
    ax.set_title("Gap between a target mood and the closest song")

    for ax in axes:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Valence (1 = negative, 9 = positive)")
        ax.set_ylabel("Arousal (1 = calm, 9 = excited)")
        ax.axhline(5, color="grey", lw=0.6)
        ax.axvline(5, color="grey", lw=0.6)
        ax.set_aspect("equal")

    if title_note:
        fig.suptitle(title_note, fontsize=10)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
