"""Build the iso-guided and direct-to-target playlists for one mood input.

Usage (after scripts/prepare_data.py):
    python scripts/make_playlist.py --current 3 7 --target 7 3
    python scripts/make_playlist.py --current 3 7 --target 7 3 --plot results/example_paths.png

--current and --target are "valence arousal" on the 1-9 slider scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from isoharmonix import config  # noqa: E402
from isoharmonix.data import read_clean_csv  # noqa: E402
from isoharmonix.planner import ISO, Mood, SongRetriever, build_session, match_quality  # noqa: E402


def plot_session(songs: pd.DataFrame, session: pd.DataFrame, current: Mood, target: Mood, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 6), constrained_layout=True)
    ax.scatter(songs["valence"], songs["arousal"], s=5, color="#cccccc", label="DEAM songs")
    styles = {ISO: ("#2f6f6a", "-", "Iso-guided"), "direct": ("#b5532a", "--", "Direct-to-target")}
    for cond, part in session.groupby("condition", sort=False):
        color, ls, label = styles[cond]
        ax.plot(part["target_valence"], part["target_arousal"], ls, color=color, lw=1, alpha=0.6)
        ax.scatter(part["song_valence"], part["song_arousal"], s=60, color=color, label=f"{label} songs", zorder=3)
        if cond == ISO:
            for r in part.itertuples():
                ax.annotate(str(r.step), (r.song_valence, r.song_arousal), xytext=(5, 5),
                            textcoords="offset points", fontsize=9, color=color)
    ax.scatter(*current.as_array(), marker="*", s=220, color="black", label="Current mood", zorder=4)
    ax.scatter(*target.as_array(), marker="X", s=140, color="black", label="Target mood", zorder=4)
    ax.set_xlim(config.SCALE_MIN, config.SCALE_MAX)
    ax.set_ylim(config.SCALE_MIN, config.SCALE_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"Playlists from ({current.valence:g}, {current.arousal:g}) to ({target.valence:g}, {target.arousal:g})")
    ax.legend(fontsize=8, loc="best")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--current", nargs=2, type=float, required=True, metavar=("V", "A"))
    p.add_argument("--target", nargs=2, type=float, required=True, metavar=("V", "A"))
    p.add_argument("--songs", default=config.DEFAULT_CLEAN_CSV)
    p.add_argument("--out", help="optional CSV path for the session table")
    p.add_argument("--plot", help="optional PNG path for a path plot")
    args = p.parse_args()

    songs = read_clean_csv(args.songs)
    current, target = Mood(*args.current), Mood(*args.target)
    session = build_session(SongRetriever(songs), current, target)

    cols = ["condition", "step", "target_valence", "target_arousal",
            "song_id", "song_valence", "song_arousal", "distance", "gap_warning"]
    print(session[cols].to_string(index=False))
    print("\nMatch quality (distance from target point to song):")
    print(match_quality(session).to_string())
    print("\nWhy-card, iso step 2:\n  " + session.loc[(session.condition == ISO) & (session.step == 2), "why"].iloc[0])

    if args.out:
        session.to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")
    if args.plot:
        plot_session(songs, session, current, target, Path(args.plot))
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
