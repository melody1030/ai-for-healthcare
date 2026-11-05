"""DEAM data discovery, loading, and cleaning utilities."""

from pathlib import Path

import pandas as pd


ANNOTATION_FILES = (
    "static_annotations_averaged_songs_1_2000.csv",
    "static_annotations_averaged_songs_2000_2058.csv",
)
SCALE_MIN, SCALE_MAX = 1.0, 9.0
RENAME = {
    "valence_mean": "valence",
    "arousal_mean": "arousal",
    "valence_std": "valence_std",
    "arousal_std": "arousal_std",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def find_file(root: Path, name: str) -> Path:
    """Find a named file recursively below ``root``."""
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find '{name}' under '{root}'.")
    return matches[0]


def load_annotations(data_dir: Path) -> pd.DataFrame:
    """Load and combine the two DEAM static annotation CSV files."""
    frames = []
    for name in ANNOTATION_FILES:
        df = pd.read_csv(find_file(data_dir, name))
        df.columns = [c.strip().replace(" ", "") for c in df.columns]
        frames.append(df[["song_id", *RENAME]].rename(columns=RENAME))
    songs = pd.concat(frames, ignore_index=True)
    songs["song_id"] = songs["song_id"].astype(int)
    return songs.drop_duplicates("song_id").sort_values("song_id").reset_index(drop=True)


def find_audio(data_dir: Path) -> dict[int, Path]:
    """Find numeric ``<song_id>.mp3`` files recursively below ``data_dir``."""
    return {
        int(path.stem): path
        for path in data_dir.rglob("*.mp3")
        if path.stem.isdigit()
    }


def clean_songs(
    songs: pd.DataFrame,
    audio: dict[int, Path],
    spread_quantile: float = 0.90,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Apply the project's fixed DEAM rating, audio, and spread filters."""
    report: dict[str, int | str] = {"Raw songs": len(songs)}
    df = songs.copy()

    has_rating = df[["valence", "arousal", "valence_std", "arousal_std"]].notna().all(axis=1)
    report["Dropped, missing rating"] = int((~has_rating).sum())
    df = df[has_rating]

    in_range = df["valence"].between(SCALE_MIN, SCALE_MAX) & df["arousal"].between(
        SCALE_MIN, SCALE_MAX
    )
    report["Dropped, rating out of 1-9"] = int((~in_range).sum())
    df = df[in_range]

    if audio:
        df = df.assign(audio_path=df["song_id"].map(audio))
        has_audio = df["audio_path"].notna()
        report["Dropped, missing audio"] = int((~has_audio).sum())
        df = df[has_audio].assign(audio_path=lambda d: d["audio_path"].astype(str))
    else:
        df = df.assign(audio_path=None)
        report["Dropped, missing audio"] = "skipped (no MP3 files found)"

    df = df.assign(spread=(df["valence_std"] + df["arousal_std"]) / 2)
    cutoff = df["spread"].quantile(spread_quantile)
    keep = df["spread"] <= cutoff
    report[f"Dropped, high rater spread (> {cutoff:.3f})"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    report["Clean songs"] = len(df)
    return df, report
