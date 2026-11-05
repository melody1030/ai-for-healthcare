"""DEAM loading and cleaning (timeline task: "DEAM loading, cleaning").

Expected input: the Kaggle archive
https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music
unzipped into ``data/DEAM``. We search the folder recursively, so the exact
sub-folder layout does not matter. We need:

* ``static_annotations_averaged_songs_1_2000.csv``    (songs 2-2000)
* ``static_annotations_averaged_songs_2000_2058.csv`` (songs 2001-2058)
* the MP3 clips, named ``<song_id>.mp3`` (``DEAM_audio/MEMD_audio``)

Output: one row per song with ``song_id, valence, arousal, valence_std,
arousal_std, spread, audio_path`` on the same 1-9 scale as the mood sliders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

ANNOTATION_FILES = (
    "static_annotations_averaged_songs_1_2000.csv",
    "static_annotations_averaged_songs_2000_2058.csv",
)

# Columns we keep from the raw files, renamed to short names.
_RENAME = {
    "valence_mean": "valence",
    "arousal_mean": "arousal",
    "valence_std": "valence_std",
    "arousal_std": "arousal_std",
}


@dataclass
class CleaningReport:
    """How many songs each cleaning step removed. Goes into the final report."""

    n_raw: int = 0
    n_missing_rating: int = 0
    n_out_of_range: int = 0
    n_missing_audio: int = 0
    n_high_spread: int = 0
    spread_cutoff: float = float("nan")
    n_clean: int = 0
    notes: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"Raw songs:                    {self.n_raw}",
            f"Dropped, missing rating:      {self.n_missing_rating}",
            f"Dropped, rating out of 1-9:   {self.n_out_of_range}",
            f"Dropped, missing audio file:  {self.n_missing_audio}",
            f"Dropped, high rater spread:   {self.n_high_spread}"
            f"  (spread > {self.spread_cutoff:.3f})",
            f"Clean songs:                  {self.n_clean}",
        ]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def _find_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{name}' under '{root}'. Download the DEAM archive "
            "from Kaggle and unzip it there (see README)."
        )
    return matches[0]


def load_annotations(data_dir: str | Path = config.DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load and stack the two DEAM song-level rating files.

    The raw CSVs have spaces in their headers (for example ``" valence_mean"``)
    and the second file has extra columns, so we normalise both first.
    """
    root = Path(data_dir)
    frames = []
    for name in ANNOTATION_FILES:
        df = pd.read_csv(_find_file(root, name))
        df.columns = [c.strip().replace(" ", "") for c in df.columns]
        missing = {"song_id", *_RENAME} - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        frames.append(df[["song_id", *_RENAME]].rename(columns=_RENAME))
    songs = pd.concat(frames, ignore_index=True)
    songs["song_id"] = songs["song_id"].astype(int)
    return songs.drop_duplicates("song_id").sort_values("song_id").reset_index(drop=True)


def find_audio(data_dir: str | Path = config.DEFAULT_DATA_DIR) -> dict[int, Path]:
    """Map song_id -> MP3 path for every ``<digits>.mp3`` under ``data_dir``."""
    audio = {}
    for path in Path(data_dir).rglob("*.mp3"):
        if path.stem.isdigit():
            audio[int(path.stem)] = path
    return audio


def clean_songs(
    songs: pd.DataFrame,
    audio: dict[int, Path] | None,
    spread_quantile: float = config.SPREAD_QUANTILE,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the data-plan rules from the proposal.

    1. Drop songs missing a rating.
    2. Drop songs with a rating outside 1-9 (sanity check).
    3. Drop songs missing an audio file. If ``audio`` is None (no clips
       downloaded yet) this step is skipped and a note is added.
    4. Drop songs with the largest rater spread (above ``spread_quantile``).
    """
    rep = CleaningReport(n_raw=len(songs))
    df = songs.copy()
    rating_cols = ["valence", "arousal", "valence_std", "arousal_std"]

    has_rating = df[rating_cols].notna().all(axis=1)
    rep.n_missing_rating = int((~has_rating).sum())
    df = df[has_rating]

    in_range = df[["valence", "arousal"]].apply(
        lambda s: s.between(config.SCALE_MIN, config.SCALE_MAX)
    ).all(axis=1)
    rep.n_out_of_range = int((~in_range).sum())
    df = df[in_range]

    if audio is None or len(audio) == 0:
        df = df.assign(audio_path=pd.NA)
        rep.notes.append("No MP3 files found; audio check skipped.")
    else:
        df = df.assign(audio_path=df["song_id"].map(lambda i: audio.get(i)))
        has_audio = df["audio_path"].notna()
        rep.n_missing_audio = int((~has_audio).sum())
        df = df[has_audio]
        df["audio_path"] = df["audio_path"].map(str)

    df = df.assign(spread=(df["valence_std"] + df["arousal_std"]) / 2)
    cutoff = float(np.quantile(df["spread"], spread_quantile)) if len(df) else np.nan
    keep = df["spread"] <= cutoff
    rep.spread_cutoff = cutoff
    rep.n_high_spread = int((~keep).sum())
    df = df[keep]

    rep.n_clean = len(df)
    cols = ["song_id", "valence", "arousal", "valence_std", "arousal_std", "spread", "audio_path"]
    return df[cols].reset_index(drop=True), rep


def load_clean_songs(
    data_dir: str | Path = config.DEFAULT_DATA_DIR,
    spread_quantile: float = config.SPREAD_QUANTILE,
) -> tuple[pd.DataFrame, CleaningReport]:
    """One call: load ratings, find clips, clean."""
    songs = load_annotations(data_dir)
    audio = find_audio(data_dir)
    return clean_songs(songs, audio or None, spread_quantile)


def read_clean_csv(path: str | Path = config.DEFAULT_CLEAN_CSV) -> pd.DataFrame:
    """Read the cleaned song table written by ``scripts/prepare_data.py``."""
    return pd.read_csv(path)
