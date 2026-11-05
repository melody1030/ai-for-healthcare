"""DEAM annotation loading, audio discovery, and the existing cleaning rules.

Audio files are discovered under a data directory. Nothing here hardcodes a
machine path. Set ``DEAM_DATA_DIR`` to point at a local copy, or place the
unzipped dataset at ``data/DEAM/`` in the repository. The dataset is never
downloaded by this module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from isoharmonix.rules import SCALE_MAX, SCALE_MIN, SPREAD_QUANTILE

ANNOTATION_FILES = (
    "static_annotations_averaged_songs_1_2000.csv",
    "static_annotations_averaged_songs_2000_2058.csv",
)
RENAME = {
    "valence_mean": "valence",
    "arousal_mean": "arousal",
    "valence_std": "valence_std",
    "arousal_std": "arousal_std",
}
_NEEDED_COLUMNS = ("song_id", *RENAME)


def find_repo_root(start: Path | None = None) -> Path:
    """Repository root, whether the process was started in the root or in ``notebooks/``."""
    here = Path.cwd() if start is None else Path(start)
    here = here.resolve()
    for candidate in (here, *here.parents):
        if (candidate / "isoharmonix").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    return here


def resolve_deam_dir(root: Path | None = None) -> Path:
    """Local DEAM directory.

    ``DEAM_DATA_DIR`` wins when it is set. Otherwise this is ``<repo>/data/DEAM``.
    The directory does not have to exist yet; callers decide what to do when it
    is missing. This function never downloads the dataset.
    """
    env = os.environ.get("DEAM_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    base = find_repo_root() if root is None else Path(root)
    return (base / "data" / "DEAM").resolve()


def find_file(root, name: str) -> Path:
    matches = sorted(Path(root).rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name!r} under {root!r}.")
    return matches[0]


def load_annotations(data_dir) -> pd.DataFrame:
    """Load the two song-level DEAM rating files.

    Headers may contain spaces (for example ``" valence_mean"``). The second
    file has extra columns; only the song id and the valence/arousal mean and
    standard deviation are kept. If a song id appears in both files, the row
    from ``static_annotations_averaged_songs_1_2000.csv`` is kept.
    """
    frames = []
    for name in ANNOTATION_FILES:
        path = find_file(data_dir, name)
        df = pd.read_csv(path)
        df.columns = [str(c).strip().replace(" ", "") for c in df.columns]
        missing = [c for c in _NEEDED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}.")
        frames.append(df.loc[:, _NEEDED_COLUMNS].rename(columns=RENAME))
    songs = pd.concat(frames, ignore_index=True)
    songs["song_id"] = songs["song_id"].astype(int)
    return songs.drop_duplicates("song_id").sort_values("song_id").reset_index(drop=True)


@dataclass(frozen=True)
class AudioScan:
    """MP3s found under a DEAM directory, keyed by the numeric file stem.

    ``playable`` maps a song id to one readable file. ``unreadable_ids`` are
    ids whose files all failed the decoder (or were empty). Ids with no file
    at all are simply absent from both maps.
    """

    files_found: int
    playable: dict
    unreadable_ids: tuple
    unreadable_files: tuple
    duplicate_ids: tuple

    @property
    def mp3_count(self) -> int:
        return self.files_found

    @property
    def unreadable_count(self) -> int:
        return len(self.unreadable_files)


def scan_audio(data_dir) -> AudioScan:
    """Find ``<song_id>.mp3`` files at any depth under ``data_dir``.

    Leading zeros are ignored (``0002.mp3`` matches song 2). The suffix match
    is case-insensitive. Non-numeric names are ignored, because the DEAM
    clips used by this study are named by song id. When several files share
    an id, a decodable file is preferred; otherwise the lexicographically
    first path is recorded as unreadable.
    """
    root = Path(data_dir)
    grouped: dict[int, list[Path]] = {}
    if root.exists():
        paths = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() == ".mp3" and p.stem.isdigit()
        ]
        paths.sort(key=lambda p: str(p.resolve()))
        for path in paths:
            grouped.setdefault(int(path.stem), []).append(path)

    playable = {}
    unreadable_ids = []
    unreadable_files = []
    duplicate_ids = []
    for song_id, candidates in grouped.items():
        if len(candidates) > 1:
            duplicate_ids.append(song_id)
        good = None
        for path in candidates:
            if classify_audio(path) == "ok":
                if good is None:
                    good = path
            else:
                unreadable_files.append(path)
        if good is None:
            unreadable_ids.append(song_id)
        else:
            playable[song_id] = good

    return AudioScan(
        files_found=sum(len(v) for v in grouped.values()),
        playable=playable,
        unreadable_ids=tuple(unreadable_ids),
        unreadable_files=tuple(unreadable_files),
        duplicate_ids=tuple(duplicate_ids),
    )


def classify_audio(path) -> str:
    """Return ``missing``, ``unreadable``, or ``ok``.

    ``ok`` means a file exists and an available decoder can open it. When
    neither ``ffprobe`` nor macOS ``afinfo`` is installed, a file with an
    MPEG frame sync is treated as readable and the caller should say that
    full decode was not available.
    """
    if _is_blank(path):
        return "missing"
    file_path = Path(str(path))
    try:
        stat = file_path.stat()
    except OSError:
        return "missing"
    if not file_path.is_file() or stat.st_size <= 0:
        return "unreadable"
    if not _mp3_has_frame_sync(file_path):
        return "unreadable"
    probe = _probe_command()
    if probe is None:
        return "ok"
    accepted = _decoder_cached(str(file_path.resolve()), stat.st_mtime_ns, stat.st_size, probe)
    return "ok" if accepted else "unreadable"


def decoder_name() -> str | None:
    """The decoder used for readability checks, or None if only the frame header is checked."""
    probe = _probe_command()
    if probe is None:
        return None
    return probe[0]


def exclude_unplayable(songs: pd.DataFrame) -> pd.DataFrame:
    """Drop songs that cannot be played when this table has any audio paths.

    A table with no audio paths is the ratings-only catalog. Those songs stay,
    so playlist logic can be tested before the MP3s are downloaded. They are
    not playback-ready. Once any path is present, blank, missing, and
    undecodable rows are removed so they cannot be recommended.
    """
    if "audio_path" not in songs.columns:
        return songs.reset_index(drop=True)
    paths = list(songs["audio_path"])
    if all(_is_blank(p) for p in paths):
        return songs.reset_index(drop=True)
    keep = [not _is_blank(p) and classify_audio(p) == "ok" for p in paths]
    return songs.loc[keep].reset_index(drop=True)


def clean_songs(
    songs: pd.DataFrame,
    audio,
    spread_quantile: float = SPREAD_QUANTILE,
    require_audio: bool | None = None,
    unreadable_ids=None,
):
    """Apply the data-plan cleaning rules.

    Songs missing a rating, rated outside 1 to 9, or above the rater-spread
    quantile are dropped. Audio is required only when MP3 files were found
    (``require_audio=True``). Unreadable ids are dropped separately from songs
    that have no file. The spread cutoff is computed after the audio filter,
    which is the original notebook rule: once audio is present the cutoff can
    move, because it is a quantile of the songs that remain.
    """
    report = {"Raw songs": len(songs)}
    df = songs.copy()

    has_rating = df[["valence", "arousal", "valence_std", "arousal_std"]].notna().all(axis=1)
    report["Dropped, missing rating"] = int((~has_rating).sum())
    df = df.loc[has_rating]

    in_range = df["valence"].between(SCALE_MIN, SCALE_MAX) & df["arousal"].between(SCALE_MIN, SCALE_MAX)
    report["Dropped, rating out of 1-9"] = int((~in_range).sum())
    df = df.loc[in_range]

    if require_audio is None:
        require_audio = bool(audio)

    if require_audio:
        playable_ids = set(audio)
        bad_ids = {int(i) for i in (unreadable_ids or ())}
        missing_audio = ~df["song_id"].isin(playable_ids | bad_ids)
        unreadable = df["song_id"].isin(bad_ids)
        report["Dropped, missing audio"] = int(missing_audio.sum())
        report["Dropped, unreadable audio"] = int(unreadable.sum())
        df = df.loc[df["song_id"].isin(playable_ids)].copy()
        df["audio_path"] = df["song_id"].map(audio).map(lambda p: None if p is None else str(p))
    else:
        df = df.assign(audio_path=None)
        report["Dropped, missing audio"] = "skipped (no MP3 files found)"
        report["Dropped, unreadable audio"] = "skipped (no MP3 files found)"

    if df.empty:
        df = df.assign(spread=pd.Series(dtype=float))
        report["Spread cutoff"] = None
        report["Dropped, high rater spread (no songs left)"] = 0
        report["Clean songs"] = 0
        return df.reset_index(drop=True), report

    df = df.assign(spread=(df["valence_std"] + df["arousal_std"]) / 2)
    cutoff = float(df["spread"].quantile(spread_quantile))
    keep = df["spread"] <= cutoff
    report["Spread cutoff"] = cutoff
    report[f"Dropped, high rater spread (> {cutoff:.3f})"] = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)
    report["Clean songs"] = len(df)
    return df, report


def _is_blank(path) -> bool:
    if path is None:
        return True
    try:
        if pd.isna(path):
            return True
    except (TypeError, ValueError):
        pass
    return str(path).strip().lower() in {"", "none", "nan"}


def _mp3_has_frame_sync(path: Path) -> bool:
    """True if the file starts like an MP3 (ID3 tag and/or a frame sync)."""
    with path.open("rb") as handle:
        data = handle.read(262144)
    start = 0
    if data.startswith(b"ID3") and len(data) >= 10:
        # Synchsafe integer size of the ID3v2 tag, then look past it.
        size = 0
        for byte in data[6:10]:
            size = (size << 7) | (byte & 0x7F)
        start = min(len(data), 10 + size)
    limit = min(len(data) - 1, start + 65536)
    for i in range(start, limit):
        if data[i] == 0xFF and data[i + 1] & 0xE0 == 0xE0:
            return True
    return False


def _probe_command():
    if shutil.which("ffprobe"):
        return (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
        )
    if shutil.which("afinfo"):
        return ("afinfo",)
    return None


@lru_cache(maxsize=8192)
def _decoder_cached(path_str: str, mtime_ns: int, size: int, probe: tuple) -> bool:
    del mtime_ns, size  # part of the cache key so a replaced file is rechecked
    cmd = (*probe, path_str)
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    if probe[0] == "ffprobe":
        return b"audio" in completed.stdout
    return True
