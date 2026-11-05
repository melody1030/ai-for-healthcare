"""Path planner and song retrieval (timeline task: "Path planner + song retrieval").

Step 2a, iso-guided path:  N evenly spaced points from the current mood to
                           the target mood. Song 1 matches the current mood,
                           song N reaches the target.
Step 2b, direct-to-target: all N points sit at the target mood (the control).
Step 3,  song retrieval:   for each point, pick the closest UNUSED song by
                           distance in valence-arousal space.

Every chosen song keeps the numbers needed for the "why this song" card:
its step, the step's target, the song's own DEAM ratings and the distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config

ISO = "iso"
DIRECT = "direct"


@dataclass(frozen=True)
class Mood:
    """A point in valence-arousal space, on the 1-9 slider scale."""

    valence: float
    arousal: float

    def __post_init__(self):
        for name in ("valence", "arousal"):
            v = getattr(self, name)
            if not (config.SCALE_MIN <= v <= config.SCALE_MAX):
                raise ValueError(
                    f"{name}={v} is outside {config.SCALE_MIN}-{config.SCALE_MAX}"
                )

    def as_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal], dtype=float)


def iso_path(current: Mood, target: Mood, n_steps: int = config.N_STEPS) -> np.ndarray:
    """Evenly spaced points from ``current`` to ``target`` (both included)."""
    if n_steps < 2:
        raise ValueError("An iso path needs at least 2 steps.")
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    return (1 - t) * current.as_array() + t * target.as_array()


def direct_path(target: Mood, n_steps: int = config.N_STEPS) -> np.ndarray:
    """All points at the target mood (the direct-to-target control)."""
    if n_steps < 1:
        raise ValueError("A playlist needs at least 1 step.")
    return np.tile(target.as_array(), (n_steps, 1))


def plan_path(
    condition: str, current: Mood, target: Mood, n_steps: int = config.N_STEPS
) -> np.ndarray:
    if condition == ISO:
        return iso_path(current, target, n_steps)
    if condition == DIRECT:
        return direct_path(target, n_steps)
    raise ValueError(f"Unknown condition '{condition}', use '{ISO}' or '{DIRECT}'.")


class SongRetriever:
    """Closest-unused-song lookup over the cleaned DEAM table."""

    def __init__(self, songs: pd.DataFrame):
        if songs.empty:
            raise ValueError("Song table is empty.")
        self.songs = songs.reset_index(drop=True)
        self._xy = self.songs[["valence", "arousal"]].to_numpy(dtype=float)
        self._nn = NearestNeighbors(metric=config.DISTANCE_METRIC).fit(self._xy)

    def closest_unused(self, point: np.ndarray, used: set[int]) -> tuple[int, float]:
        """Return (row index, distance) of the closest song whose id is not in ``used``.

        We ask for a few neighbours first and widen the search only when they
        are all used, so this stays fast even late in a session.
        """
        n_total = len(self.songs)
        k = min(n_total, len(used) + 1)
        ids = self.songs["song_id"].to_numpy()
        while True:
            dist, idx = self._nn.kneighbors(np.asarray(point, float)[None, :], n_neighbors=k)
            for d, i in zip(dist[0], idx[0]):
                if int(ids[i]) not in used:
                    return int(i), float(d)
            if k == n_total:
                raise RuntimeError("Every song has already been used.")
            k = min(n_total, k * 2)

    def retrieve(
        self, path: np.ndarray, used: set[int] | None = None, condition: str = ""
    ) -> pd.DataFrame:
        """Pick one song per path point, never repeating a song.

        ``used`` is updated in place, so passing the same set to several calls
        keeps songs unique across playlists.
        """
        used = set() if used is None else used
        rows = []
        n = len(path)
        for step, point in enumerate(path, start=1):
            i, d = self.closest_unused(point, used)
            song = self.songs.iloc[i]
            used.add(int(song["song_id"]))
            rows.append(_row(condition, step, n, point, song, d))
        return pd.DataFrame(rows)


def _row(condition: str, step: int, n: int, point: np.ndarray, song: pd.Series, dist: float) -> dict:
    row = {
        "condition": condition,
        "step": step,
        "n_steps": n,
        "target_valence": round(float(point[0]), 3),
        "target_arousal": round(float(point[1]), 3),
        "song_id": int(song["song_id"]),
        "song_valence": float(song["valence"]),
        "song_arousal": float(song["arousal"]),
        "distance": round(dist, 3),
        "gap_warning": dist > config.GAP_WARNING,
        "audio_path": song.get("audio_path", None),
    }
    row["why"] = why_text(row)
    return row


def why_text(row: dict) -> str:
    """Plain-language "why this song" text for the app's why-card."""
    text = (
        f"Song {row['step']} of {row['n_steps']}. "
        f"Target for this step: valence {row['target_valence']:.1f}, "
        f"arousal {row['target_arousal']:.1f}. "
        f"This song is rated valence {row['song_valence']:.1f}, "
        f"arousal {row['song_arousal']:.1f} by DEAM listeners. "
        f"It was the closest unused song to the target (distance {row['distance']:.2f})."
    )
    if row["gap_warning"]:
        text += " Few songs exist near this target, so the match is less close."
    return text


def build_playlist(
    retriever: SongRetriever,
    condition: str,
    current: Mood,
    target: Mood,
    n_steps: int = config.N_STEPS,
    used: set[int] | None = None,
) -> pd.DataFrame:
    path = plan_path(condition, current, target, n_steps)
    return retriever.retrieve(path, used=used, condition=condition)


def build_session(
    retriever: SongRetriever,
    current: Mood,
    target: Mood,
    n_steps: int = config.N_STEPS,
) -> pd.DataFrame:
    """Both playlists for one participant, with no song in both.

    Rule: the iso-guided playlist is built first, then the direct playlist is
    built from the songs that are left. Only the iso playlist's last song sits
    at the target, so the direct playlist loses at most one song near the
    target. (Taking turns would let the direct playlist use up the songs near
    the target before the iso playlist reaches its last step.)
    Use ``match_quality`` to check that both playlists match equally well.
    """
    used: set[int] = set()
    iso = build_playlist(retriever, ISO, current, target, n_steps, used)
    direct = build_playlist(retriever, DIRECT, current, target, n_steps, used)
    return pd.concat([iso, direct], ignore_index=True)


def match_quality(session: pd.DataFrame) -> pd.DataFrame:
    """Mean and max distance between target points and chosen songs, per condition.

    A large difference between conditions would be a confound, so this is
    worth reporting with the study results.
    """
    return session.groupby("condition", sort=False)["distance"].agg(["mean", "max"]).round(3)
