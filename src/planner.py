"""Mood paths, nearest-song retrieval, and session construction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


SCALE_MIN, SCALE_MAX = 1.0, 9.0
N_STEPS = 5
DISTANCE_METRIC = "euclidean"
GAP_WARNING = 1.0
ISO, DIRECT = "iso", "direct"


@dataclass(frozen=True)
class Mood:
    """A valence-arousal point on the project's 1–9 mood scale."""

    valence: float
    arousal: float

    def __post_init__(self) -> None:
        for name in ("valence", "arousal"):
            if not SCALE_MIN <= getattr(self, name) <= SCALE_MAX:
                raise ValueError(f"{name} must be between {SCALE_MIN} and {SCALE_MAX}")

    def as_array(self) -> np.ndarray:
        """Return the mood as a numeric ``[valence, arousal]`` array."""
        return np.array([self.valence, self.arousal], dtype=float)


def iso_path(current: Mood, target: Mood, n_steps: int = N_STEPS) -> np.ndarray:
    """Return evenly spaced mood targets from current mood to target mood."""
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    return (1 - t) * current.as_array() + t * target.as_array()


def direct_path(target: Mood, n_steps: int = N_STEPS) -> np.ndarray:
    """Return ``n_steps`` copies of the target mood."""
    return np.tile(target.as_array(), (n_steps, 1))


def plan_path(condition: str, current: Mood, target: Mood, n_steps: int = N_STEPS) -> np.ndarray:
    """Build the path for either the Iso or Direct condition."""
    paths = {
        ISO: lambda: iso_path(current, target, n_steps),
        DIRECT: lambda: direct_path(target, n_steps),
    }
    return paths[condition]()


class SongRetriever:
    """Retrieve nearest unused songs in valence-arousal space."""

    def __init__(self, songs: pd.DataFrame) -> None:
        self.songs = songs.reset_index(drop=True)
        self.ids = self.songs["song_id"].to_numpy()
        self.nn = NearestNeighbors(metric=DISTANCE_METRIC).fit(
            self.songs[["valence", "arousal"]].to_numpy()
        )

    def closest_unused(self, point: np.ndarray, used: set[int]) -> tuple[int, float]:
        """Return the nearest unused row index and its Euclidean distance."""
        k = min(len(self.songs), len(used) + 1)
        while True:
            dist, idx = self.nn.kneighbors(np.asarray(point, float)[None, :], n_neighbors=k)
            for distance, index in zip(dist[0], idx[0]):
                if int(self.ids[index]) not in used:
                    return int(index), float(distance)
            if k == len(self.songs):
                raise RuntimeError("Every song has already been used.")
            k = min(len(self.songs), 2 * k)


def nearest_gap(songs: pd.DataFrame, points: np.ndarray) -> np.ndarray:
    """Return each point's distance to its nearest song."""
    nn = NearestNeighbors(n_neighbors=1, metric=DISTANCE_METRIC).fit(
        songs[["valence", "arousal"]].to_numpy()
    )
    distances, _ = nn.kneighbors(np.asarray(points, dtype=float))
    return distances[:, 0]


def why_text(row: dict) -> str:
    """Create the explainability text shown for a selected song."""
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
    n_steps: int = N_STEPS,
    used: set[int] | None = None,
    fixed_song_ids: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Build one playlist, optionally assigning fixed songs to specific steps."""
    used = set() if used is None else used
    fixed_song_ids = {} if fixed_song_ids is None else fixed_song_ids
    fixed_ids = set(fixed_song_ids.values())
    path = plan_path(condition, current, target, n_steps)
    rows = []
    for step, point in enumerate(path, start=1):
        if step in fixed_song_ids:
            song_id = int(fixed_song_ids[step])
            matches = np.flatnonzero(retriever.ids == song_id)
            if len(matches) != 1:
                raise ValueError(f"Fixed song ID {song_id} is not in the retriever.")
            index = int(matches[0])
            distance = float(
                np.linalg.norm(
                    retriever.songs.loc[index, ["valence", "arousal"]].to_numpy(float) - point
                )
            )
        else:
            index, distance = retriever.closest_unused(point, used | fixed_ids)
        song = retriever.songs.iloc[index]
        used.add(int(song["song_id"]))
        row = {
            "condition": condition,
            "step": step,
            "n_steps": n_steps,
            "target_valence": round(point[0], 3),
            "target_arousal": round(point[1], 3),
            "song_id": int(song["song_id"]),
            "song_valence": song["valence"],
            "song_arousal": song["arousal"],
            "distance": round(distance, 3),
            "gap_warning": distance > GAP_WARNING,
            "audio_path": song["audio_path"],
        }
        row["why"] = why_text(row)
        rows.append(row)
    return pd.DataFrame(rows)


def build_session(
    retriever: SongRetriever,
    current: Mood,
    target: Mood,
    n_steps: int = N_STEPS,
) -> pd.DataFrame:
    """Build disjoint Iso and Direct playlists with reserved target songs."""
    target_point = target.as_array()[None, :]
    n_candidates = min(6, len(retriever.songs))
    distances, indices = retriever.nn.kneighbors(target_point, n_neighbors=n_candidates)
    if n_candidates < 6:
        raise RuntimeError("At least six songs are required for the two target allocations.")
    candidate_ids = [int(retriever.ids[index]) for index in indices[0]]
    candidate_distances = distances[0]
    endpoint_pos = min(
        range(6),
        key=lambda index: abs(
            candidate_distances[index] - np.mean(np.delete(candidate_distances, index))
        ),
    )
    iso_endpoint_id = candidate_ids[endpoint_pos]
    direct_ids = [song_id for index, song_id in enumerate(candidate_ids) if index != endpoint_pos]
    reserved_ids = set(candidate_ids)

    iso = build_playlist(
        retriever,
        ISO,
        current,
        target,
        n_steps,
        used=reserved_ids.copy(),
        fixed_song_ids={n_steps: iso_endpoint_id},
    )
    direct_used = set(retriever.ids) - set(direct_ids)
    direct = build_playlist(
        retriever,
        DIRECT,
        current,
        target,
        n_steps,
        used=direct_used,
        fixed_song_ids={
            step: song_id for step, song_id in enumerate(direct_ids, start=1)
        },
    )
    assert set(iso["song_id"]).isdisjoint(set(direct["song_id"]))
    return pd.concat([iso, direct], ignore_index=True)


def match_quality(session: pd.DataFrame) -> pd.DataFrame:
    """Return mean and maximum selected-song distance by condition."""
    return session.groupby("condition", sort=False)["distance"].agg(["mean", "max"]).round(3)
