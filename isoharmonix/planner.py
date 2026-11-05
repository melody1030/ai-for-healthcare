"""Iso-guided and direct-to-target playlists, and the song each waypoint gets.

Waypoints follow the study rules: five iso steps evenly spaced from the
current mood to the target, and five direct steps that all sit on the target.
Songs are DEAM (valence, arousal) ratings. Distance is Euclidean.

``build_session(..., allocation="joint")`` assigns the ten songs together so
neither playlist is penalized for being built second. ``allocation="iso_first"``
reproduces the original notebook rule. Which default the pilot uses still
needs team approval; joint is the default on this branch because iso-first
changes match quality when the same songs are eligible for both playlists.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from isoharmonix.data import classify_audio, exclude_unplayable
from isoharmonix.rules import (
    ALLOCATIONS,
    DIRECT,
    DISTANCE_METRIC,
    GAP_WARNING,
    ISO,
    N_STEPS,
    SCALE_MAX,
    SCALE_MIN,
)

if DISTANCE_METRIC != "euclidean":
    raise RuntimeError("Song retrieval implements Euclidean distance only.")

SESSION_COLUMNS = [
    "condition",
    "step",
    "n_steps",
    "target_valence",
    "target_arousal",
    "song_id",
    "song_valence",
    "song_arousal",
    "distance",
    "gap_warning",
    "audio_path",
    "why",
]


def _check_axis(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number between {SCALE_MIN:g} and {SCALE_MAX:g}")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number between {SCALE_MIN:g} and {SCALE_MAX:g}")
    number = float(value)
    if not math.isfinite(number) or not SCALE_MIN <= number <= SCALE_MAX:
        raise ValueError(f"{name} must be between {SCALE_MIN:g} and {SCALE_MAX:g}")
    return number


def _check_steps(n_steps: int) -> int:
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 2:
        raise ValueError("n_steps must be an integer >= 2")
    return int(n_steps)


class Mood:
    """A valence-arousal point on the 1-9 slider scale."""

    def __init__(self, valence, arousal):
        self.valence = _check_axis("valence", valence)
        self.arousal = _check_axis("arousal", arousal)

    def as_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal], dtype=float)

    def __eq__(self, other):
        return (
            isinstance(other, Mood)
            and self.valence == other.valence
            and self.arousal == other.arousal
        )

    def __repr__(self):
        return f"Mood({self.valence:g}, {self.arousal:g})"


def iso_path(current: Mood, target: Mood, n_steps: int = N_STEPS) -> np.ndarray:
    """Evenly spaced waypoints. The first is the current mood and the last is the target."""
    n_steps = _check_steps(n_steps)
    weights = np.linspace(0.0, 1.0, n_steps)[:, None]
    points = (1.0 - weights) * current.as_array() + weights * target.as_array()
    points[0] = current.as_array()
    points[-1] = target.as_array()
    return points


def direct_path(target: Mood, n_steps: int = N_STEPS) -> np.ndarray:
    """Every waypoint is the target mood."""
    n_steps = _check_steps(n_steps)
    return np.tile(target.as_array(), (n_steps, 1))


def plan_path(condition: str, current: Mood, target: Mood, n_steps: int = N_STEPS) -> np.ndarray:
    if condition == ISO:
        return iso_path(current, target, n_steps)
    if condition == DIRECT:
        return direct_path(target, n_steps)
    raise ValueError(f"condition must be {ISO!r} or {DIRECT!r}, got {condition!r}")


class SongRetriever:
    """Closest unused DEAM song for a valence-arousal point.

    Ties are broken by the smaller ``song_id``, so the same catalog always
    yields the same song. Songs whose ``audio_path`` is set but missing or
    undecodable are not eligible. A catalog with no audio paths is kept, for
    ratings-only checks; those rows are not playback-ready.
    """

    def __init__(self, songs: pd.DataFrame, *, check_audio: bool = True):
        required = {"song_id", "valence", "arousal"}
        missing = required - set(songs.columns)
        if missing:
            raise ValueError(f"songs is missing columns: {sorted(missing)}")
        frame = songs.reset_index(drop=True).copy()
        if "audio_path" not in frame.columns:
            frame["audio_path"] = None
        if check_audio:
            frame = exclude_unplayable(frame)
        if frame.empty:
            raise ValueError("No eligible songs to retrieve from.")
        if frame["song_id"].duplicated().any():
            raise ValueError("song_id values must be unique.")
        self.songs = frame.reset_index(drop=True)
        self.ids = self.songs["song_id"].to_numpy(dtype=int)
        self.coords = self.songs[["valence", "arousal"]].to_numpy(dtype=float)

    def closest_unused(self, point, used: set) -> tuple[int, float]:
        """Row index and Euclidean distance of the closest song whose id is not in ``used``."""
        point = np.asarray(point, dtype=float)
        distances = np.linalg.norm(self.coords - point, axis=1)
        if used:
            distances[np.isin(self.ids, list(used))] = np.inf
        # Primary key is distance, secondary key is song id.
        order = np.lexsort((self.ids, distances))
        row = int(order[0])
        distance = float(distances[row])
        if not math.isfinite(distance):
            raise RuntimeError(
                f"Not enough eligible songs: {len(self.songs)} available and all are already used."
            )
        return row, distance


def why_text(row: dict, allocation: str) -> str:
    if allocation == "joint":
        choice = "It was assigned across both playlists so that no song is repeated"
    else:
        choice = "It was the closest unused song to the target"
    text = (
        f"Song {row['step']} of {row['n_steps']}. "
        f"Target for this step: valence {row['target_valence']:.1f}, "
        f"arousal {row['target_arousal']:.1f}. "
        f"This song is rated valence {row['song_valence']:.1f}, "
        f"arousal {row['song_arousal']:.1f} by DEAM listeners. "
        f"{choice} (distance {row['distance']:.2f})."
    )
    if row["gap_warning"]:
        text += " Few songs exist near this target, so the match is less close."
    return text


def _audio_value(song) -> str | None:
    path = song["audio_path"] if "audio_path" in song.index else None
    if path is None:
        return None
    try:
        if pd.isna(path):
            return None
    except (TypeError, ValueError):
        pass
    text = str(path).strip()
    if text.lower() in {"", "none", "nan"}:
        return None
    return text


def _make_row(condition, step, n_steps, point, song, distance, allocation) -> dict:
    row = {
        "condition": condition,
        "step": int(step),
        "n_steps": int(n_steps),
        "target_valence": round(float(point[0]), 3),
        "target_arousal": round(float(point[1]), 3),
        "song_id": int(song["song_id"]),
        "song_valence": float(song["valence"]),
        "song_arousal": float(song["arousal"]),
        "distance": round(float(distance), 3),
        "gap_warning": bool(distance > GAP_WARNING),
        "audio_path": _audio_value(song),
    }
    row["why"] = why_text(row, allocation)
    return row


def build_playlist(
    retriever: SongRetriever,
    condition: str,
    current: Mood,
    target: Mood,
    n_steps: int = N_STEPS,
    used: set | None = None,
    allocation: str = "iso_first",
) -> pd.DataFrame:
    """One playlist. ``used`` is updated in place so a second playlist can avoid those songs."""
    n_steps = _check_steps(n_steps)
    if used is None:
        used = set()
    path = plan_path(condition, current, target, n_steps)
    rows = []
    for step, point in enumerate(path, start=1):
        index, distance = retriever.closest_unused(point, used)
        song = retriever.songs.iloc[index]
        used.add(int(song["song_id"]))
        rows.append(_make_row(condition, step, n_steps, point, song, distance, allocation))
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)


def build_session(
    retriever: SongRetriever,
    current: Mood,
    target: Mood,
    n_steps: int = N_STEPS,
    allocation: str = "joint",
) -> pd.DataFrame:
    """Both playlists for one participant. No song id appears twice.

    ``allocation="joint"`` minimizes the sum of Euclidean distances over the
    ten waypoints and does not depend on playlist order. ``"iso_first"`` is
    the original rule. ``"direct_first"`` reverses that rule so the order
    effect can be measured. Rows are always returned iso-then-direct.
    """
    if allocation not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {ALLOCATIONS}, got {allocation!r}")
    n_steps = _check_steps(n_steps)
    if allocation == "joint":
        return _build_joint(retriever, current, target, n_steps)
    used: set = set()
    if allocation == "iso_first":
        iso = build_playlist(retriever, ISO, current, target, n_steps, used, allocation)
        direct = build_playlist(retriever, DIRECT, current, target, n_steps, used, allocation)
    else:
        direct = build_playlist(retriever, DIRECT, current, target, n_steps, used, allocation)
        iso = build_playlist(retriever, ISO, current, target, n_steps, used, allocation)
    return pd.concat([iso, direct], ignore_index=True)


def match_quality(session: pd.DataFrame) -> pd.DataFrame:
    """Mean and max waypoint-to-song distance for each condition."""
    return session.groupby("condition", sort=False)["distance"].agg(["mean", "max"]).round(3)


def session_audio_report(session: pd.DataFrame) -> dict:
    """How many recommended rows point at a file the decoder can open."""
    statuses = [classify_audio(path) for path in session["audio_path"]]
    return {
        "rows": len(statuses),
        "ok": statuses.count("ok"),
        "missing": statuses.count("missing"),
        "unreadable": statuses.count("unreadable"),
    }


def _build_joint(retriever: SongRetriever, current: Mood, target: Mood, n_steps: int) -> pd.DataFrame:
    slots = []
    for condition, path in (
        (ISO, iso_path(current, target, n_steps)),
        (DIRECT, direct_path(target, n_steps)),
    ):
        for step, point in enumerate(path, start=1):
            slots.append({"condition": condition, "step": step, "point": np.asarray(point, dtype=float)})
    assignment = _assign_slots(retriever.coords, retriever.ids, slots)
    rows = []
    for slot, (row_index, distance) in zip(slots, assignment):
        song = retriever.songs.iloc[row_index]
        rows.append(
            _make_row(slot["condition"], slot["step"], n_steps, slot["point"], song, distance, "joint")
        )
    frame = pd.DataFrame(rows, columns=SESSION_COLUMNS)
    # Iso block, then direct block, each in step order. The solver's internal
    # slot order must not leak into the table Tiffany's app reads.
    return frame.sort_values(["condition", "step"], key=_condition_then_step).reset_index(drop=True)


def _condition_then_step(column: pd.Series):
    if column.name == "condition":
        return column.map({ISO: 0, DIRECT: 1})
    return column


def _target_key(point) -> tuple[float, float]:
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def _assign_slots(coords: np.ndarray, ids: np.ndarray, slots: list[dict]) -> list[tuple[int, float]]:
    """Minimum-cost unique songs for each slot, then a fair split of tied targets.

    The Hungarian step minimizes total Euclidean distance. It is solved on a
    canonical slot order so the chosen song set does not depend on whether
    the iso slots were listed first. Among steps that ask for the same point,
    that total is unchanged by swapping songs. Those songs are then dealt by
    how many steps each condition still has to fill, not by which playlist
    was listed first.
    """
    n_slots = len(slots)
    n_songs = int(len(ids))
    if n_songs < n_slots:
        raise RuntimeError(
            f"Not enough eligible songs for a session: need {n_slots}, have {n_songs}."
        )

    order = sorted(
        range(n_slots),
        key=lambda i: (_target_key(slots[i]["point"]), slots[i]["condition"], slots[i]["step"]),
    )
    ordered_slots = [slots[i] for i in order]
    points = np.vstack([slot["point"] for slot in ordered_slots])
    cost = np.linalg.norm(coords[None, :, :] - points[:, None, :], axis=2)
    # Smaller song ids win exact ties. The penalty is far below the 0.01
    # precision of DEAM ratings, so it cannot overturn a closer song.
    perturbed = cost + ids.astype(np.float64) * 1e-12
    row_ind, col_ind = linear_sum_assignment(perturbed)
    assigned_ordered = np.empty(n_slots, dtype=int)
    assigned_ordered[row_ind] = col_ind

    assigned = np.empty(n_slots, dtype=int)
    for position, slot_index in enumerate(order):
        assigned[slot_index] = assigned_ordered[position]

    groups: dict[tuple[float, float], list[int]] = {}
    for index, slot in enumerate(slots):
        groups.setdefault(_target_key(slot["point"]), []).append(index)

    for slot_indexes in groups.values():
        if len(slot_indexes) == 1:
            continue
        target = slots[slot_indexes[0]]["point"]
        song_rows = [int(assigned[i]) for i in slot_indexes]
        for slot_index, song_row in _balance_group(
            slot_indexes, song_rows, slots, coords, ids, target
        ).items():
            assigned[slot_index] = song_row

    result = []
    for index, slot in enumerate(slots):
        song_row = int(assigned[index])
        distance = float(np.linalg.norm(coords[song_row] - slot["point"]))
        result.append((song_row, distance))
    return result


def _balance_group(slot_indexes, song_rows, slots, coords, ids, target) -> dict[int, int]:
    """Deal songs for one shared target without favoring playlist order.

    Songs are sorted from closest to farthest, then by song id. Each next
    song goes to the condition with the largest fraction of its steps still
    empty, which is a weighted round-robin: a condition that needs five songs
    at this point receives five, and a condition that needs one receives one,
    spread through the quality list instead of taking the closest block.

    Ties break by more unfilled steps, then by not taking two songs in a row,
    then by condition name. The name is only a last resort, so the result does
    not depend on row order. When both conditions need the same number of
    songs, that last tie gives ``direct`` the single closest song and the
    later songs alternate. Team review can flip that name tie before the pilot.
    """
    ranked = sorted(
        song_rows,
        key=lambda row: (float(np.linalg.norm(coords[row] - target)), int(ids[row])),
    )
    by_condition: dict[str, list[int]] = {}
    for index in slot_indexes:
        by_condition.setdefault(slots[index]["condition"], []).append(index)
    for indexes in by_condition.values():
        indexes.sort(key=lambda index: slots[index]["step"])

    quotas = {condition: len(indexes) for condition, indexes in by_condition.items()}
    received = {condition: 0 for condition in quotas}
    cursor = {condition: 0 for condition in quotas}
    assignment = {}
    last = None
    for song_row in ranked:
        open_conditions = [c for c in quotas if received[c] < quotas[c]]

        def priority(condition, _last=last, _received=received, _quotas=quotas):
            remaining = _quotas[condition] - _received[condition]
            # min() selects the largest remaining fraction, then the larger count.
            return (
                -remaining / _quotas[condition],
                -remaining,
                condition == _last,
                condition,
            )

        condition = min(open_conditions, key=priority)
        slot_index = by_condition[condition][cursor[condition]]
        cursor[condition] += 1
        assignment[slot_index] = song_row
        received[condition] += 1
        last = condition
    return assignment
