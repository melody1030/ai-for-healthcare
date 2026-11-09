"""Minimal Streamlit app for the IsoHarmonix within-subject pilot."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

from isoharmonix.data import (  # noqa: E402
    clean_songs,
    load_annotations,
    resolve_deam_dir,
)
from isoharmonix.planner import (  # noqa: E402
    SESSION_COLUMNS,
    Mood,
    SongRetriever,
    build_session,
)
from isoharmonix.rules import (  # noqa: E402
    GAP_WARNING,
    DIRECT,
    ISO,
)
from isoharmonix.study_data import PreparedAudioError, load_prepared_audio  # noqa: E402


PILOT_MODE = True
STUDY_DATA_DIR = ROOT / "study" / "data"
OUTPUT_DIR = STUDY_DATA_DIR / "pilot" if PILOT_MODE else STUDY_DATA_DIR
COUNTERBALANCE_PATH = ROOT / "study" / (
    "counterbalancing_pilot.csv" if PILOT_MODE else "counterbalancing.csv"
)
OUTPUT_FILES = {
    "participants": OUTPUT_DIR / "participants.csv",
    "ratings": OUTPUT_DIR / "ratings.csv",
    "songs_played": OUTPUT_DIR / "songs_played.csv",
    "session_log": OUTPUT_DIR / "session_log.csv",
}
SESSION_LOG_COLUMNS = [
    "participant_id", "session_date", "researcher", "start_time",
    "end_time", "app_problems", "notes",
]
CONSENT_TEXT = """**IsoHarmonix: music and everyday mood**

We are students in the AI for Healthcare course at Cornell Tech. We are testing
two ways of ordering songs in a playlist, to see how they affect everyday mood.

**What you will do.** You will listen to two short playlists (about 4 minutes each)
through headphones and answer short questions about your mood before and after each
one. The session takes about 30 minutes.

**Risks.** The risks are very low. The songs are ordinary music clips. If anything
makes you uncomfortable, you can stop at any time.

**This is not treatment.** This study does not diagnose or treat any health condition.

**Your data.** We do not record your name with your answers. We use a participant
number instead. Only the project team can see the data, and we only report combined,
anonymous results. You can ask us to delete your data until 17 November 2026.

**Your choice.** Taking part is voluntary. You can skip any question or stop at any
time, with no problem at all.

"""
MOOD_ANCHORS = (
    "Pleasant: 1 = very unhappy, sad; 5 = neutral; 9 = very happy, pleased",
    "Energy: 1 = very calm, relaxed, sleepy; 5 = neutral; "
    "9 = very excited, alert, tense",
)
MOOD_LABELS = {
    "target_pleasant": "Pleasant feeling you would like",
    "target_energy": "Energy you would like",
    "current_pleasant": "How pleasant do you feel?",
    "current_energy": "How much energy do you feel?",
}


def apply_styles() -> None:
    """Apply restrained styling for the participant-facing study interface."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8f6;
        }
        .block-container {
            max-width: 800px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        .study-card {
            background: #ffffff;
            border: 1px solid #e3e8e4;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(34, 54, 44, 0.06);
            padding: 1.25rem 1.4rem;
            margin: 0.8rem 0 1.2rem;
        }
        .study-progress {
            color: #64736a;
            font-size: 0.82rem;
            letter-spacing: 0.01em;
            margin-bottom: 1.4rem;
        }
        div.stButton > button[kind="primary"] {
            background: #2f6f6a;
            border-color: #2f6f6a;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #255a56;
            border-color: #255a56;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_progress(stage: str) -> None:
    """Show study progress without revealing the randomized conditions."""
    stages = {
        "setup": "Setup",
        "playlist_a": "Playlist A",
        "ratings_a": "Feedback",
        "break": "Break",
        "playlist_b": "Playlist B",
        "ratings_b": "Final",
    }
    labels = list(stages.values())
    current = {
        "setup": 0,
        "playlist": 1 if st.session_state.get("playlist_index", 0) == 0 else 4,
        "ratings": 2 if st.session_state.get("playlist_index", 0) == 0 else 5,
        "break": 3,
        "second_pre": 4,
        "final": 5,
        "complete": 5,
    }.get(stage, 0)
    parts = [
        f"<strong>{label}</strong>" if index == current else label
        for index, label in enumerate(labels)
    ]
    st.markdown(
        f'<div class="study-progress">{"  ·  ".join(parts)}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_study_songs() -> pd.DataFrame:
    """Load cleaned DEAM songs with local audio paths for the study."""
    data_dir = resolve_deam_dir(ROOT)
    raw_songs = load_annotations(data_dir)
    audio = load_prepared_audio(ROOT, data_dir)
    songs, _ = clean_songs(
        raw_songs,
        audio.playable,
        require_audio=audio.files_found > 0,
        unreadable_ids=audio.unreadable_ids,
    )
    if songs.empty:
        raise ValueError("No usable DEAM songs were found after cleaning.")
    return songs


@st.cache_resource(show_spinner=False)
def load_retriever(songs: pd.DataFrame) -> SongRetriever:
    """Build the nearest-song retriever once for the app process."""
    return SongRetriever(songs, check_audio=False)


@st.cache_data(show_spinner=False)
def supported_targets(songs: pd.DataFrame) -> tuple[tuple[int, int], ...]:
    """Return integer target moods with six songs within the gap threshold."""
    retriever = load_retriever(songs)
    targets = []
    for valence in range(1, 10):
        for arousal in range(1, 10):
            point = np.array([valence, arousal], dtype=float)
            distances = np.linalg.norm(retriever.coords - point, axis=1)
            if len(distances) >= 6 and np.partition(distances, 5)[5] <= GAP_WARNING:
                targets.append((valence, arousal))
    return tuple(targets)


def describe_target(target: tuple[int, int]) -> str:
    """Return a participant-friendly label while preserving the numeric target."""
    valence, arousal = target
    valence_words = {
        1: "very unhappy, sad",
        2: "unhappy",
        3: "somewhat unhappy",
        4: "slightly unhappy",
        5: "neutral",
        6: "slightly happy",
        7: "somewhat happy",
        8: "happy",
        9: "very happy, pleased",
    }
    arousal_words = {
        1: "very calm, relaxed, sleepy",
        2: "calm, relaxed",
        3: "somewhat calm",
        4: "slightly calm",
        5: "neutral energy",
        6: "slightly energized",
        7: "somewhat energized",
        8: "energized",
        9: "very excited, alert, tense",
    }
    return (
        f"{MOOD_LABELS['target_pleasant']}: {valence_words[valence]}; "
        f"{MOOD_LABELS['target_energy']}: {arousal_words[arousal]}"
    )


def initialize_state(
    participant_id: str,
    current: Mood,
    target: Mood,
    session: pd.DataFrame,
    assignment: pd.Series,
    background: dict,
) -> None:
    """Initialize the participant's one-time study state."""
    condition_order = [
        assignment["playlist_A_condition"],
        assignment["playlist_B_condition"],
    ]
    playlists = {
        condition: session.query("condition == @condition").reset_index(drop=True)
        for condition in (ISO, DIRECT)
    }
    st.session_state.update(
        {
            "started": True,
            "participant_id": participant_id,
            "assignment": assignment.to_dict(),
            "background": background,
            "age_18_plus": "Yes",
            "consent": "Yes",
            "initial_current": current,
            "target": target,
            "condition_order": condition_order,
            "playlists": playlists,
            "playlist_index": 0,
            "song_index": 0,
            "responses": {},
            "second_pre": None,
            "break_started_at": None,
            "skipped_clips": [],
            "last_completed_step": None,
            "resume_count": 0,
            "app_problems": [],
            "saved": False,
        }
    )


def reset_study() -> None:
    """Clear study state so another pilot participant can use the app."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def checkpoint_directory() -> Path:
    """Return the pilot or formal checkpoint directory without changing output mode."""
    mode_dir = STUDY_DATA_DIR / "pilot" if PILOT_MODE else STUDY_DATA_DIR
    return mode_dir / "checkpoints"


def checkpoint_path(participant_id: str) -> Path:
    if not participant_id or Path(participant_id).name != participant_id:
        raise ValueError("Invalid participant ID for checkpoint path.")
    return checkpoint_directory() / f"{participant_id}.json"


def checkpoint_payload(state: dict) -> dict:
    """Serialize the exact generated playlists and all state needed to resume."""
    return {
        "participant_id": state["participant_id"],
        "assignment": state["assignment"],
        "background": state["background"],
        "age_18_plus": state["age_18_plus"],
        "consent": state["consent"],
        "initial_current": [state["initial_current"].valence, state["initial_current"].arousal],
        "target": [state["target"].valence, state["target"].arousal],
        "condition_order": list(state["condition_order"]),
        "playlists": {
            condition: state["playlists"][condition].to_dict(orient="records")
            for condition in (ISO, DIRECT)
        },
        "playlist_index": int(state["playlist_index"]),
        "song_index": int(state["song_index"]),
        "responses": state["responses"],
        "second_pre": state["second_pre"],
        "break_started_at": state["break_started_at"],
        "started_at": state["started_at"],
        "stage": state["stage"],
        "last_completed_step": state.get("last_completed_step"),
        "skipped_clips": state.get("skipped_clips", []),
        "resume_count": int(state.get("resume_count", 0)),
        "app_problems": state.get("app_problems", []),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def state_from_checkpoint(payload: dict) -> dict:
    """Rehydrate saved playlists and progress without running retrieval again."""
    playlists = {
        condition: pd.DataFrame.from_records(
            payload["playlists"][condition], columns=SESSION_COLUMNS
        )
        for condition in (ISO, DIRECT)
    }
    return {
        "started": True,
        "eligible": True,
        "participant_id": payload["participant_id"],
        "assignment": payload["assignment"],
        "background": payload["background"],
        "age_18_plus": payload["age_18_plus"],
        "consent": payload["consent"],
        "initial_current": Mood(*payload["initial_current"]),
        "target": Mood(*payload["target"]),
        "condition_order": payload["condition_order"],
        "playlists": playlists,
        "playlist_index": int(payload["playlist_index"]),
        "song_index": int(payload["song_index"]),
        "responses": payload["responses"],
        "second_pre": payload["second_pre"],
        "break_started_at": payload["break_started_at"],
        "started_at": payload["started_at"],
        "stage": payload["stage"],
        "last_completed_step": payload.get("last_completed_step"),
        "skipped_clips": payload.get("skipped_clips", []),
        "resume_count": int(payload.get("resume_count", 0)),
        "app_problems": payload.get("app_problems", []),
        "saved": False,
    }


def read_checkpoint(participant_id: str) -> dict | None:
    path = checkpoint_path(participant_id)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as checkpoint_file:
        return json.load(checkpoint_file)


def resume_checkpoint(payload: dict, state: dict | None = None) -> None:
    state = st.session_state if state is None else state
    state.update(state_from_checkpoint(payload))
    state["resume_count"] += 1
    state.pop("resume_prompt", None)
    persist_checkpoint(state)


def persist_checkpoint(state: dict | None = None) -> None:
    state = st.session_state if state is None else state
    path = checkpoint_path(state["participant_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8"
    ) as temporary:
        json.dump(
            _json_safe(checkpoint_payload(state)),
            temporary,
            ensure_ascii=False,
            allow_nan=False,
        )
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    update_session_log(state)


def delete_checkpoint(participant_id: str) -> None:
    checkpoint_path(participant_id).unlink(missing_ok=True)


def participant_id_is_completed(participant_id: str) -> bool:
    """Return whether the participant already has a saved protocol record."""
    path = OUTPUT_FILES["participants"]
    if not path.exists():
        return False
    responses = pd.read_csv(path, usecols=["participant_id"], dtype=str)
    return participant_id in set(responses["participant_id"].dropna())


def advance_current_clip(state: dict, skip_reason: str | None = None) -> None:
    """Advance the existing playlist cursor, optionally recording a skipped clip."""
    playlist_index = state["playlist_index"]
    condition = state["condition_order"][playlist_index]
    playlist_label = "A" if playlist_index == 0 else "B"
    row = state["playlists"][condition].iloc[state["song_index"]]
    timestamp = datetime.now(timezone.utc).isoformat()
    completed = {
        "playlist": playlist_label,
        "condition": condition,
        "step": int(row["step"]),
        "song_id": int(row["song_id"]),
        "timestamp": timestamp,
    }
    state["last_completed_step"] = completed
    if skip_reason is not None:
        skipped = {**completed, "reason": skip_reason}
        state.setdefault("skipped_clips", []).append(skipped)
        state.setdefault("app_problems", []).append(
            f"Skipped Playlist {playlist_label} ({condition}) step {row['step']} "
            f"song {row['song_id']}: {skip_reason}"
        )
    if state["song_index"] + 1 < len(state["playlists"][condition]):
        state["song_index"] += 1
    else:
        state["stage"] = "ratings"


def _advance_and_persist(skip_reason: str | None = None) -> None:
    advance_current_clip(st.session_state, skip_reason)
    persist_checkpoint()
    st.rerun()


def render_playlist() -> None:
    """Render one condition's five clips sequentially."""
    playlist_index = st.session_state["playlist_index"]
    condition = st.session_state["condition_order"][playlist_index]
    playlist_label = "A" if playlist_index == 0 else "B"
    playlist = st.session_state["playlists"][condition]
    song_index = st.session_state["song_index"]
    row = playlist.iloc[song_index]

    st.header(f"Playlist {playlist_label}")
    st.caption(f"Clip {song_index + 1} of 5")
    st.progress((song_index + 1) / 5)
    audio_path = row["audio_path"]
    audio_available = (
        isinstance(audio_path, (str, os.PathLike))
        and bool(os.fspath(audio_path).strip())
        and Path(audio_path).is_file()
    )
    with st.container(border=True):
        if audio_available:
            st.audio(audio_path, format="audio/mp3")
        else:
            st.error("This clip is not available locally.")

    if not audio_available:
        if st.button("Skip this clip →", type="primary", key=f"skip_missing_{playlist_index}_{song_index}"):
            _advance_and_persist("missing_file")
        return

    if st.button(
        "I finished this clip →" if song_index < 4 else "I finished this playlist →",
        type="primary",
        key=f"finish_clip_{playlist_index}_{song_index}",
    ):
        _advance_and_persist()
    if st.button(
        "Audio not working? Skip this clip",
        type="secondary",
        key=f"skip_audio_{playlist_index}_{song_index}",
    ):
        _advance_and_persist("participant_reported_audio_issue")


def render_playlist_ratings() -> None:
    """Collect post-playlist mood and experience ratings."""
    playlist_index = st.session_state["playlist_index"]
    condition = st.session_state["condition_order"][playlist_index]
    playlist_label = "A" if playlist_index == 0 else "B"

    st.header(f"After Playlist {playlist_label}")
    for anchor in MOOD_ANCHORS:
        st.caption(anchor)
    with st.form(f"ratings_{playlist_label}"):
        post_valence = st.slider(MOOD_LABELS["current_pleasant"], 1, 9, 5)
        post_arousal = st.slider(MOOD_LABELS["current_energy"], 1, 9, 5)
        st.caption("Fit: 1 = not at all, 7 = extremely well")
        fit = st.slider("How well did the playlist fit your mood?", 1, 7, 4)
        st.caption("Satisfaction: 1 = very dissatisfied, 7 = very satisfied")
        satisfaction = st.slider("How satisfied were you with this playlist?", 1, 7, 4)
        recognised_song = st.selectbox(
            "Did you recognise any of the songs?",
            ["Yes", "No", "Not sure"],
        )
        st.caption("Transition naturalness: 1 = very unnatural, 7 = very natural")
        naturalness = st.slider(
            "How natural did the transition between songs feel?",
            1,
            7,
            4,
        )
        submitted = st.form_submit_button("Submit feedback", type="primary")

    if submitted:
        st.session_state["responses"][condition] = {
            "post_valence": post_valence,
            "post_arousal": post_arousal,
            "perceived_fit": fit,
            "satisfaction": satisfaction,
            "recognised_song": recognised_song,
            "transition_naturalness": naturalness,
        }
        if playlist_index == 0:
            st.session_state["playlist_index"] = 1
            st.session_state["song_index"] = 0
            st.session_state["break_started_at"] = datetime.now(timezone.utc).isoformat()
            st.session_state["stage"] = "break"
        else:
            st.session_state["stage"] = "final"
        persist_checkpoint()
        st.rerun()


@st.fragment(run_every="1s")
def render_break() -> None:
    """Enforce the protocol's three-minute quiet break without blocking Streamlit."""
    started = datetime.fromisoformat(st.session_state["break_started_at"])
    remaining = break_remaining_seconds(started, datetime.now(timezone.utc))
    st.header("Quiet break")
    st.write("Now please rest quietly for three minutes. There will be no music.")
    minutes, seconds = divmod(remaining, 60)
    st.metric("Time remaining", f"{minutes}:{seconds:02d}")
    st.progress((180 - remaining) / 180)
    with st.form("break_complete"):
        submitted = st.form_submit_button(
            "Continue to current mood",
            type="primary",
            disabled=remaining > 0,
        )
    if submitted and remaining == 0:
        st.session_state["stage"] = "second_pre"
        persist_checkpoint()
        st.rerun()


def break_remaining_seconds(started: datetime, now: datetime) -> int:
    """Return the persisted three-minute break time remaining."""
    return max(0, 180 - int((now - started).total_seconds()))


def render_second_pre() -> None:
    """Collect the mood re-measurement before Playlist B."""
    st.header("Take a short moment before continuing")
    st.write(
        "When you feel ready, report how you feel right now."
    )
    for anchor in MOOD_ANCHORS:
        st.caption(anchor)
    with st.form("second_pre_mood"):
        pre_valence = st.slider(MOOD_LABELS["current_pleasant"], 1, 9, 5)
        pre_arousal = st.slider(MOOD_LABELS["current_energy"], 1, 9, 5)
        submitted = st.form_submit_button("Continue to Playlist B", type="primary")
    if submitted:
        st.session_state["second_pre"] = {
            "valence": pre_valence,
            "arousal": pre_arousal,
        }
        st.session_state["stage"] = "playlist"
        persist_checkpoint()
        st.rerun()


def _write_csv_atomic(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write one protocol CSV atomically, preserving its exact column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        frame = pd.concat([existing, frame], ignore_index=True)
        frame = frame.loc[:, columns]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=path.parent, delete=False
    ) as temporary:
        frame.to_csv(temporary, index=False)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def update_session_log(state: dict, end_time: str = "") -> None:
    """Upsert this session's protocol row and keep structured recovery notes."""
    path = OUTPUT_FILES["session_log"]
    participant_id = state["participant_id"]
    started_at = datetime.fromisoformat(state["started_at"])
    transition_naturalness = {
        position: state["responses"][condition]["transition_naturalness"]
        for position, condition in zip(("A", "B"), state["condition_order"])
        if "transition_naturalness" in state["responses"].get(condition, {})
    }
    notes = {
        "last_completed_step": state.get("last_completed_step"),
        "skipped_clips": state.get("skipped_clips", []),
        "resume_count": int(state.get("resume_count", 0)),
    }
    if transition_naturalness:
        notes["transition_naturalness"] = transition_naturalness
    row = {
        "participant_id": participant_id,
        "session_date": started_at.date().isoformat(),
        "researcher": "",
        "start_time": state["started_at"],
        "end_time": end_time,
        "app_problems": "; ".join(state.get("app_problems", [])),
        "notes": json.dumps(notes, sort_keys=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str).reindex(columns=SESSION_LOG_COLUMNS)
        existing = existing.loc[existing["participant_id"] != participant_id]
        frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row], columns=SESSION_LOG_COLUMNS)
    frame = frame.loc[:, SESSION_LOG_COLUMNS]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=path.parent, delete=False
    ) as temporary:
        frame.to_csv(temporary, index=False)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def save_responses(state: dict | None = None) -> None:
    """Write one completed participant to the four protocol CSV files."""
    state = st.session_state if state is None else state
    if state.get("saved"):
        return
    participant_id = state["participant_id"]
    initial = state["initial_current"]
    target = state["target"]
    second_pre = state["second_pre"]
    responses = state["responses"]
    assignment = state["assignment"]
    background = state["background"]
    timestamp = datetime.now(timezone.utc)
    participant_row = {
        "participant_id": participant_id,
        "order": assignment["order"],
        "playlist_A_condition": assignment["playlist_A_condition"],
        "playlist_B_condition": assignment["playlist_B_condition"],
        "age_18_plus": state["age_18_plus"],
        "consent": state["consent"],
        **background,
        "target_valence": target.valence,
        "target_arousal": target.arousal,
        "close_to_target_flag": int(
            np.linalg.norm(
                np.array([initial.valence, initial.arousal])
                - np.array([target.valence, target.arousal])
            )
            < 2
        ),
        "preferred_playlist": state["final_preference"],
        "noticed_difference": state["noticed_difference"],
        "comments": state["comment"],
    }
    ratings = []
    songs_played = []
    for playlist_index, condition in enumerate(state["condition_order"]):
        response = responses[condition]
        if playlist_index == 0:
            pre_valence = initial.valence
            pre_arousal = initial.arousal
        else:
            pre_valence = second_pre["valence"]
            pre_arousal = second_pre["arousal"]
        ratings.append(
            {
                "participant_id": participant_id,
                "position": "A" if playlist_index == 0 else "B",
                "condition": condition,
                "valence_before": pre_valence,
                "arousal_before": pre_arousal,
                "valence_after": response["post_valence"],
                "arousal_after": response["post_arousal"],
                "satisfaction": response["satisfaction"],
                "perceived_fit": response["perceived_fit"],
                "recognised_song": response["recognised_song"],
            }
        )
        playlist = state["playlists"][condition]
        for _, song in playlist.iterrows():
            songs_played.append(
                {
                    "participant_id": participant_id,
                    "position": "A" if playlist_index == 0 else "B",
                    "condition": condition,
                    "step": int(song["step"]),
                    "n_steps": int(song["n_steps"]),
                    "target_valence": float(song["target_valence"]),
                    "target_arousal": float(song["target_arousal"]),
                    "song_id": int(song["song_id"]),
                    "song_valence": float(song["song_valence"]),
                    "song_arousal": float(song["song_arousal"]),
                    "distance": float(song["distance"]),
                    "gap_warning": bool(song["gap_warning"]),
                }
            )
    _write_csv_atomic(
        OUTPUT_FILES["participants"],
        [participant_row],
        [
            "participant_id", "order", "playlist_A_condition", "playlist_B_condition",
            "age_18_plus", "consent", "age_group", "gender", "ethnicity",
            "music_for_mood", "music_hours", "target_valence", "target_arousal",
            "close_to_target_flag", "preferred_playlist", "noticed_difference", "comments",
        ],
    )
    _write_csv_atomic(
        OUTPUT_FILES["ratings"],
        ratings,
        [
            "participant_id", "position", "condition", "valence_before",
            "arousal_before", "valence_after", "arousal_after", "satisfaction",
            "perceived_fit", "recognised_song",
        ],
    )
    _write_csv_atomic(
        OUTPUT_FILES["songs_played"],
        songs_played,
        [
            "participant_id", "position", "condition", "step", "n_steps",
            "target_valence", "target_arousal", "song_id", "song_valence",
            "song_arousal", "distance", "gap_warning",
        ],
    )
    update_session_log(state, end_time=timestamp.isoformat())
    state["saved"] = True
    delete_checkpoint(participant_id)


def render_final() -> None:
    """Collect final preference and save the complete study response."""
    st.header("Final comparison")
    st.write("Which playlist did you prefer overall?")
    with st.form("final_feedback"):
        preference = st.radio(
            "Which playlist did you prefer?",
            ["Playlist A", "Playlist B", "Neither"],
            index=None,
        )
        noticed_difference = st.text_area(
            "Did you notice any difference between the two playlists? If yes, what was it?"
        )
        comment = st.text_area("Is there anything else you would like to tell us?")
        submitted = st.form_submit_button("Save study response", type="primary")
    if submitted and preference is None:
        st.warning("Please choose Playlist A, Playlist B, or Neither before saving.")
    elif submitted:
        st.session_state["final_preference"] = preference
        st.session_state["noticed_difference"] = noticed_difference
        st.session_state["comment"] = comment
        save_responses()
        st.session_state["stage"] = "complete"
        st.rerun()


def counterbalance_assignment(participant_id: str) -> pd.Series | None:
    """Return the protocol assignment for a participant ID."""
    assignments = pd.read_csv(COUNTERBALANCE_PATH, dtype=str)
    matches = assignments.loc[assignments["participant_id"] == participant_id]
    return None if matches.empty else matches.iloc[0]


def render_consent() -> None:
    """Collect the required age and consent decisions before study setup."""
    st.header("Eligibility and consent")
    st.markdown(CONSENT_TEXT)
    with st.form("eligibility_consent"):
        age_18_plus = st.radio(
            "I am 18 years or older.",
            ["Yes", "No"],
            index=None,
        )
        consent = st.radio(
            "I have read the information above and I agree to take part.",
            ["Yes", "No"],
            index=None,
        )
        submitted = st.form_submit_button("Continue", type="primary")
    if not submitted:
        return
    eligibility = eligibility_result(age_18_plus, consent)
    if eligibility == "under_18":
        st.session_state["eligibility_stopped"] = True
        st.warning("Thank you. You are not eligible to continue.")
    elif eligibility == "no_consent":
        st.session_state["eligibility_stopped"] = True
        st.warning("Thank you. The study will not continue without consent.")
    elif eligibility == "incomplete":
        st.warning("Please answer both questions before continuing.")
    else:
        st.session_state["eligible"] = True
        st.session_state["age_18_plus"] = age_18_plus
        st.session_state["consent"] = consent
        st.rerun()


def eligibility_result(age_18_plus: str | None, consent: str | None) -> str:
    """Classify the two required consent decisions before any study data exists."""
    if age_18_plus is None or consent is None:
        return "incomplete"
    if age_18_plus == "No":
        return "under_18"
    if consent == "No":
        return "no_consent"
    return "eligible"


def render_start(songs: pd.DataFrame, retriever: SongRetriever) -> None:
    """Render participant setup and generate playlists once."""
    st.header("Participant setup")
    resume_id = st.session_state.get("resume_prompt")
    if resume_id:
        st.warning(f"An unfinished session exists for participant {resume_id}.")
        if st.button("Resume previous session", type="primary"):
            try:
                payload = read_checkpoint(resume_id)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                st.error(f"Could not read the saved session: {error}")
            else:
                if payload is None:
                    st.error("The saved session is no longer available.")
                else:
                    resume_checkpoint(payload)
                    st.rerun()
        return
    st.info(
        "For the best experience, please use headphones if possible and set the "
        "volume to a comfortable level."
    )
    targets = supported_targets(songs)
    if not targets:
        st.error("No supported target moods are available.")
        return

    with st.form("participant_setup"):
        participant_id = st.text_input("Participant ID")
        st.caption("Use the assigned ID from the study counterbalancing list.")
        age_group = st.selectbox(
            "What is your age group?",
            ["", "18-21", "22-25", "26-30", "31-40", "over 40", "Prefer not to say"],
        )
        gender = st.selectbox(
            "What is your gender?",
            [
                "",
                "Woman",
                "Man",
                "Non-binary",
                "Prefer to self-describe",
                "Prefer not to say",
            ],
        )
        gender_self_description = ""
        if gender == "Prefer to self-describe":
            gender_self_description = st.text_input("Prefer to self-describe")
        ethnicity = st.multiselect(
            "Which ethnic background(s) best describe you? (choose all that apply)",
            [
                "Asian",
                "Black or African American",
                "Hispanic or Latino/a/x",
                "Middle Eastern or North African",
                "Native American or Alaska Native",
                "Native Hawaiian or Pacific Islander",
                "White",
                "Other: ___",
                "Prefer not to say",
            ],
        )
        ethnicity_other = ""
        if "Other: ___" in ethnicity:
            ethnicity_other = st.text_input("Other ethnic background")
        music_for_mood = st.selectbox(
            "How often do you listen to music to change or manage your mood?",
            ["", "Never", "Rarely", "Sometimes", "Often", "Almost every day"],
        )
        music_hours = st.selectbox(
            "How many hours of music do you listen to on a typical day?",
            ["", "Less than 1", "1-2", "2-4", "More than 4"],
        )
        for anchor in MOOD_ANCHORS:
            st.caption(anchor)
        current_valence = st.slider(MOOD_LABELS["current_pleasant"], 1, 9, 5)
        current_arousal = st.slider(MOOD_LABELS["current_energy"], 1, 9, 5)
        target = st.selectbox(
            "Choose the feeling and energy you would like",
            targets,
            format_func=describe_target,
        )
        submitted = st.form_submit_button("Start study", type="primary")

    assignment = counterbalance_assignment(participant_id.strip()) if submitted else None
    if submitted and not participant_id.strip():
        st.warning("Please enter your assigned participant ID.")
    elif submitted and assignment is None:
        st.warning("Please use an assigned participant ID from the study list.")
    elif submitted and participant_id_is_completed(participant_id.strip()):
        st.warning(
            f"Participant ID {participant_id.strip()} already has a completed session. "
            "Please use a different ID."
        )
    elif submitted and checkpoint_path(participant_id.strip()).is_file():
        st.session_state["resume_prompt"] = participant_id.strip()
        st.rerun()
    elif submitted:
        background = {
            "age_group": age_group,
            "gender": gender_self_description if gender == "Prefer to self-describe" else gender,
            "ethnicity": "; ".join(
                ethnicity + ([ethnicity_other] if ethnicity_other else [])
            ),
            "music_for_mood": music_for_mood,
            "music_hours": music_hours,
        }
        current = Mood(current_valence, current_arousal)
        target_mood = Mood(*target)
        session = build_session(
            retriever,
            current,
            target_mood,
            allocation="joint",
        )
        initialize_state(
            participant_id.strip(),
            current,
            target_mood,
            session,
            assignment,
            background,
        )
        st.session_state["started_at"] = datetime.now(timezone.utc).isoformat()
        st.session_state["stage"] = "playlist"
        persist_checkpoint()
        st.rerun()


def main() -> None:
    """Run the Streamlit study interface."""
    st.set_page_config(page_title="IsoHarmonix Study", page_icon="🎵")
    apply_styles()
    st.title("IsoHarmonix Music Study")
    st.caption(
        "You'll listen to two short music playlists and rate how you feel before and "
        "after each one. Please listen to each clip before continuing. This is a class "
        "research pilot and is not medical treatment."
    )

    preparing = st.empty()
    preparing.info("Preparing the study...")
    try:
        songs = load_study_songs()
        retriever = load_retriever(songs)
    except (FileNotFoundError, ValueError, KeyError, PreparedAudioError) as error:
        preparing.empty()
        if not st.session_state.get("started"):
            if isinstance(error, PreparedAudioError):
                st.error(
                    "Study audio has not been prepared yet. Run:\n\n"
                    "`python scripts/prepare_study_data.py`"
                )
            else:
                st.error(f"Could not load the local DEAM study data: {error}")
        else:
            st.error("Study data is unavailable. Please contact the researcher.")
        return
    preparing.empty()

    if st.session_state.get("eligibility_stopped"):
        return
    if not st.session_state.get("eligible"):
        render_progress("setup")
        render_consent()
        return
    if not st.session_state.get("started"):
        render_progress("setup")
        render_start(songs, retriever)
        return

    stage = st.session_state.get("stage", "playlist")
    render_progress(stage)
    if stage == "playlist":
        render_playlist()
    elif stage == "ratings":
        render_playlist_ratings()
    elif stage == "break":
        render_break()
    elif stage == "second_pre":
        render_second_pre()
    elif stage == "final":
        render_final()
    elif stage == "complete":
        st.success("Thank you!")
        st.write("Your responses have been recorded.")
        if PILOT_MODE and st.button("Start another pilot session"):
            reset_study()


if __name__ == "__main__":
    main()
