"""Minimal Streamlit app for the IsoHarmonix within-subject pilot."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

from isoharmonix.data import (  # noqa: E402
    clean_songs,
    load_annotations,
    resolve_deam_dir,
)
from isoharmonix.planner import Mood, SongRetriever, build_session  # noqa: E402
from isoharmonix.rules import (  # noqa: E402
    GAP_WARNING,
    DIRECT,
    ISO,
)
from isoharmonix.study_data import PreparedAudioError, load_prepared_audio  # noqa: E402


PILOT_MODE = True
RESPONSES_PATH = ROOT / "results" / (
    "pilot_responses.csv" if PILOT_MODE else "study_responses.csv"
)


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
        "reset": "Reset",
        "playlist_b": "Playlist B",
        "ratings_b": "Final",
    }
    labels = list(stages.values())
    current = {
        "setup": 0,
        "playlist": 1 if st.session_state.get("playlist_index", 0) == 0 else 4,
        "ratings": 2 if st.session_state.get("playlist_index", 0) == 0 else 5,
        "second_pre": 3,
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
    return SongRetriever(songs)


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
        1: "very negative",
        2: "negative",
        3: "somewhat negative",
        4: "slightly negative",
        5: "neutral",
        6: "slightly positive",
        7: "somewhat positive",
        8: "positive",
        9: "very positive",
    }
    arousal_words = {
        1: "very calm / low energy",
        2: "calm / low energy",
        3: "somewhat calm",
        4: "slightly calm",
        5: "moderate energy",
        6: "slightly excited",
        7: "somewhat excited",
        8: "excited / high energy",
        9: "very excited / high energy",
    }
    return (
        f"{valence_words[valence]}, {arousal_words[arousal]} "
        f"(valence {valence}, arousal {arousal})"
    )


def initialize_state(
    participant_id: int,
    current: Mood,
    target: Mood,
    session: pd.DataFrame,
) -> None:
    """Initialize the participant's one-time study state."""
    condition_order = [ISO, DIRECT] if participant_id % 2 else [DIRECT, ISO]
    playlists = {
        condition: session.query("condition == @condition").reset_index(drop=True)
        for condition in (ISO, DIRECT)
    }
    st.session_state.update(
        {
            "started": True,
            "participant_id": participant_id,
            "initial_current": current,
            "target": target,
            "condition_order": condition_order,
            "playlists": playlists,
            "playlist_index": 0,
            "song_index": 0,
            "responses": {},
            "second_pre": None,
            "saved": False,
        }
    )


def reset_study() -> None:
    """Clear study state so another pilot participant can use the app."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def participant_id_is_completed(participant_id: int) -> bool:
    """Return whether the current output CSV already contains this participant."""
    if not RESPONSES_PATH.exists():
        return False
    responses = pd.read_csv(RESPONSES_PATH, usecols=["participant_id"])
    return participant_id in set(responses["participant_id"].dropna().astype(int))


def render_playlist() -> None:
    """Render one condition's five clips sequentially."""
    condition = st.session_state["condition_order"][st.session_state["playlist_index"]]
    playlist_label = "A" if st.session_state["playlist_index"] == 0 else "B"
    playlist = st.session_state["playlists"][condition]
    song_index = st.session_state["song_index"]
    row = playlist.iloc[song_index]

    st.header(f"Playlist {playlist_label}")
    st.caption(f"Clip {song_index + 1} of 5")
    st.progress((song_index + 1) / 5)
    with st.container(border=True):
        audio_path = row["audio_path"]
        if not audio_path or not Path(audio_path).is_file():
            st.error("This clip is not available locally. Please contact the study administrator.")
            return
        st.audio(audio_path, format="audio/mp3")

    if st.button(
        "I finished this clip →" if song_index < 4 else "I finished this playlist →",
        type="primary",
    ):
        if song_index < 4:
            st.session_state["song_index"] += 1
        else:
            st.session_state["stage"] = "ratings"
        st.rerun()


def render_playlist_ratings() -> None:
    """Collect post-playlist mood and experience ratings."""
    playlist_index = st.session_state["playlist_index"]
    condition = st.session_state["condition_order"][playlist_index]
    playlist_label = "A" if playlist_index == 0 else "B"

    st.header(f"After Playlist {playlist_label}")
    st.caption("Valence: 1 = very negative, 9 = very positive")
    st.caption("Arousal: 1 = very calm/low energy, 9 = very excited/high energy")
    with st.form(f"ratings_{playlist_label}"):
        post_valence = st.slider("Mood valence after this playlist", 1, 9, 5)
        post_arousal = st.slider("Mood arousal after this playlist", 1, 9, 5)
        st.caption("Fit: 1 = not at all, 7 = extremely well")
        fit = st.slider("How well did the playlist fit your mood?", 1, 7, 4)
        st.caption("Satisfaction: 1 = very dissatisfied, 7 = very satisfied")
        satisfaction = st.slider("How satisfied were you with this playlist?", 1, 7, 4)
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
            "transition_naturalness": naturalness,
        }
        if playlist_index == 0:
            st.session_state["playlist_index"] = 1
            st.session_state["song_index"] = 0
            st.session_state["stage"] = "second_pre"
        else:
            st.session_state["stage"] = "final"
        st.rerun()


def render_second_pre() -> None:
    """Collect the mood re-measurement before Playlist B."""
    st.header("Take a short moment before continuing")
    st.write(
        "When you feel ready, report how you feel right now."
    )
    with st.form("second_pre_mood"):
        pre_valence = st.slider("Current valence right now", 1, 9, 5)
        pre_arousal = st.slider("Current arousal right now", 1, 9, 5)
        submitted = st.form_submit_button("Continue to Playlist B", type="primary")
    if submitted:
        st.session_state["second_pre"] = {
            "valence": pre_valence,
            "arousal": pre_arousal,
        }
        st.session_state["stage"] = "playlist"
        st.rerun()


def save_responses() -> None:
    """Append both condition rows for the completed participant session."""
    participant_id = st.session_state["participant_id"]
    initial = st.session_state["initial_current"]
    target = st.session_state["target"]
    second_pre = st.session_state["second_pre"]
    responses = st.session_state["responses"]
    rows = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for playlist_index, condition in enumerate(st.session_state["condition_order"]):
        response = responses[condition]
        if playlist_index == 0:
            pre_valence = initial.valence
            pre_arousal = initial.arousal
        else:
            pre_valence = second_pre["valence"]
            pre_arousal = second_pre["arousal"]
        rows.append(
            {
                "participant_id": participant_id,
                "playlist_label": "A" if playlist_index == 0 else "B",
                "condition": condition,
                "condition_order": playlist_index + 1,
                "initial_current_valence": initial.valence,
                "initial_current_arousal": initial.arousal,
                "pre_valence": pre_valence,
                "pre_arousal": pre_arousal,
                "target_valence": target.valence,
                "target_arousal": target.arousal,
                **response,
                "final_preference": st.session_state["final_preference"],
                "comment": st.session_state["comment"],
                "completed_at_utc": timestamp,
            }
        )
    new_rows = pd.DataFrame(rows)
    RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESPONSES_PATH.exists():
        new_rows.to_csv(RESPONSES_PATH, mode="a", header=False, index=False)
    else:
        new_rows.to_csv(RESPONSES_PATH, index=False)
    st.session_state["saved"] = True


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
        comment = st.text_area("Optional comment")
        submitted = st.form_submit_button("Save study response", type="primary")
    if submitted and preference is None:
        st.warning("Please choose Playlist A, Playlist B, or Neither before saving.")
    elif submitted:
        st.session_state["final_preference"] = preference
        st.session_state["comment"] = comment
        save_responses()
        st.session_state["stage"] = "complete"
        st.rerun()


def render_start(songs: pd.DataFrame, retriever: SongRetriever) -> None:
    """Render participant setup and generate playlists once."""
    st.header("Participant setup")
    st.info(
        "For the best experience, please use headphones if possible and set the "
        "volume to a comfortable level."
    )
    targets = supported_targets(songs)
    if not targets:
        st.error("No supported target moods are available.")
        return

    with st.form("participant_setup"):
        participant_id = st.number_input("Participant ID", min_value=1, step=1, value=1)
        st.caption("Valence: 1 = very negative, 9 = very positive")
        st.caption("Arousal: 1 = very calm/low energy, 9 = very excited/high energy")
        current_valence = st.slider("Current valence", 1, 9, 5)
        current_arousal = st.slider("Current arousal", 1, 9, 5)
        target = st.selectbox(
            "How would you like to feel after listening?",
            targets,
            format_func=describe_target,
        )
        submitted = st.form_submit_button("Start study", type="primary")

    if submitted and participant_id_is_completed(int(participant_id)):
        st.warning(
            f"Participant ID {int(participant_id)} already has a completed session. "
            "Please use a different ID."
        )
    elif submitted:
        current = Mood(current_valence, current_arousal)
        target_mood = Mood(*target)
        session = build_session(
            retriever,
            current,
            target_mood,
            allocation="joint",
        )
        initialize_state(int(participant_id), current, target_mood, session)
        st.session_state["stage"] = "playlist"
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
