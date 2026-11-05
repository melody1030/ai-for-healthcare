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
from isoharmonix.planner import Mood, SongRetriever, build_session  # noqa: E402
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
            "saved": False,
        }
    )


def reset_study() -> None:
    """Clear study state so another pilot participant can use the app."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def participant_id_is_completed(participant_id: str) -> bool:
    """Return whether the participant already has a saved protocol record."""
    path = OUTPUT_FILES["participants"]
    if not path.exists():
        return False
    responses = pd.read_csv(path, usecols=["participant_id"], dtype=str)
    return participant_id in set(responses["participant_id"].dropna())


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


def save_responses() -> None:
    """Write one completed participant to the four protocol CSV files."""
    if st.session_state.get("saved"):
        return
    participant_id = st.session_state["participant_id"]
    initial = st.session_state["initial_current"]
    target = st.session_state["target"]
    second_pre = st.session_state["second_pre"]
    responses = st.session_state["responses"]
    assignment = st.session_state["assignment"]
    background = st.session_state["background"]
    timestamp = datetime.now(timezone.utc)
    session_date = timestamp.date().isoformat()
    start_time = st.session_state["started_at"]
    participant_row = {
        "participant_id": participant_id,
        "order": assignment["order"],
        "playlist_A_condition": assignment["playlist_A_condition"],
        "playlist_B_condition": assignment["playlist_B_condition"],
        "age_18_plus": st.session_state["age_18_plus"],
        "consent": st.session_state["consent"],
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
        "preferred_playlist": st.session_state["final_preference"],
        "noticed_difference": st.session_state["noticed_difference"],
        "comments": st.session_state["comment"],
    }
    ratings = []
    songs_played = []
    for playlist_index, condition in enumerate(st.session_state["condition_order"]):
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
        playlist = st.session_state["playlists"][condition]
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
    notes = json.dumps(
        {
            "transition_naturalness": {
                position: responses[condition]["transition_naturalness"]
                for position, condition in zip(("A", "B"), st.session_state["condition_order"])
            }
        },
        sort_keys=True,
    )
    session_row = {
        "participant_id": participant_id,
        "session_date": session_date,
        "researcher": "",
        "start_time": start_time,
        "end_time": timestamp.isoformat(),
        "app_problems": "",
        "notes": notes,
    }
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
    _write_csv_atomic(
        OUTPUT_FILES["session_log"],
        [session_row],
        [
            "participant_id", "session_date", "researcher", "start_time",
            "end_time", "app_problems", "notes",
        ],
    )
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
    st.write(
        "I am 18 years or older. I have read the information above and I agree "
        "to take part."
    )
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
