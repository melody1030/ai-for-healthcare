"""Pilot app recovery and participant-facing copy checks."""

from datetime import datetime, timedelta, timezone
import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext

import numpy as np
import pandas as pd

import app
from isoharmonix.planner import Mood, SESSION_COLUMNS, SongRetriever, build_session
from isoharmonix.rules import DIRECT, ISO


ROOT = Path(__file__).resolve().parents[1]


def _study_state():
    valence, arousal = np.meshgrid(np.arange(1, 10), np.arange(1, 10))
    songs = pd.DataFrame(
        {
            "song_id": np.arange(1, 82),
            "valence": valence.ravel().astype(float),
            "arousal": arousal.ravel().astype(float),
        }
    )
    current, target = Mood(3, 7), Mood(7, 3)
    session = build_session(
        SongRetriever(songs), current, target, allocation="joint"
    )
    return {
        "started": True,
        "eligible": True,
        "participant_id": "P_TEST",
        "assignment": {
            "order": "iso_first",
            "playlist_A_condition": ISO,
            "playlist_B_condition": DIRECT,
        },
        "background": {
            "age_group": "22-25",
            "gender": "Prefer not to say",
            "ethnicity": "Prefer not to say",
            "music_for_mood": "Sometimes",
            "music_hours": "1-2",
        },
        "age_18_plus": "Yes",
        "consent": "Yes",
        "initial_current": current,
        "target": target,
        "condition_order": [ISO, DIRECT],
        "playlists": {
            condition: session.loc[session["condition"] == condition]
            .reset_index(drop=True)
            .copy()
            for condition in (ISO, DIRECT)
        },
        "playlist_index": 0,
        "song_index": 0,
        "responses": {},
        "second_pre": None,
        "break_started_at": None,
        "started_at": "2026-11-09T12:00:00+00:00",
        "stage": "playlist",
        "last_completed_step": None,
        "skipped_clips": [],
        "resume_count": 0,
        "app_problems": [],
        "saved": False,
    }


def _roundtrip(state):
    payload = json.loads(json.dumps(app._json_safe(app.checkpoint_payload(state))))
    return app.state_from_checkpoint(payload)


class CheckpointTests(unittest.TestCase):
    def test_pilot_and_formal_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "study" / "data"
            with patch.object(app, "STUDY_DATA_DIR", data_dir), patch.object(app, "PILOT_MODE", True):
                self.assertEqual(
                    app.checkpoint_path("P01"),
                    data_dir / "pilot" / "checkpoints" / "P01.json",
                )
            with patch.object(app, "STUDY_DATA_DIR", data_dir), patch.object(app, "PILOT_MODE", False):
                self.assertEqual(
                    app.checkpoint_path("P01"),
                    data_dir / "checkpoints" / "P01.json",
                )

    def test_resume_after_playlist_a_clip_three_preserves_playlists(self):
        state = _study_state()
        for _ in range(3):
            app.advance_current_clip(state)
        self.assertEqual(state["song_index"], 3)
        self.assertEqual(state["last_completed_step"]["step"], 3)

        restored = _roundtrip(state)
        self.assertEqual(restored["stage"], "playlist")
        self.assertEqual(restored["playlist_index"], 0)
        self.assertEqual(restored["song_index"], 3)
        for condition in (ISO, DIRECT):
            pd.testing.assert_frame_equal(
                state["playlists"][condition], restored["playlists"][condition]
            )

    def test_resume_after_playlist_a_feedback_preserves_break_and_responses(self):
        state = _study_state()
        state["playlist_index"] = 1
        state["stage"] = "break"
        state["break_started_at"] = "2026-11-09T12:04:00+00:00"
        state["responses"][ISO] = {
            "post_valence": 6,
            "post_arousal": 5,
            "perceived_fit": 5,
            "satisfaction": 5,
            "recognised_song": "No",
            "transition_naturalness": 4,
        }

        restored = _roundtrip(state)
        self.assertEqual(restored["stage"], "break")
        self.assertEqual(restored["playlist_index"], 1)
        self.assertEqual(restored["responses"], state["responses"])
        self.assertEqual(restored["break_started_at"], state["break_started_at"])
        for condition in (ISO, DIRECT):
            pd.testing.assert_frame_equal(
                state["playlists"][condition], restored["playlists"][condition]
            )

    def test_resume_during_quiet_break_keeps_original_timer(self):
        state = _study_state()
        state["stage"] = "break"
        state["playlist_index"] = 1
        state["break_started_at"] = "2026-11-09T12:00:00+00:00"
        restored = _roundtrip(state)
        remaining = app.break_remaining_seconds(
            datetime.fromisoformat(restored["break_started_at"]),
            datetime.fromisoformat("2026-11-09T12:01:00+00:00"),
        )
        self.assertEqual(remaining, 120)
        self.assertEqual(restored["stage"], "break")

    def test_missing_audio_skip_advances_and_logs_song_id(self):
        state = _study_state()
        skipped_row = state["playlists"][ISO].iloc[0]
        app.advance_current_clip(state, "missing_file")
        skip = state["skipped_clips"][0]
        self.assertEqual(state["song_index"], 1)
        self.assertEqual(skip["playlist"], "A")
        self.assertEqual(skip["condition"], ISO)
        self.assertEqual(skip["step"], 1)
        self.assertEqual(skip["song_id"], int(skipped_row["song_id"]))
        self.assertEqual(skip["reason"], "missing_file")
        datetime.fromisoformat(skip["timestamp"])

        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "study" / "data"
            log_path = Path(temporary) / "session_log.csv"
            with patch.object(app, "STUDY_DATA_DIR", data_dir), patch.dict(
                app.OUTPUT_FILES, {"session_log": log_path}
            ):
                app.persist_checkpoint(state)
            log = pd.read_csv(log_path, keep_default_na=False)
            notes = json.loads(log.loc[0, "notes"])
            self.assertEqual(notes["skipped_clips"][0]["song_id"], int(skipped_row["song_id"]))
            self.assertIn("Skipped Playlist A", log.loc[0, "app_problems"])
            self.assertEqual(log.loc[0, "end_time"], "")

    def test_resume_upserts_one_session_log_row_and_increments_count(self):
        state = _study_state()
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "study" / "data"
            log_path = Path(temporary) / "session_log.csv"
            with patch.object(app, "STUDY_DATA_DIR", data_dir), patch.dict(
                app.OUTPUT_FILES, {"session_log": log_path}
            ):
                app.persist_checkpoint(state)
                payload = app.read_checkpoint(state["participant_id"])
                app.resume_checkpoint(payload, state)
                frame = pd.read_csv(log_path, keep_default_na=False)
            self.assertEqual(len(frame), 1)
            self.assertEqual(json.loads(frame.loc[0, "notes"])["resume_count"], 1)
            self.assertEqual(frame.loc[0, "end_time"], "")

    def test_final_save_keeps_official_schemas_and_deletes_checkpoint(self):
        state = _study_state()
        state.update(
            {
                "stage": "final",
                "playlist_index": 1,
                "second_pre": {"valence": 5, "arousal": 5},
                "final_preference": "Playlist A",
                "noticed_difference": "",
                "comment": "",
            }
        )
        state["responses"] = {
            condition: {
                "post_valence": 5,
                "post_arousal": 5,
                "perceived_fit": 4,
                "satisfaction": 4,
                "recognised_song": "No",
                "transition_naturalness": 4,
            }
            for condition in (ISO, DIRECT)
        }
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            data_dir = temp / "study" / "data"
            paths = {
                key: temp / f"{key}.csv"
                for key in ("participants", "ratings", "songs_played", "session_log")
            }
            with patch.object(app, "STUDY_DATA_DIR", data_dir), patch.dict(
                app.OUTPUT_FILES, paths
            ):
                app.persist_checkpoint(state)
                self.assertTrue(app.checkpoint_path(state["participant_id"]).exists())
                app.save_responses(state)
                self.assertFalse(app.checkpoint_path(state["participant_id"]).exists())
                self.assertEqual(len(pd.read_csv(paths["session_log"])), 1)
                self.assertTrue(pd.read_csv(paths["session_log"]).loc[0, "end_time"])
                for key in paths:
                    template = pd.read_csv(ROOT / "study" / "templates" / f"{key}.csv")
                    self.assertEqual(list(pd.read_csv(paths[key]).columns), list(template.columns))


class ParticipantCopyTests(unittest.TestCase):
    def test_consent_text_matches_questionnaire_section_zero(self):
        questionnaire = (ROOT / "docs" / "questionnaire.md").read_text(encoding="utf-8")
        section = questionnaire.split("## Section 0. Consent", 1)[1].split("\n| # | Item", 1)[0]
        source_lines = []
        for line in section.splitlines():
            if line.startswith(">"):
                source_lines.append(line[1:].lstrip())
            elif line.strip():
                source_lines.append(line)
            else:
                source_lines.append("")
        source_text = "\n".join(source_lines).strip()
        self.assertEqual(app.CONSENT_TEXT.strip(), source_text)
        self.assertNotIn("[team email]", app.CONSENT_TEXT)
        self.assertNotIn("Questions? Contact", app.CONSENT_TEXT)

    def test_mood_ui_copy_has_no_technical_axis_terms(self):
        visible_copy = list(app.MOOD_LABELS.values()) + list(app.MOOD_ANCHORS)
        visible_copy += [app.describe_target((value, value)) for value in range(1, 10)]
        for text in visible_copy:
            self.assertNotRegex(text.lower(), r"\b(valence|arousal)\b")
        render_tree = ast.parse(
            "\n".join(
                inspect.getsource(function)
                for function in (
                    app.render_start,
                    app.render_second_pre,
                    app.render_playlist_ratings,
                    app.render_playlist,
                )
            )
        )
        for node in ast.walk(render_tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "st"
            ):
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    self.assertNotRegex(argument.value.lower(), r"\b(valence|arousal)\b")
        playlist_copy = inspect.getsource(app.render_playlist)
        self.assertIn("Skip this clip →", playlist_copy)
        self.assertIn("Audio not working? Skip this clip", playlist_copy)
        self.assertEqual(app.MOOD_LABELS["target_pleasant"], "Pleasant feeling you would like")
        self.assertEqual(app.MOOD_LABELS["target_energy"], "Energy you would like")
        self.assertEqual(app.MOOD_LABELS["current_pleasant"], "How pleasant do you feel?")
        self.assertEqual(app.MOOD_LABELS["current_energy"], "How much energy do you feel?")

    def test_supported_target_set_is_unchanged(self):
        coordinates = np.repeat(
            np.array([(valence, energy) for valence in range(1, 10) for energy in range(1, 10)]),
            7,
            axis=0,
        )
        songs = pd.DataFrame(
            {
                "song_id": np.arange(1, len(coordinates) + 1),
                "valence": coordinates[:, 0],
                "arousal": coordinates[:, 1],
            }
        )
        expected = tuple((valence, energy) for valence in range(1, 10) for energy in range(1, 10))
        self.assertEqual(app.supported_targets(songs), expected)
        self.assertEqual(len(app.supported_targets(songs)), 81)

    def test_render_start_still_uses_joint_allocation(self):
        tree = ast.parse(inspect.getsource(app.render_start))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_session"
        ]
        self.assertEqual(len(calls), 1)
        self.assertTrue(
            any(
                keyword.arg == "allocation"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "joint"
                for keyword in calls[0].keywords
            )
        )

    def test_playlist_a_clip_one_renders_all_applicable_buttons(self):
        class FakeStreamlit:
            def __init__(self, state):
                self.session_state = state
                self.buttons = []
                self.headings = []
                self.captions = []

            def header(self, text):
                self.headings.append(text)

            def caption(self, text):
                self.captions.append(text)

            def progress(self, value):
                pass

            def container(self, **kwargs):
                return nullcontext()

            def audio(self, path, **kwargs):
                pass

            def error(self, text):
                pass

            def button(self, label, **kwargs):
                self.buttons.append((label, kwargs["key"]))
                return False

        state = _study_state()
        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "clip.mp3"
            audio_path.touch()
            state["playlists"][ISO].loc[0, "audio_path"] = str(audio_path)
            fake_st = FakeStreamlit(state)
            with patch.object(app, "st", fake_st):
                app.render_playlist()
        self.assertIn("Playlist A", fake_st.headings)
        self.assertIn("Clip 1 of 5", fake_st.captions)
        self.assertEqual(
            fake_st.buttons,
            [
                ("I finished this clip →", "finish_clip_0_0"),
                ("Audio not working? Skip this clip", "skip_audio_0_0"),
            ],
        )

        missing_state = _study_state()
        fake_st = FakeStreamlit(missing_state)
        with patch.object(app, "st", fake_st):
            app.render_playlist()
        self.assertEqual(
            fake_st.buttons,
            [("Skip this clip →", "skip_missing_0_0")],
        )


if __name__ == "__main__":
    unittest.main()
