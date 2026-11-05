"""Audio discovery and playable-song filtering. Uses temporary files, not DEAM."""

import os
import shutil
import unittest
from pathlib import Path

import pandas as pd

from isoharmonix.data import (
    classify_audio,
    clean_songs,
    decoder_name,
    load_annotations,
    resolve_deam_dir,
    scan_audio,
)
from isoharmonix.planner import Mood, SongRetriever, build_session, session_audio_report


# MPEG1 Layer 3, 128 kbps, 44100 Hz, stereo. One frame is 417 bytes.
# afinfo on macOS opens this frame; the header check accepts it without a decoder.
_MP3_FRAME = bytes.fromhex("FFFB9000") + b"\x00" * (417 - 4)


def _write_mp3(path: Path, frames: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_MP3_FRAME * frames)


def _ratings(rows):
    frame = pd.DataFrame(rows, columns=["song_id", "valence", "arousal", "valence_std", "arousal_std"])
    return frame


class AudioDiscoveryTests(unittest.TestCase):
    def test_scan_matches_nested_numeric_names_and_ignores_other_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_mp3(base / "DEAM_audio" / "MEMD_audio" / "0002.mp3")
            _write_mp3(base / "elsewhere" / "15.MP3")
            (base / "note.mp3").write_bytes(b"not-a-song-id")
            (base / "2.txt").write_text("nope")
            (base / "empty.mp3").write_bytes(b"")  # stem is not digits
            scan = scan_audio(base)
            self.assertEqual(scan.files_found, 2)
            self.assertEqual(set(scan.playable), {2, 15})
            self.assertTrue(str(scan.playable[2]).endswith("0002.mp3"))
            self.assertEqual(classify_audio(scan.playable[2]), "ok")

    def test_decoder_rejects_empty_and_garbage_and_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            empty = base / "1.mp3"
            empty.write_bytes(b"")
            garbage = base / "2.mp3"
            garbage.write_bytes(b"this is not audio")
            good = base / "3.mp3"
            _write_mp3(good)
            self.assertEqual(classify_audio(empty), "unreadable")
            self.assertEqual(classify_audio(garbage), "unreadable")
            self.assertEqual(classify_audio(base / "missing.mp3"), "missing")
            self.assertEqual(classify_audio(None), "missing")
            self.assertEqual(classify_audio(good), "ok")
            if decoder_name() is None:
                self.skipTest("header check accepted the fixture; no external decoder is installed")
            # The assertion above already required a real open when a decoder exists.
            self.assertIn(decoder_name(), {"afinfo", "ffprobe"})

    def test_unreadable_duplicate_loses_to_a_readable_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bad = base / "a" / "4.mp3"
            good = base / "b" / "4.mp3"
            bad.parent.mkdir()
            good.parent.mkdir()
            bad.write_bytes(b"nope")
            _write_mp3(good)
            scan = scan_audio(base)
            self.assertEqual(scan.playable[4].resolve(), good.resolve())
            self.assertEqual(scan.unreadable_ids, ())
            self.assertEqual(scan.unreadable_count, 1)
            self.assertEqual(scan.duplicate_ids, (4,))

    def test_data_dir_comes_from_the_environment_not_a_hardcoded_home(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("DEAM_DATA_DIR")
            os.environ["DEAM_DATA_DIR"] = tmp
            try:
                self.assertEqual(resolve_deam_dir(), Path(tmp).resolve())
                self.assertNotIn("/Users/", str(resolve_deam_dir()).replace(str(Path(tmp).resolve()), ""))
            finally:
                if previous is None:
                    os.environ.pop("DEAM_DATA_DIR", None)
                else:
                    os.environ["DEAM_DATA_DIR"] = previous


class CleaningTests(unittest.TestCase):
    def _csv(self, directory: Path, name: str, text: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text)

    def test_loader_strips_headers_ignores_extra_columns_and_keeps_first_duplicate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "annotations" / "song_level"
            self._csv(
                root,
                "static_annotations_averaged_songs_1_2000.csv",
                "song_id, valence_mean, valence_std, arousal_mean, arousal_std\n"
                "2, 3.5, 0.2, 6.5, 0.4\n"
                "2000, 5.0, 0.1, 5.0, 0.1\n",
            )
            self._csv(
                Path(tmp) / "extra",
                "static_annotations_averaged_songs_2000_2058.csv",
                " song_id, valence_mean, valence_std, arousal_mean, arousal_std, extra_column\n"
                "2000, 9.0, 0.9, 9.0, 0.9, ignore-me\n"
                "2058, 4.0, 0.2, 4.0, 0.2, ignore-me\n",
            )
            songs = load_annotations(tmp)
            self.assertEqual(list(songs["song_id"]), [2, 2000, 2058])
            row = songs.set_index("song_id").loc[2000]
            self.assertEqual(row["valence"], 5.0)

    def test_cleaning_rules_without_audio(self):
        songs = _ratings(
            [
                (1, 5, 5, 0.1, 0.1),
                (2, None, 5, 0.1, 0.1),
                (3, 0, 5, 0.1, 0.1),
                (4, 5, 5, 3, 3),
            ]
        )
        # Add enough low-spread songs that song 4 is above the 90th percentile.
        extra = [(10 + i, 5, 5, 0.2, 0.2) for i in range(20)]
        songs = pd.concat([songs, _ratings(extra)], ignore_index=True)
        cleaned, report = clean_songs(songs, audio={})
        self.assertNotIn(2, set(cleaned["song_id"]))
        self.assertNotIn(3, set(cleaned["song_id"]))
        self.assertNotIn(4, set(cleaned["song_id"]))
        self.assertIn(1, set(cleaned["song_id"]))
        self.assertTrue(cleaned["audio_path"].isna().all())
        self.assertEqual(report["Dropped, missing audio"], "skipped (no MP3 files found)")
        self.assertLessEqual(cleaned["spread"].max(), report["Spread cutoff"] + 1e-9)

    def test_missing_and_unreadable_audio_are_dropped_only_when_files_exist(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_mp3(base / "MEMD_audio" / "1.mp3")
            _write_mp3(base / "MEMD_audio" / "2.mp3")
            (base / "MEMD_audio" / "3.mp3").write_bytes(b"broken")
            scan = scan_audio(base)
            songs = _ratings(
                [
                    (1, 5, 5, 0.1, 0.1),
                    (2, 6, 4, 0.1, 0.1),
                    (3, 4, 6, 0.1, 0.1),
                    (4, 5, 4, 0.1, 0.1),  # no file
                ]
            )
            cleaned, report = clean_songs(
                songs,
                scan.playable,
                require_audio=scan.mp3_count > 0,
                unreadable_ids=scan.unreadable_ids,
            )
            self.assertEqual(set(cleaned["song_id"]), {1, 2})
            self.assertEqual(report["Dropped, missing audio"], 1)
            self.assertEqual(report["Dropped, unreadable audio"], 1)
            self.assertTrue(Path(cleaned.set_index("song_id").loc[1, "audio_path"]).is_file())


class PlayableRecommendationTests(unittest.TestCase):
    def test_unreadable_or_missing_files_are_not_recommended_when_audio_exists(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Song 1 is the perfect match but cannot be played. Song 2 is next.
            (base / "1.mp3").write_bytes(b"broken")
            _write_mp3(base / "2.mp3")
            for song_id in range(3, 13):
                _write_mp3(base / f"{song_id}.mp3")
            scan = scan_audio(base)
            rows = [(1, 7, 3, 0.1, 0.1), (2, 7.4, 3.2, 0.1, 0.1)]
            rows += [(i, 5, 5, 0.1, 0.1) for i in range(3, 13)]
            rows.append((99, 7, 3, 0.1, 0.1))  # closer duplicate mood, no file
            cleaned, _report = clean_songs(
                _ratings(rows),
                scan.playable,
                require_audio=True,
                unreadable_ids=scan.unreadable_ids,
            )
            session = build_session(SongRetriever(cleaned), Mood(3, 7), Mood(7, 3))
            self.assertNotIn(1, set(session["song_id"]))
            self.assertNotIn(99, set(session["song_id"]))
            report = session_audio_report(session)
            self.assertEqual(report["ok"], 10)
            self.assertEqual(report["missing"], 0)
            self.assertEqual(report["unreadable"], 0)
            for path in session["audio_path"]:
                self.assertTrue(Path(path).is_file())
                self.assertEqual(classify_audio(path), "ok")

    def test_ratings_only_catalog_still_plans_but_is_not_playable(self):
        songs = pd.DataFrame(
            {
                "song_id": list(range(1, 13)),
                "valence": [5.0] * 12,
                "arousal": [5.0] * 12,
                "audio_path": [None] * 12,
            }
        )
        session = build_session(SongRetriever(songs), Mood(5, 5), Mood(6, 6))
        report = session_audio_report(session)
        self.assertEqual(report["missing"], 10)
        self.assertEqual(report["ok"], 0)
        self.assertTrue(session["audio_path"].isna().all())


class DecoderAvailabilityTests(unittest.TestCase):
    def test_this_environment_reports_its_decoder(self):
        # Documented for the validation report. Header-only mode is allowed,
        # but this machine is expected to have afinfo or ffprobe when present.
        name = decoder_name()
        if name is None and shutil.which("afinfo") is None and shutil.which("ffprobe") is None:
            self.skipTest("no external audio decoder installed")
        self.assertIsNotNone(name)


if __name__ == "__main__":
    unittest.main()
