"""Real DEAM checks. Skipped when the dataset is not on disk.

A skip means the check was not verified. It is not a pass of playback or of
the published ratings catalog.
"""

import unittest
from pathlib import Path

import pandas as pd

from isoharmonix.data import classify_audio, clean_songs, load_annotations, resolve_deam_dir, scan_audio
from isoharmonix.planner import Mood, SongRetriever, build_session, session_audio_report
from isoharmonix.rules import N_STEPS


def _deam_root() -> Path | None:
    root = resolve_deam_dir()
    marker = "static_annotations_averaged_songs_1_2000.csv"
    if root.exists() and any(root.rglob(marker)):
        return root
    return None


DEAM = _deam_root()


@unittest.skipUnless(DEAM is not None, "DEAM annotations are not in data/DEAM or DEAM_DATA_DIR; not verified")
class DeamAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_annotations(DEAM)
        cls.scan = scan_audio(DEAM)
        cls.songs, cls.report = clean_songs(
            cls.raw,
            cls.scan.playable,
            require_audio=cls.scan.mp3_count > 0,
            unreadable_ids=cls.scan.unreadable_ids,
        )

    def test_ratings_and_session_rules(self):
        self.assertGreater(len(self.raw), 1000)
        self.assertTrue(self.songs["song_id"].is_unique)
        self.assertGreaterEqual(len(self.songs), 2 * N_STEPS)
        session = build_session(SongRetriever(self.songs), Mood(3, 7), Mood(7, 3))
        self.assertEqual(len(session), 2 * N_STEPS)
        self.assertTrue(session["song_id"].is_unique)
        iso = session[session["condition"] == "iso"]
        direct = session[session["condition"] == "direct"]
        self.assertAlmostEqual(iso.iloc[0]["target_valence"], 3)
        self.assertAlmostEqual(iso.iloc[0]["target_arousal"], 7)
        self.assertAlmostEqual(iso.iloc[-1]["target_valence"], 7)
        self.assertAlmostEqual(iso.iloc[-1]["target_arousal"], 3)
        self.assertTrue((direct["target_valence"] == 7).all())
        self.assertTrue((direct["target_arousal"] == 3).all())
        # Distances agree with the ratings of the chosen songs.
        ratings = self.songs.set_index("song_id")
        for row in session.itertuples():
            song = ratings.loc[row.song_id]
            distance = ((song["valence"] - row.target_valence) ** 2 + (song["arousal"] - row.target_arousal) ** 2) ** 0.5
            self.assertAlmostEqual(row.distance, round(float(distance), 3), places=3)

    def test_audio_files_if_present(self):
        if self.scan.mp3_count == 0:
            self.skipTest(
                f"No MP3 files under {DEAM}. Annotation session was built; playback is not verified."
            )
        session = build_session(SongRetriever(self.songs), Mood(3, 7), Mood(7, 3))
        report = session_audio_report(session)
        self.assertEqual(report["ok"], 2 * N_STEPS)
        self.assertEqual(report["unreadable"], 0)
        sample = session["audio_path"].head(3)
        for path in sample:
            self.assertTrue(Path(path).is_file())
            self.assertEqual(classify_audio(path), "ok")
            self.assertEqual(Path(path).stem.lstrip("0"), str(int(session.loc[session["audio_path"] == path, "song_id"].iloc[0])))


if __name__ == "__main__":
    unittest.main()
