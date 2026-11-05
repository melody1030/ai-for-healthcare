"""Waypoint and retrieval tests on synthetic catalogs. No DEAM files required."""

import math
import unittest

import numpy as np
import pandas as pd

from isoharmonix.planner import (
    SESSION_COLUMNS,
    Mood,
    SongRetriever,
    build_playlist,
    build_session,
    direct_path,
    iso_path,
    plan_path,
)
from isoharmonix.rules import DIRECT, GAP_WARNING, ISO, N_STEPS


def _songs(rows):
    return pd.DataFrame(rows, columns=["song_id", "valence", "arousal"])


def _grid(n=9):
    valence, arousal = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))
    return pd.DataFrame(
        {
            "song_id": np.arange(1, n * n + 1),
            "valence": valence.ravel().astype(float),
            "arousal": arousal.ravel().astype(float),
        }
    )


class WaypointTests(unittest.TestCase):
    def test_iso_from_tense_to_calm_has_five_even_steps(self):
        current, target = Mood(3, 7), Mood(7, 3)
        path = iso_path(current, target)
        self.assertEqual(path.shape, (N_STEPS, 2))
        np.testing.assert_allclose(path[0], [3, 7])
        np.testing.assert_allclose(path[-1], [7, 3])
        np.testing.assert_allclose(np.diff(path, axis=0), np.tile([1.0, -1.0], (N_STEPS - 1, 1)))

    def test_direct_path_is_five_copies_of_the_target(self):
        path = direct_path(Mood(7, 3))
        self.assertEqual(len(path), N_STEPS)
        np.testing.assert_allclose(path, np.tile([7.0, 3.0], (N_STEPS, 1)))

    def test_current_equals_target(self):
        mood = Mood(4, 6)
        iso = iso_path(mood, mood)
        direct = direct_path(mood)
        np.testing.assert_allclose(iso, np.tile([4.0, 6.0], (N_STEPS, 1)))
        np.testing.assert_allclose(direct, np.tile([4.0, 6.0], (N_STEPS, 1)))

    def test_extreme_corners_stay_on_the_scale_and_are_evenly_spaced(self):
        path = iso_path(Mood(1, 9), Mood(9, 1))
        np.testing.assert_allclose(path[0], [1, 9])
        np.testing.assert_allclose(path[-1], [9, 1])
        np.testing.assert_allclose(np.diff(path, axis=0), np.tile([2.0, -2.0], (N_STEPS - 1, 1)))
        self.assertTrue(np.all((path >= 1) & (path <= 9)))

    def test_mood_rejects_values_outside_1_to_9(self):
        for args in ((0, 5), (10, 5), (1, 9.1), (float("nan"), 5), (5, float("inf"))):
            with self.assertRaises(ValueError):
                Mood(*args)
        with self.assertRaises(ValueError):
            Mood("3", 7)
        with self.assertRaises(ValueError):
            plan_path("neither", Mood(3, 7), Mood(7, 3))

    def test_paths_are_deterministic(self):
        np.testing.assert_array_equal(
            iso_path(Mood(2, 8), Mood(8, 2)),
            iso_path(Mood(2, 8), Mood(8, 2)),
        )


class RetrievalTests(unittest.TestCase):
    def test_nearest_neighbor_euclidean_distance_and_song_id_tie_break(self):
        songs = _songs(
            [
                (2, 7.0, 3.2),  # distance to (7, 3) = 0.2
                (1, 7.0, 3.2),  # same point, lower id wins
                (3, 9.0, 9.0),
            ]
        )
        retriever = SongRetriever(songs)
        index, distance = retriever.closest_unused([7, 3], set())
        self.assertEqual(int(retriever.ids[index]), 1)
        self.assertAlmostEqual(distance, 0.2)

        index, distance = retriever.closest_unused([7, 3], {1})
        self.assertEqual(int(retriever.ids[index]), 2)
        self.assertAlmostEqual(distance, 0.2)

    def test_lower_id_does_not_beat_a_closer_song(self):
        songs = _songs([(1, 8.0, 3.0), (9, 7.1, 3.0)])
        retriever = SongRetriever(songs)
        index, distance = retriever.closest_unused([7, 3], set())
        self.assertEqual(int(retriever.ids[index]), 9)
        self.assertAlmostEqual(distance, 0.1)

    def test_iso_first_on_a_perfect_grid_hits_every_iso_waypoint(self):
        session = build_session(
            SongRetriever(_grid()), Mood(3, 7), Mood(7, 3), allocation="iso_first"
        )
        iso = session[session["condition"] == ISO]
        self.assertEqual(len(session), 2 * N_STEPS)
        self.assertTrue(session["song_id"].is_unique)
        self.assertTrue((iso["distance"] == 0).all())
        np.testing.assert_allclose(
            iso[["target_valence", "target_arousal"]].to_numpy(),
            [[3, 7], [4, 6], [5, 5], [6, 4], [7, 3]],
        )
        direct = session[session["condition"] == DIRECT]
        np.testing.assert_allclose(
            direct[["target_valence", "target_arousal"]].to_numpy(),
            np.tile([7.0, 3.0], (N_STEPS, 1)),
        )

    def test_no_song_is_reused_within_or_across_playlists(self):
        # Every waypoint wants the same song. Uniqueness has to come from the used set.
        songs = _songs([(10 + i, 5 + 0.01 * i, 5) for i in range(12)])
        session = build_session(SongRetriever(songs), Mood(5, 5), Mood(5, 5), allocation="iso_first")
        self.assertEqual(len(session), 10)
        self.assertEqual(session["song_id"].nunique(), 10)
        iso_ids = set(session.loc[session["condition"] == ISO, "song_id"])
        direct_ids = set(session.loc[session["condition"] == DIRECT, "song_id"])
        self.assertTrue(iso_ids.isdisjoint(direct_ids))

    def test_distance_matches_manual_euclidean_and_gap_flag(self):
        songs = _songs([(1, 7.2, 3.2), (2, 1.0, 1.0)])
        session = build_playlist(SongRetriever(songs), DIRECT, Mood(1, 1), Mood(7, 3), n_steps=2)
        # Only two songs and two direct steps, both targeting (7, 3).
        first = session.iloc[0]
        expected = math.hypot(7.2 - 7.0, 3.2 - 3.0)
        self.assertAlmostEqual(first["distance"], round(expected, 3))
        self.assertFalse(first["gap_warning"])
        self.assertAlmostEqual(first["distance"], 0.283)
        second = session.iloc[1]
        far = math.hypot(1.0 - 7.0, 1.0 - 3.0)
        self.assertGreater(far, GAP_WARNING)
        self.assertTrue(second["gap_warning"])
        self.assertIn("Few songs exist near this target", second["why"])

    def test_sparse_corner_flags_every_step(self):
        songs = _songs([(i, 5.0, 5.0) for i in range(1, 13)])
        session = build_session(SongRetriever(songs), Mood(1, 8), Mood(1, 9), allocation="iso_first")
        self.assertEqual(len(session), 10)
        self.assertTrue(session["gap_warning"].all())
        self.assertGreater(session["distance"].min(), GAP_WARNING)

    def test_insufficient_songs_raise(self):
        songs = _songs([(1, 5, 5), (2, 5, 6), (3, 6, 5)])
        retriever = SongRetriever(songs)
        with self.assertRaises(RuntimeError):
            build_session(retriever, Mood(5, 5), Mood(6, 6))
        with self.assertRaises(RuntimeError):
            build_playlist(retriever, ISO, Mood(5, 5), Mood(6, 6))

    def test_why_card_for_legacy_closest_song(self):
        # Exact songs on the (3, 7) -> (7, 3) iso waypoints, plus fillers for direct.
        rows = [(10, 3, 7), (11, 4, 6), (12, 5, 5), (13, 6, 4), (14, 7, 3)]
        rows += [(20 + i, 7, 3.5 + 0.1 * i) for i in range(5)]
        session = build_session(SongRetriever(_songs(rows)), Mood(3, 7), Mood(7, 3), allocation="iso_first")
        text = session.query("condition == 'iso' and step == 2")["why"].iloc[0]
        self.assertEqual(
            text,
            "Song 2 of 5. Target for this step: valence 4.0, arousal 6.0. "
            "This song is rated valence 4.0, arousal 6.0 by DEAM listeners. "
            "It was the closest unused song to the target (distance 0.00).",
        )
        self.assertEqual(list(session.columns), SESSION_COLUMNS)

    def test_joint_why_card_does_not_claim_an_independent_nearest_neighbor(self):
        songs = _songs([(i, 5.0, 5.0) for i in range(1, 13)])
        session = build_session(SongRetriever(songs), Mood(5, 5), Mood(5, 5))
        self.assertIn("assigned across both playlists", session.iloc[0]["why"])
        self.assertNotIn("closest unused", session.iloc[0]["why"])

    def test_output_does_not_depend_on_row_order(self):
        songs = _grid()
        shuffled = songs.sample(frac=1, random_state=7).reset_index(drop=True)
        base = build_session(SongRetriever(songs), Mood(2, 8), Mood(8, 2))
        again = build_session(SongRetriever(shuffled), Mood(2, 8), Mood(8, 2))
        pd.testing.assert_frame_equal(base, again)

    def test_repeated_calls_match(self):
        retriever = SongRetriever(_grid())
        first = build_session(retriever, Mood(3, 7), Mood(7, 3))
        second = build_session(retriever, Mood(3, 7), Mood(7, 3))
        pd.testing.assert_frame_equal(first, second)

    def test_empty_catalog_is_rejected(self):
        with self.assertRaises(ValueError):
            SongRetriever(_songs([]))


if __name__ == "__main__":
    unittest.main()
