"""Selection-order bias, and the joint assignment that removes it.

These tests use fixed songs and fixed moods. They do not read DEAM.
"""

import math
import unittest

import pandas as pd

from isoharmonix.planner import Mood, SongRetriever, build_session, match_quality
from isoharmonix.rules import DIRECT, ISO


def _songs(rows):
    return pd.DataFrame(rows, columns=["song_id", "valence", "arousal"])


def _means(session):
    quality = match_quality(session)
    return float(quality.loc[ISO, "mean"]), float(quality.loc[DIRECT, "mean"])


def _raw_total(session, songs):
    positions = songs.set_index("song_id")
    total = 0.0
    for row in session.itertuples():
        total += math.hypot(
            positions.loc[row.song_id, "valence"] - row.target_valence,
            positions.loc[row.song_id, "arousal"] - row.target_arousal,
        )
    return total


class OrderBiasTests(unittest.TestCase):
    def setUp(self):
        # Perfect songs on the iso waypoints from (3, 7) to (7, 3), plus a
        # cluster around the target. Whoever is built first takes the best
        # target songs.
        self.songs = _songs(
            [
                (10, 3, 7),
                (11, 4, 6),
                (12, 5, 5),
                (13, 6, 4),
                (14, 7, 3),
                (20, 7.5, 3.5),
                (21, 8, 3),
                (22, 7, 4),
                (23, 6, 3),
                (24, 7, 2),
            ]
        )
        self.retriever = SongRetriever(self.songs)
        self.current = Mood(3, 7)
        self.target = Mood(7, 3)

    def test_iso_first_and_direct_first_select_different_songs(self):
        iso_first = build_session(self.retriever, self.current, self.target, allocation="iso_first")
        direct_first = build_session(
            self.retriever, self.current, self.target, allocation="direct_first"
        )
        self.assertEqual(
            list(iso_first.query("condition == 'iso'")["song_id"]),
            [10, 11, 12, 13, 14],
        )
        self.assertEqual(
            list(iso_first.query("condition == 'direct'")["song_id"]),
            [20, 21, 22, 23, 24],
        )
        self.assertEqual(
            list(direct_first.query("condition == 'direct'")["song_id"]),
            [14, 20, 21, 22, 23],
        )
        self.assertEqual(int(direct_first.query("condition == 'iso' and step == 5")["song_id"].iloc[0]), 24)

        iso_mean_first, direct_mean_first = _means(iso_first)
        iso_mean_second, direct_mean_second = _means(direct_first)
        self.assertLess(iso_mean_first, iso_mean_second)
        self.assertGreater(direct_mean_first, direct_mean_second)
        # Same songs are feasible either way, so the totals match. The harm
        # is which condition is given the closer songs.
        self.assertAlmostEqual(
            _raw_total(iso_first, self.songs),
            _raw_total(direct_first, self.songs),
            places=6,
        )

    def test_joint_assignment_is_order_independent_and_optimal(self):
        joint = build_session(self.retriever, self.current, self.target, allocation="joint")
        iso = joint[joint["condition"] == ISO]
        direct = joint[joint["condition"] == DIRECT]
        # Steps 1-4 have their own waypoints, so they keep the perfect songs.
        self.assertEqual(list(iso["song_id"].iloc[:4]), [10, 11, 12, 13])
        self.assertTrue((iso["distance"].iloc[:4] == 0).all())
        # One iso slot and five direct slots share the target. Direct has more
        # unfilled steps, so it receives the closest target song (14). Iso's
        # final step receives the next one (20), not whichever playlist ran first.
        self.assertEqual(int(iso["song_id"].iloc[4]), 20)
        self.assertEqual(list(direct["song_id"]), [14, 21, 22, 23, 24])
        self.assertTrue(joint["song_id"].is_unique)

        iso_first = build_session(self.retriever, self.current, self.target, allocation="iso_first")
        direct_first = build_session(
            self.retriever, self.current, self.target, allocation="direct_first"
        )
        joint_total = _raw_total(joint, self.songs)
        self.assertLessEqual(joint_total, _raw_total(iso_first, self.songs) + 1e-9)
        self.assertLessEqual(joint_total, _raw_total(direct_first, self.songs) + 1e-9)

    def test_equal_moods_do_not_give_one_condition_the_five_closest_songs(self):
        # Song k sits 0.1*k from (5, 5). Both playlists ask for that same point.
        songs = _songs([(k, 5 + 0.1 * k, 5) for k in range(1, 16)])
        retriever = SongRetriever(songs)
        mood = Mood(5, 5)
        iso_first = build_session(retriever, mood, mood, allocation="iso_first")
        direct_first = build_session(retriever, mood, mood, allocation="direct_first")
        joint = build_session(retriever, mood, mood, allocation="joint")

        def ids(session, condition):
            return list(session.loc[session["condition"] == condition, "song_id"])

        self.assertEqual(ids(iso_first, ISO), [1, 2, 3, 4, 5])
        self.assertEqual(ids(direct_first, ISO), [6, 7, 8, 9, 10])
        # Joint uses the ten closest songs and alternates them. direct wins
        # the single name tie, then the ranks alternate. Neither side gets
        # ranks 1-5 as a block.
        self.assertEqual(ids(joint, DIRECT), [1, 3, 5, 7, 9])
        self.assertEqual(ids(joint, ISO), [2, 4, 6, 8, 10])
        self.assertTrue(joint["song_id"].is_unique)

        iso_gap = abs(_means(iso_first)[0] - _means(direct_first)[0])
        joint_gap = abs(_means(joint)[0] - _means(joint)[1])
        self.assertGreater(iso_gap, 0.4)
        self.assertLess(joint_gap, 0.15)

    def test_joint_result_does_not_depend_on_catalog_order(self):
        songs = _songs([(k, 5 + 0.1 * k, 5) for k in range(1, 16)])
        shuffled = songs.sample(frac=1, random_state=1).reset_index(drop=True)
        mood = Mood(5, 5)
        first = build_session(SongRetriever(songs), mood, mood, allocation="joint")
        second = build_session(SongRetriever(shuffled), mood, mood, allocation="joint")
        pd.testing.assert_frame_equal(first, second)


class LegitimateDifferenceTests(unittest.TestCase):
    def test_direct_remains_farther_when_the_target_neighborhood_is_sparse(self):
        # Iso waypoints each have a perfect song. The target has only those
        # and a few neighbors, so direct still needs the more distant ones.
        # Joint allocation must not flatten that real difference to zero.
        songs = _songs(
            [
                (10, 3, 7),
                (11, 4, 6),
                (12, 5, 5),
                (13, 6, 4),
                (14, 7, 3),
                (21, 8, 3),
                (22, 7, 4),
                (23, 6, 3),
                (24, 7, 2),
                (25, 9, 1),
            ]
        )
        session = build_session(SongRetriever(songs), Mood(3, 7), Mood(7, 3), allocation="joint")
        iso_mean, direct_mean = _means(session)
        self.assertLess(iso_mean, direct_mean)


if __name__ == "__main__":
    unittest.main()
