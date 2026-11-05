import numpy as np
import pandas as pd
import pytest

from isoharmonix.planner import (
    DIRECT,
    ISO,
    Mood,
    SongRetriever,
    build_playlist,
    build_session,
    direct_path,
    iso_path,
)


@pytest.fixture
def grid_songs():
    """One song on every integer point of the 1-9 grid (81 songs)."""
    v, a = np.meshgrid(np.arange(1, 10), np.arange(1, 10))
    return pd.DataFrame(
        {"song_id": np.arange(81) + 100, "valence": v.ravel().astype(float), "arousal": a.ravel().astype(float)}
    )


def test_iso_path_matches_then_reaches_target():
    path = iso_path(Mood(3, 7), Mood(7, 3), n_steps=5)
    assert path.shape == (5, 2)
    np.testing.assert_allclose(path[0], [3, 7])
    np.testing.assert_allclose(path[-1], [7, 3])
    np.testing.assert_allclose(np.diff(path, axis=0), [[1, -1]] * 4)  # even steps


def test_direct_path_all_at_target():
    np.testing.assert_allclose(direct_path(Mood(7, 3), 5), [[7, 3]] * 5)


def test_mood_rejects_out_of_range():
    with pytest.raises(ValueError):
        Mood(0, 5)
    with pytest.raises(ValueError):
        Mood(5, 9.5)


def test_iso_playlist_picks_exact_grid_songs(grid_songs):
    r = SongRetriever(grid_songs)
    pl = build_playlist(r, ISO, Mood(3, 7), Mood(7, 3))
    assert pl[["song_valence", "song_arousal"]].values.tolist() == [[3, 7], [4, 6], [5, 5], [6, 4], [7, 3]]
    assert (pl["distance"] == 0).all()
    assert pl["why"].iloc[1].startswith("Song 2 of 5.")


def test_direct_playlist_has_no_repeats(grid_songs):
    r = SongRetriever(grid_songs)
    pl = build_playlist(r, DIRECT, Mood(3, 7), Mood(7, 3))
    assert pl["song_id"].is_unique
    assert pl["distance"].iloc[0] == 0
    assert (pl["distance"].iloc[1:] == 1).all()  # next closest are the 4 neighbours


def test_session_no_song_in_both_playlists(grid_songs):
    s = build_session(SongRetriever(grid_songs), Mood(3, 7), Mood(7, 3))
    assert len(s) == 10
    assert s["song_id"].is_unique
    assert s.groupby("condition", sort=False).size().tolist() == [5, 5]
    assert s["condition"].tolist() == [ISO] * 5 + [DIRECT] * 5
    # iso is built first, so it gets the exact grid songs
    assert (s.loc[s.condition == ISO, "distance"] == 0).all()


def test_gap_warning_when_no_close_song():
    songs = pd.DataFrame({"song_id": [1, 2], "valence": [1.0, 9.0], "arousal": [1.0, 9.0]})
    pl = build_playlist(SongRetriever(songs), ISO, Mood(1, 1), Mood(9, 9), n_steps=2)
    assert not pl["gap_warning"].any()
    pl = build_playlist(SongRetriever(songs), DIRECT, Mood(5, 5), Mood(5, 5), n_steps=1)
    assert pl["gap_warning"].iloc[0]
    assert "less close" in pl["why"].iloc[0]


def test_runs_out_of_songs():
    songs = pd.DataFrame({"song_id": [1, 2], "valence": [1.0, 9.0], "arousal": [1.0, 9.0]})
    with pytest.raises(RuntimeError):
        build_playlist(SongRetriever(songs), DIRECT, Mood(5, 5), Mood(5, 5), n_steps=3)
