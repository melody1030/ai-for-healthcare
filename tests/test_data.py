from pathlib import Path

import pandas as pd
import pytest

from isoharmonix.coverage import coverage_table, nearest_gap
from isoharmonix.data import clean_songs, find_audio, load_annotations


def _write_fake_deam(root: Path) -> None:
    """Tiny DEAM-like folder with the same quirky headers as the real files."""
    ann = root / "DEAM_Annotations" / "annotations" / "song_level"
    ann.mkdir(parents=True)
    (ann / "static_annotations_averaged_songs_1_2000.csv").write_text(
        "song_id, valence_mean, valence_std, arousal_mean, arousal_std\n"
        "2,3.1,0.94,3,0.63\n"
        "3,3.5,1.75,3.3,1.62\n"
        "4,5.0,0.5,5.0,0.5\n"
        "5,,0.5,5.0,0.5\n"
    )
    (ann / "static_annotations_averaged_songs_2000_2058.csv").write_text(
        "song_id, valence_mean, valence_std, valence_ max_mean, valence_max_std, "
        "valence_min_mean, valence_min_std, arousal_mean, arousal_std, arousal_max_mean, "
        "arousal_max_std, arousal_min_mean, arousal_min_std\n"
        "2001,3.2,0.98,5.0,1.41,2.2,0.98,6.6,0.8,8.6,0.49,3.4,1.02\n"
    )
    audio = root / "DEAM_audio" / "MEMD_audio"
    audio.mkdir(parents=True)
    for sid in (2, 3, 5, 2001):
        (audio / f"{sid}.mp3").write_bytes(b"")


def test_load_annotations_merges_both_files(tmp_path):
    _write_fake_deam(tmp_path)
    songs = load_annotations(tmp_path)
    assert list(songs.columns) == ["song_id", "valence", "arousal", "valence_std", "arousal_std"]
    assert songs["song_id"].tolist() == [2, 3, 4, 5, 2001]
    assert songs.set_index("song_id").loc[2001, "arousal"] == 6.6


def test_clean_drops_missing_rating_audio_and_high_spread(tmp_path):
    _write_fake_deam(tmp_path)
    songs = load_annotations(tmp_path)
    clean, rep = clean_songs(songs, find_audio(tmp_path), spread_quantile=0.5)
    assert rep.n_missing_rating == 1      # song 5
    assert rep.n_missing_audio == 1       # song 4
    assert rep.n_high_spread == 1         # song 3 has the largest spread
    assert clean["song_id"].tolist() == [2, 2001]
    assert rep.n_clean == 2


def test_clean_without_audio_skips_audio_check(tmp_path):
    _write_fake_deam(tmp_path)
    songs = load_annotations(tmp_path)
    _, rep = clean_songs(songs, None, spread_quantile=1.0)
    assert rep.n_missing_audio == 0
    assert rep.n_clean == 4
    assert rep.notes


def test_missing_files_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_annotations(tmp_path)


def test_coverage_gap():
    songs = pd.DataFrame({"song_id": [1], "valence": [5.0], "arousal": [5.0]})
    assert nearest_gap(songs, [[5, 5], [8, 9]]).tolist() == [0.0, 5.0]
    table = coverage_table(songs, step=1.0)
    assert len(table) == 81
    assert table["covered"].sum() == 5  # the centre and its 4 direct neighbours
