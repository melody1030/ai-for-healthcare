"""Validate local DEAM audio and prepare the participant-study manifest."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isoharmonix.data import clean_songs, load_annotations, resolve_deam_dir  # noqa: E402
from isoharmonix.study_data import prepare_audio  # noqa: E402


def main() -> None:
    """Prepare and summarize the validated DEAM study data."""
    data_dir = resolve_deam_dir(ROOT)
    raw_songs = load_annotations(data_dir)
    audio, manifest_path = prepare_audio(ROOT, data_dir)
    songs, _ = clean_songs(
        raw_songs,
        audio.playable,
        require_audio=audio.files_found > 0,
        unreadable_ids=audio.unreadable_ids,
    )
    print(f"DEAM directory: {data_dir}")
    print(f"Total MP3 files: {audio.files_found}")
    print(f"Playable files: {len(audio.playable)}")
    print(f"Unreadable files: {len(audio.unreadable_files)}")
    print(f"Final eligible songs: {len(songs)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
