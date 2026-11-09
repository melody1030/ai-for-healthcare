"""Persistent preparation helpers for the participant study data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from isoharmonix.data import AudioScan, decoder_name, scan_audio


class PreparedAudioError(RuntimeError):
    """Raised when validated study audio is not prepared or is stale."""


def audio_manifest_path(root: Path) -> Path:
    """Return the ignored manifest path for a repository."""
    return Path(root) / "data" / ".isoharmonix_audio_scan.json"


def dataset_fingerprint(data_dir: Path) -> str:
    """Fingerprint annotations and audio metadata without decoding audio."""
    files = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mp3", ".csv"}:
            continue
        stat = path.stat()
        files.append(
            (
                str(path.relative_to(data_dir)),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    payload = {
        "data_dir": str(data_dir.resolve()),
        "decoder": decoder_name(),
        "files": files,
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def write_audio_manifest(root: Path, data_dir: Path, audio: AudioScan) -> Path:
    """Persist a validated audio scan and return its manifest path."""
    manifest_path = audio_manifest_path(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "data_dir": str(data_dir.resolve()),
                "fingerprint": dataset_fingerprint(data_dir),
                "files_found": audio.files_found,
                "playable": {
                    str(song_id): str(path)
                    for song_id, path in audio.playable.items()
                },
                "unreadable_ids": list(audio.unreadable_ids),
                "unreadable_files": [str(path) for path in audio.unreadable_files],
                "duplicate_ids": list(audio.duplicate_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest_path


def load_prepared_audio(root: Path, data_dir: Path) -> AudioScan:
    """Load the researcher-prepared manifest without rescanning the dataset."""
    manifest_path = audio_manifest_path(root)
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("data_dir") != str(data_dir.resolve()):
            raise PreparedAudioError(
                f"Study audio manifest is for a different DEAM directory: {manifest_path}"
            )
        return AudioScan(
            files_found=int(manifest["files_found"]),
            playable={
                int(song_id): Path(path)
                for song_id, path in manifest["playable"].items()
            },
            unreadable_ids=tuple(manifest["unreadable_ids"]),
            unreadable_files=tuple(
                Path(path) for path in manifest["unreadable_files"]
            ),
            duplicate_ids=tuple(manifest["duplicate_ids"]),
        )
    except PreparedAudioError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise PreparedAudioError(
            f"Study audio has not been prepared yet: {manifest_path}"
        ) from error


def prepare_audio(root: Path, data_dir: Path) -> tuple[AudioScan, Path]:
    """Run the source-of-truth audio scan and persist its result."""
    audio = scan_audio(data_dir)
    return audio, write_audio_manifest(root, data_dir, audio)
