# IsoHarmonix

An adaptive, Iso-Principle music assistant for everyday mood regulation.
Team 9: Alison Wu, Lisa Wang, Tiffany Chiang, Melody Huang (AI for Healthcare, Fall 2026).

The first song matches the listener's current mood, and each next song moves one
step closer to the target mood. We compare this **iso-guided** playlist with a
**direct-to-target** playlist (all songs at the target mood).

## Timeline progress

| Weeks | Task | Owner | Status |
|---|---|---|---|
| 9/29 | Proposal | All | Done |
| 10/6 to 10/20 | DEAM loading, cleaning, coverage plot | A | **Done** (notebook Parts 1 to 3) |
| 10/6 to 10/20 | Path planner + song retrieval | B | **Done** (notebook Parts 4 to 6) |
| 10/13 to 10/27 | Simple app: sliders, playback, why-card | C | To do (the why-card text is ready, see below) |
| 10/13 to 10/27 | Questionnaire + study protocol | D | To do |
| 10/27 | Integration + pilot (checkpoint) | All | To do |
| 11/3 to 11/17 | User study sessions | C, D | To do |
| 11/17 to 12/1 | Analysis, explainability, bias | A, B | To do |
| 12/1 to 12/10 | Final report + presentation | All | To do |

## Setup

```bash
pip install -r requirements.txt
```

Download the [DEAM dataset from Kaggle](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music)
and unzip it into `data/DEAM/`, or set `DEAM_DATA_DIR` to that folder.
The exact folder layout inside does not matter: the loader searches for these files:

- `static_annotations_averaged_songs_1_2000.csv`
- `static_annotations_averaged_songs_2000_2058.csv`
- the MP3 clips, named `<song_id>.mp3` (leading zeros and `.MP3` are accepted)

`data/` and audio files are in `.gitignore`, so the dataset is never committed.
Nothing in the code hardcodes a machine-specific path, and the library does not download DEAM.

Playlist checks that do not need the dataset:

```bash
python -m unittest discover -s tests -v
```

Tests that need the real annotations or MP3s skip when `data/DEAM/` is absent. A skip means that check was not verified.

## Usage

All the work is in one notebook:
[`notebooks/IsoHarmonix_data_and_planner.ipynb`](notebooks/IsoHarmonix_data_and_planner.ipynb)

```bash
jupyter notebook notebooks/IsoHarmonix_data_and_planner.ipynb
```

Run all cells from top to bottom. The notebook also works in Google Colab: if
`data/DEAM/` is not found, it tries to download DEAM with `kagglehub` (needs a Kaggle
API token).

| Part | Content | Output |
|---|---|---|
| 0 | Setup and fixed rules | |
| 1 to 2 | Load and clean DEAM | `data/songs_clean.csv` |
| 3 | Coverage plot and gap table | `results/coverage_plot.png`, `results/coverage_grid.csv` |
| 4 to 5 | Path planner and song retrieval | `results/example_session.csv`, `results/example_paths.png` |
| 6 | Checks on a toy song set | |
| 7 to 8 | Match quality and points for the team | |

**For the app (Tiffany):** `build_session(retriever, Mood(3, 7), Mood(7, 3))` returns
one row per song with `condition` ("iso"/"direct"), `step`, the step's target, the
song's DEAM ratings, `distance`, `gap_warning`, `audio_path` and `why` (the why-card text).
That call uses `allocation="joint"`. Pass `allocation="iso_first"` to reproduce the original rule.

## How it works

1. **Mood input:** current and target mood, each a (valence, arousal) point on 1 to 9.
2. **Path planner**
   - *Iso-guided:* 5 evenly spaced points from the current mood to the target. Song 1 matches, song 5 reaches the target.
   - *Direct-to-target:* all 5 points at the target.
3. **Song retrieval:** for each waypoint, a DEAM song is chosen by Euclidean distance. No song is repeated inside a session.
   The default, `allocation="joint"`, assigns all 10 songs together by minimum total distance (`scipy.optimize.linear_sum_assignment`). Build order does not decide which playlist gets the closer songs. When several steps share a target, those songs are dealt according to how many steps each condition still has open.
   `allocation="iso_first"` is the original rule: the iso playlist is built first, then the direct playlist uses the songs that are left. The checked-in `results/example_session.csv` was produced that way, with empty `audio_path`s, before any MP3s were available.
   **The pilot should not start on `joint` until the team agrees.** Exact distance ties break toward the smaller `song_id`.
4. **Why-card:** each song comes with a sentence like
   *"Song 2 of 5. Target for this step: valence 4.0, arousal 6.0. This song is rated valence 4.0, arousal 5.9 by DEAM listeners. It was the closest unused song to the target (distance 0.10)."*
   That "closest unused song" wording is the original iso-first card. `joint` instead says the song was assigned across both playlists so that no song is repeated.

All fixed rules (5 steps, Euclidean distance, spread cutoff, gap warning) are in
the "Fixed rules" cell in Part 0 of the notebook. They should be frozen before the pilot.

## Data cleaning rules

| Issue (from proposal) | What the code does |
|---|---|
| Songs missing a rating, or missing or unreadable audio | Dropped when MP3s are found. A file counts as readable only if a decoder can open it (`ffprobe`, otherwise macOS `afinfo`). If no MP3s are found, songs are kept for ratings-only checks and are not playback-ready. |
| Raters disagree (possible mislabels) | Spread = (valence_std + arousal_std) / 2. Songs above the 90th percentile are dropped. |
| Songs uneven across the mood space | Coverage plot and gap table, plus a `gap_warning` flag on any song more than 1.0 from its target. |

## First results (DEAM ratings only, before adding audio)

From the notebook:

- 1,802 raw songs, 179 dropped for high rater spread (spread > 1.885), **1,623 clean songs**.
- Median gap from an integer slider setting to the closest song: 0.63. Only **64%** of the 81 slider settings have a song within 1.0.
- The corners are empty. The worst targets are (1, 9) "very tense", gap 3.07, and (9, 1) "very calm and happy", gap 2.75. Most songs lie on a diagonal from sad/calm to happy/excited.

![Coverage plot](results/coverage_plot.png)

Example session, current mood (3, 7) "tense", target (7, 3) "calm and content". This figure and `results/example_session.csv` are the original iso-first run. Re-running Part 5 overwrites the CSV with a joint session:

![Example playlists](results/example_paths.png)

## Points for the team to decide before the pilot

1. **Match quality differs between conditions.** The direct playlist needs 5 songs
   at the same point, so its songs are further from their target. Over 300 random
   mood pairs, with the original iso-first rule and ratings only, the mean distance
   was **0.57 for iso and 1.10 for direct**. Part of a gap like that is the waypoint
   design, and part can be build order. Report `match_quality()` for every participant.
   Those two numbers were not recomputed here: the clean DEAM table is not in the repo.
2. **Allocation rule (needs approval).** `build_session` now defaults to `joint`.
   On a fixed synthetic diagonal catalog (800 songs, seed 0; 300 integer mood pairs, seed 1),
   mean distances were iso-first 0.962 / 1.432, direct-first 1.042 / 1.341, and joint 0.985 / 1.388 (iso / direct).
   The song list changed in all 300 pairs when the greedy order was reversed. The direct playlist stayed farther on average, which is the waypoint difference, not the order effect.
   On the 81 cases where the current mood equals the target, iso-first and direct-first swap exactly (1.363 vs 1.546). Joint brings those to 1.480 iso / 1.429 direct. The leftover gap is the documented name tie: `direct` wins the single closest shared song, then the songs alternate.
3. **Slider limits.** Because the corners have no songs, we could limit the sliders
   (for example to 2 to 8), or show the gap warning in the app.
4. **Spread cutoff.** 90th percentile is our current choice. It is computed after the audio filter, so it can move once MP3s are present. Please confirm before we freeze it.
