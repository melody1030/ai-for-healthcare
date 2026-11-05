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
| 10/13 to 10/27 | Questionnaire + study protocol | D | **Draft ready for team review** ([protocol](docs/study_protocol.md), [questionnaire](docs/questionnaire.md), [counterbalancing](study/counterbalancing.csv)) |
| 10/27 | Integration + pilot (checkpoint) | All | To do |
| 11/3 to 11/17 | User study sessions | C, D | To do |
| 11/17 to 12/1 | Analysis, explainability, bias | A, B | To do |
| 12/1 to 12/10 | Final report + presentation | All | To do |

## Setup

```bash
pip install -r requirements.txt
```

Download the [DEAM dataset from Kaggle](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music)
and unzip it into `data/DEAM/`. The exact folder layout inside does not matter:
the loader searches for these files:

- `static_annotations_averaged_songs_1_2000.csv`
- `static_annotations_averaged_songs_2000_2058.csv`
- the MP3 clips, named `<song_id>.mp3`

`data/` is in `.gitignore`, so the dataset is never committed.

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

## How it works

1. **Mood input:** current and target mood, each a (valence, arousal) point on 1 to 9.
2. **Path planner**
   - *Iso-guided:* 5 evenly spaced points from the current mood to the target. Song 1 matches, song 5 reaches the target.
   - *Direct-to-target:* all 5 points at the target.
3. **Song retrieval:** for each point, the closest unused DEAM song by Euclidean distance (scikit-learn `NearestNeighbors`).
   In one session, no song appears in both playlists. The iso playlist is built first, then the direct playlist uses the songs that are left.
4. **Why-card:** each song comes with a sentence like
   *"Song 2 of 5. Target for this step: valence 4.0, arousal 6.0. This song is rated valence 4.0, arousal 5.9 by DEAM listeners. It was the closest unused song to the target (distance 0.10)."*

All fixed rules (5 steps, Euclidean distance, spread cutoff, gap warning) are in
the "Fixed rules" cell in Part 0 of the notebook. They should be frozen before the pilot.

## Data cleaning rules

| Issue (from proposal) | What the code does |
|---|---|
| Songs missing a rating or audio file | Dropped. The audio check runs only when MP3 files are found. |
| Raters disagree (possible mislabels) | Spread = (valence_std + arousal_std) / 2. Songs above the 90th percentile are dropped. |
| Songs uneven across the mood space | Coverage plot and gap table, plus a `gap_warning` flag on any song more than 1.0 from its target. |

## First results (DEAM ratings only, before adding audio)

From the notebook:

- 1,802 raw songs, 179 dropped for high rater spread (spread > 1.885), **1,623 clean songs**.
- Median gap from an integer slider setting to the closest song: 0.63. Only **64%** of the 81 slider settings have a song within 1.0.
- The corners are empty. The worst targets are (1, 9) "very tense", gap 3.07, and (9, 1) "very calm and happy", gap 2.75. Most songs lie on a diagonal from sad/calm to happy/excited.

![Coverage plot](results/coverage_plot.png)

Example session, current mood (3, 7) "tense", target (7, 3) "calm and content":

![Example playlists](results/example_paths.png)

## Points for the team to decide before the pilot

Our proposal says all rules are **fixed before the study**. If we change them after
seeing the results, people could say we chose the rules that gave a nice answer. So we
need to agree on these points once, write them in the notebook's "Fixed rules" cell, and
not change them after the pilot.

**Two basic ideas first**
- **Every song is a dot on a 9 × 9 map:** valence (1 = sad, 9 = happy) and arousal
  (1 = calm, 9 = excited). The user's mood is also a dot on this map.
- **Distance = how good the match is.** The system picks the closest song to each
  target point. Small distance = good match. Big distance = poor match.

### 1. The two playlists do not match equally well

- **Iso:** the 5 songs aim at 5 *different* points, so each one gets the closest song.
- **Direct:** all 5 songs aim at the *same* point and no song can repeat, so it gets the
  1st to 5th closest songs. The later ones are further away.
- Example, tense (3, 7) to calm (7, 3): mean distance **0.15 for iso, 0.57 for direct**.
  Over 300 random moods: **0.57 vs 1.10**.

**Why it matters:** if iso works better, someone could say "the iso songs were just
chosen more accurately". This second explanation is a **confound**.

**Proposal:** we cannot fully remove it (it comes from the design), so we (1) save the
match distance for every participant, (2) run the main test again without participants
whose direct playlist has a mean distance above 1.0, and (3) report it as a limitation.

### 2. Some moods have almost no songs

Most songs lie on a diagonal from "sad and calm" to "happy and excited". Two corners are
almost empty: (1, 9) very sad and tense (closest song 3.07 away) and (9, 1) very happy
and calm (2.75 away).

| Slider range | Moods with a song within 1.0 | Mean distance, iso | Mean distance, direct |
|---|---|---|---|
| 1 to 9 (now) | 64% | 0.57 | 1.10 |
| **2 to 8** | **90%** | **0.24** | **0.55** |
| 3 to 7 | 100% | 0.09 | 0.23 |

**Proposal:** sliders from 2 to 8 in the app. It makes both playlists match much better
and also shrinks the gap in point 1. 3 to 7 is too narrow: a very stressed person could
not report how they really feel. (The questionnaire keeps the full 1 to 9 scale.)

### 3. Which songs to remove because raters disagreed

Each DEAM song was rated by several people. When they disagree a lot, we are not sure
where the song belongs on the map. Our rule removes the 10% of songs with the most
disagreement (90th percentile cutoff).

| Rule | Songs left | Moods with a song within 1.0 |
|---|---|---|
| Remove 20% | 1,441 | 64% |
| **Remove 10% (now)** | **1,623** | **64%** |
| Remove 5% | 1,711 | 65% |
| Remove none | 1,802 | 67% |

**Proposal:** keep "remove 10%". The numbers hardly change, so this choice is low risk.

### Questions for the team

1. Sliders from 2 to 8: yes or no?
2. Save match distance and re-run the test without poorly matched participants: yes or no?
3. Keep removing the 10% of songs with the most rater disagreement: yes or no?

Three more decisions about the study session (break length, hiding the why-card, moods
very close to the target) are in [Section 11 of the study protocol](docs/study_protocol.md#11-decisions-to-confirm-before-the-pilot).
