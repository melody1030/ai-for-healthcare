# IsoHarmonix Study Protocol

**Version:** 0.1 (draft for team review) · **Owner:** Melody (D) · **Timeline task:** Questionnaire + study protocol (10/13 to 10/27)

This document says exactly how every study session runs, so that all sessions are the
same no matter who runs them. The questionnaire items are in
[`questionnaire.md`](questionnaire.md).

> **Freeze rule.** After the pilot (week of 10/27), this protocol and the rules in the
> notebook's "Fixed rules" cell are frozen. Any later change must be written in the
> change log at the end of this document, with the date and the reason.

---

## 1. Research question and hypothesis

**Research question.** Does a match-then-shift (iso-guided) playlist move self-reported
mood closer to a desired state than a direct-to-target playlist?

**Hypothesis (H1).** The target-distance change is larger after the iso-guided playlist
than after the direct-to-target playlist.

**Secondary questions.**
- Do participants rate the two playlists differently for satisfaction and perceived fit?
- Does the effect come mainly from a change in valence, in arousal, or both?

This is a **non-clinical** study about everyday mood. We do not diagnose or treat
anything, and we do not make any health claims.

## 2. Design

| Item | Plan |
|---|---|
| Design | Within-subject: every participant hears both playlists. |
| Conditions | **Iso-guided:** 5 songs from the current mood to the target mood. **Direct-to-target:** 5 songs at the target mood. |
| Order | Counterbalanced: half of the participants hear iso first, half hear direct first. Assignment is fixed in advance in [`study/counterbalancing.csv`](../study/counterbalancing.csv): random blocks of 4 (2 iso first, 2 direct first, fixed seed), so the groups stay equal at 12, 16 or 20 people. Pilot IDs are in `study/counterbalancing_pilot.csv`. |
| Blinding | Participants hear "Playlist A" and "Playlist B". We do not tell them which one is iso until the debrief. |
| Songs | 5 DEAM clips per playlist, 45 seconds each (about 4 minutes per playlist). No song is played twice in one session. |
| Sample | 12 to 20 adults. |
| Session length | About 30 minutes. |

## 3. Participants

**Inclusion criteria**
- 18 years or older.
- Can hear music through headphones at a normal volume.
- Can read and understand English instructions.

**Exclusion criteria**
- Took part in the pilot (pilot data is not used in the main analysis).
- Is a member of the project team.

**Recruitment.** Classmates, friends and other students, by message or in person. This is
a convenience sample, which we will state as a limitation. Participation is voluntary and
unpaid. Saying no has no effect on anyone's grade or relationship with the team.

## 4. Ethics and data protection

- **Approval first.** Before the first real session, ask the course instructor whether
  this study needs review or an exemption from the Cornell IRB. Do not start the main
  study until we have an answer. (This is one of the roadblocks in our proposal, so we
  should ask in the week of 10/13.)
- **Consent.** Every participant reads the consent text (questionnaire Section 0) and
  agrees before any data is collected.
- **Right to stop.** Participants can stop at any time, skip any question, and ask us to
  delete their data until the analysis starts (11/17).
- **Low risk.** The music is ordinary instrumental and popular-style clips. We do not try
  to make anyone feel sad or stressed (no mood induction).
- **If someone is upset.** Stop the session, be kind, and give the support information in
  the debrief (Section 6 of the questionnaire). Write a short note in the session log.
- **Anonymous data.** Data files use only a participant ID (P01, P02, ...). Names and
  consent records are kept in a separate private file that is not in the GitHub repo.
- **Storage.** Study data stays in a private shared folder for the team. Only anonymous,
  combined results go into the repo and the final report.

## 5. Materials and setup

- One laptop with the IsoHarmonix app (Tiffany's Streamlit app) and the cleaned DEAM clips.
- One pair of over-ear headphones, the same pair for every participant.
- The questionnaire (Google Form or the app's own form; see questionnaire.md).
- The session checklist (Section 6) and this participant's row in `counterbalancing.csv`.
- A quiet room with no other people talking.

**Before each session**
1. Charge the laptop and close notifications.
2. Open the app and check that audio plays.
3. Check the participant ID and order (A = iso or A = direct) in `counterbalancing.csv`.
4. In the app, set the study mode so that the **why-card is hidden** (it shows "step 2 of
   5" and the target values, which would tell the participant which playlist is iso).

## 6. Session procedure

Times are approximate. The researcher reads the **bold** sentences out loud, word for
word, so every participant gets the same instructions.

| # | Step | Time | Who |
|---|---|---|---|
| 1 | Welcome and consent (Section 0). | 3 min | Researcher reads, participant agrees |
| 2 | Background questions (Section 1). | 2 min | Participant |
| 3 | Volume check: play a short test sound that is **not** a DEAM study song, and let the participant set a comfortable volume. After this, the volume is not changed. | 1 min | Researcher |
| 4 | Choose the target mood (Section 2). **"Think about how you would like to feel after listening to some music today. Please set the two sliders to that feeling."** This target is the same for both playlists. | 1 min | Participant |
| 5 | Mood before Playlist A (Section 3, items 3.1 and 3.2). | 1 min | Participant |
| 6 | The app builds Playlist A from this mood and the target, using the order in `counterbalancing.csv`. **"Please relax and listen to the music. Try not to use your phone."** | 4 min | App |
| 7 | Mood after Playlist A, then playlist ratings (Section 4). | 2 min | Participant |
| 8 | Break: **"Now please rest quietly for three minutes. There will be no music."** Use a timer. | 3 min | Researcher |
| 9 | Mood before Playlist B (Section 3 again). | 1 min | Participant |
| 10 | The app builds Playlist B from this new mood and the same target. | 4 min | App |
| 11 | Mood after Playlist B, then playlist ratings (Section 4 again). | 2 min | Participant |
| 12 | Final questions (Section 5). | 2 min | Participant |
| 13 | Debrief (Section 6): explain the two playlists, show the why-cards, give support information, say thank you. | 3 min | Researcher |
| 14 | Session log: write the date, start time, researcher, and any problems. | 1 min | Researcher |

**Why each playlist uses its own "before" mood.** Mood may carry over from the first
playlist to the second (a roadblock in our proposal). The 3-minute break reduces this,
and measuring mood again before Playlist B means each playlist starts from the
participant's real current mood, which is exactly how the app is meant to work.

**Rules during the session**
- Do not talk during the music.
- Do not explain the iso idea until the debrief.
- If the app crashes, write down the step, restart, and continue from the last completed
  step. Note it in the session log.
- If a song does not play, skip it and note the song ID.

## 7. Measures

All mood ratings use the Self-Assessment Manikin (SAM) style 1 to 9 scales (Bradley &
Lang, 1994). Full wording is in questionnaire.md.

| Measure | When | Scale |
|---|---|---|
| Target valence, target arousal | Once, step 4 | 1 to 9 |
| Valence, arousal (before) | Before each playlist | 1 to 9 |
| Valence, arousal (after) | After each playlist | 1 to 9 |
| Satisfaction | After each playlist | 1 to 7 |
| Perceived fit | After each playlist | 1 to 7 |
| Recognised any song | After each playlist | yes / no |
| Preferred playlist, open comment | End | A / B / no preference, free text |
| Gender, ethnic background, age range, music habits | Start | categories |

**Primary outcome: target-distance change** (for each playlist)

```
distance_before = sqrt((valence_before - target_valence)^2 + (arousal_before - target_arousal)^2)
distance_after  = sqrt((valence_after  - target_valence)^2 + (arousal_after  - target_arousal)^2)
target_distance_change = distance_before - distance_after
```

A bigger number means the participant moved closer to the mood they wanted.

**Also saved automatically by the app:** the 5 song IDs of each playlist, each song's
target point and distance, and `match_quality()` (mean and max distance per playlist).

## 8. Analysis plan (fixed before the study)

1. **Primary test.** Compare `target_distance_change` between iso and direct within each
   participant, using a two-sided Wilcoxon signed-rank test, α = 0.05. Report the median
   difference and the effect size r = Z / √N.
2. **Secondary tests.** The same test for satisfaction and perceived fit. Also describe
   the valence change and arousal change separately. Secondary results are exploratory.
3. **Order check.** Compare the iso minus direct difference between the two order groups
   (Mann-Whitney U). If there is a clear order effect, report it.
4. **Match-quality check (sensitivity analysis).** Run the primary test again without
   participants whose direct playlist has a mean distance above 1.0. If the result
   changes, report both versions.
5. **Bias checks.** Describe results by gender and by ethnic background. With 12 to 20
   people these are descriptive only, not conclusive.
6. **Missing data.** If a participant stops before finishing both playlists, they are
   left out of the paired tests, and we report how many.

## 9. Data files

All files use one row per record and the variable names in questionnaire.md. Empty
templates are in [`study/templates/`](../study/templates/).

| File | One row per | Content |
|---|---|---|
| `participants.csv` | participant | ID, order, target mood, background, final questions |
| `ratings.csv` | participant × playlist | condition, position (A/B), mood before/after, satisfaction, fit, recognised song |
| `songs_played.csv` | participant × song | output of `build_session()`: condition, step, target point, song ID, distance |
| `session_log.csv` | session | date, researcher, start time, problems |

## 10. Pilot (week of 10/27, integration checkpoint)

Run the full session with 2 or 3 people who are not in the main study. Check:

- Does the whole session fit in about 30 minutes?
- Are the instructions and questions easy to understand?
- Does the app save all data files correctly?
- Do participants guess which playlist is the "special" one?
- Do the slider limits (if we use 2 to 8) feel too narrow?

After the pilot, fix any problems, write them in the change log, and **freeze** the
protocol.

## 11. Decisions to confirm before the pilot

| # | Decision | Proposal |
|---|---|---|
| 1 | Slider range for the app | 2 to 8 (see notebook analysis). The questionnaire keeps 1 to 9. |
| 2 | Match-quality sensitivity analysis | Yes, cutoff 1.0 (Section 8, point 4). |
| 3 | Rater-spread cutoff | Keep the 90th percentile. |
| 4 | Break length between playlists | 3 minutes of quiet rest. |
| 5 | Why-card during the study | Hidden during listening, shown at the debrief. |
| 6 | Current mood very close to the target | If the distance is below 2, ask the participant if this really is the mood they want. If yes, continue and flag it in the data. |

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-05 | Version 0.1 drafted | |
