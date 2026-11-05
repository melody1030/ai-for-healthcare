# IsoHarmonix Questionnaire

**Version:** 0.1 (draft for team review) · Goes with [`study_protocol.md`](study_protocol.md)

Use this text when building the Google Form or the app's form. The `variable_name` in
each item is the column name in the data files, so please keep them the same.

**Order in a session**

| Section | When | Times |
|---|---|---|
| 0. Consent | Start | 1 |
| 1. About you | Start | 1 |
| 2. Target mood | Before the first playlist | 1 |
| 3. Your mood right now | Before and after each playlist | 4 |
| 4. About this playlist | After each playlist | 2 |
| 5. Final questions | End | 1 |
| 6. Debrief | End (researcher reads) | 1 |

---

## Section 0. Consent

> **IsoHarmonix: music and everyday mood**
>
> We are students in the AI for Healthcare course at Cornell University. We are testing
> two ways of ordering songs in a playlist, to see how they affect everyday mood.
>
> **What you will do.** You will listen to two short playlists (about 4 minutes each)
> through headphones and answer short questions about your mood before and after each
> one. The session takes about 30 minutes.
>
> **Risks.** The risks are very low. The songs are ordinary music clips. If anything
> makes you uncomfortable, you can stop at any time.
>
> **This is not treatment.** This study does not diagnose or treat any health condition.
>
> **Your data.** We do not record your name with your answers. We use a participant
> number instead. Only the project team can see the data, and we only report combined,
> anonymous results. You can ask us to delete your data until 17 November 2026.
>
> **Your choice.** Taking part is voluntary. You can skip any question or stop at any
> time, with no problem at all.
>
> **Questions?** Contact the team: [team email].

| # | Item | Answer | variable_name |
|---|---|---|---|
| 0.1 | I am 18 years or older. | Yes / No | `age_18_plus` |
| 0.2 | I have read the information above and I agree to take part. | Yes / No | `consent` |

If either answer is "No", the session ends here, kindly.

---

## Section 1. About you

These questions help us check whether the system works equally well for different
people. You can skip any of them.

| # | Item | Answer options | variable_name |
|---|---|---|---|
| 1.1 | What is your age group? | 18-21 · 22-25 · 26-30 · 31-40 · over 40 · Prefer not to say | `age_group` |
| 1.2 | What is your gender? | Woman · Man · Non-binary · Prefer to self-describe: ___ · Prefer not to say | `gender` |
| 1.3 | Which ethnic background(s) best describe you? (choose all that apply) | Asian · Black or African American · Hispanic or Latino/a/x · Middle Eastern or North African · Native American or Alaska Native · Native Hawaiian or Pacific Islander · White · Other: ___ · Prefer not to say | `ethnicity` |
| 1.4 | How often do you listen to music to change or manage your mood? | Never · Rarely · Sometimes · Often · Almost every day | `music_for_mood` |
| 1.5 | How many hours of music do you listen to on a typical day? | Less than 1 · 1-2 · 2-4 · More than 4 | `music_hours` |

---

## Section 2. Target mood (once)

> Think about **how you would like to feel** after listening to some music today. Please
> set the two scales to that feeling.

| # | Item | Scale | variable_name |
|---|---|---|---|
| 2.1 | **Pleasant feeling you would like:** | 1 = very unhappy, sad · 5 = neutral · 9 = very happy, pleased | `target_valence` |
| 2.2 | **Energy you would like:** | 1 = very calm, relaxed, sleepy · 5 = neutral · 9 = very excited, alert, tense | `target_arousal` |

---

## Section 3. Your mood right now (4 times)

Asked **before** and **after** each playlist. Show the SAM pictures with the scales if
possible (Bradley & Lang, 1994).

> Please rate **how you feel right now**, at this moment. There are no right or wrong
> answers.

| # | Item | Scale | variable_name |
|---|---|---|---|
| 3.1 | **How pleasant do you feel?** | 1 = very unhappy, sad · 5 = neutral · 9 = very happy, pleased | `valence_before` / `valence_after` |
| 3.2 | **How much energy do you feel?** | 1 = very calm, relaxed, sleepy · 5 = neutral · 9 = very excited, alert, tense | `arousal_before` / `arousal_after` |

---

## Section 4. About this playlist (after each playlist)

> These questions are about the playlist you just heard.

| # | Item | Scale | variable_name |
|---|---|---|---|
| 4.1 | **Overall, how satisfied are you with this playlist?** | 1 = not satisfied at all · 4 = neutral · 7 = very satisfied | `satisfaction` |
| 4.2 | **How well did the songs fit how you were feeling?** | 1 = did not fit at all · 4 = neutral · 7 = fit very well | `perceived_fit` |
| 4.3 | Did you recognise any of the songs? | Yes / No / Not sure | `recognised_song` |

---

## Section 5. Final questions (once)

| # | Item | Answer | variable_name |
|---|---|---|---|
| 5.1 | Which playlist did you prefer? | Playlist A · Playlist B · No preference | `preferred_playlist` |
| 5.2 | Did you notice any difference between the two playlists? If yes, what was it? | Free text | `noticed_difference` |
| 5.3 | Is there anything else you would like to tell us? | Free text (optional) | `comments` |

Question 5.2 also tells us whether participants guessed the design (a blinding check).

---

## Section 6. Debrief (researcher reads)

> Thank you for taking part!
>
> **What we were testing.** One playlist started with music that matched your mood
> before the session and then changed step by step towards the mood you chose. This is
> called the "Iso-Principle" from music therapy. The other playlist went straight to the
> mood you chose. Playlist [A/B] was the step-by-step one. We want to know which one
> helps people move closer to the mood they want.
>
> [Show the why-cards for both playlists in the app.]
>
> Please do not tell other possible participants about the two kinds of playlists until
> the study ends (late November), because it could change their answers.
>
> **Support.** If you are feeling low or stressed, support is available. At Cornell you
> can contact Cornell Health Counseling and Psychological Services (CAPS). In the US you
> can also call or text **988** (Suicide & Crisis Lifeline) at any time.
>
> If you want your data deleted, contact us before 17 November 2026 with your
> participant number: **[P__]**.

---

## Researcher-entered fields (not shown to participants)

| variable_name | Meaning |
|---|---|
| `participant_id` | P01, P02, ... from `counterbalancing.csv` |
| `order` | `iso_first` or `direct_first` |
| `playlist_A_condition`, `playlist_B_condition` | `iso` or `direct` |
| `position` | `A` or `B` (in `ratings.csv`) |
| `condition` | `iso` or `direct` (in `ratings.csv`) |
| `session_date`, `researcher`, `start_time` | Session log |
| `close_to_target_flag` | `1` if the current mood was less than 2 away from the target at the start (protocol Section 11, point 6) |
| `notes` | Problems during the session |
