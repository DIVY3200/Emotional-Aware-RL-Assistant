# User study protocol

A small, honest study (8–12 participants) is enough to turn "it works on my
face" into evidence. This protocol is designed so that one person can run it
with friends or classmates in about 25 minutes per participant.

> If you plan to publish the results (paper, thesis chapter), ask your
> department about ethics approval **before** collecting data.

## Question and hypotheses

Does the learned policy help more than a hand-written rule or a random policy?

- **H1** — Responses chosen by `ppo_personalised` are followed by a larger
  improvement in detected valence than those of `rule_cooldown` and `random`.
- **H2** — Participants rate `ppo_personalised` as more helpful and less
  intrusive than the other two conditions.

## Design

- **Within-subject**: every participant experiences all three conditions.
- **Blinded**: conditions are run by code (`--condition A|B|C`); the mapping
  lives in `configs/default.yaml` and is not shown to participants.
- **Counterbalanced** order (Latin square) to cancel learning and fatigue effects:

| Participant | 1st | 2nd | 3rd |
|---|---|---|---|
| P1, P4, P7, P10 | A | B | C |
| P2, P5, P8, P11 | B | C | A |
| P3, P6, P9, P12 | C | A | B |

## Procedure (per participant)

1. **Consent** (below). Explain the webcam is used live, no images are stored,
   and they can stop at any time.
2. **Warm-up** (1 min): show the G / B keys ("press G if a reply helped,
   B if it did not").
3. For each condition, in the participant's order:
   1. A short, mildly frustrating task (for example a timed puzzle or a typing
      test) for 5 minutes while the assistant runs:
      ```bash
      python -m emotion_rl.assistant.app --condition A --user P1
      ```
   2. The questionnaire for that condition (below).
   3. One minute of rest.
4. **Debrief**: reveal what the study was about; ask for open feedback.

Use the same task difficulty, lighting, chair position and volume for every
session.

## Measures

**Objective** (from `data/interactions/*.jsonl`, computed by `emotion_rl.study`):

- valence change in the turn after the assistant spoke
- mean live reward per session
- share of turns spoken, thumbs up / down counts

**Subjective** (7-point Likert, 1 = strongly disagree, 7 = strongly agree), one row
per participant and condition in `data/study/questionnaire.csv`
(template: `docs/questionnaire_template.csv`):

| Column | Statement |
|---|---|
| `helpful` | The assistant's responses helped me feel better. |
| `appropriate` | The responses fitted how I was feeling. |
| `natural` | The assistant sounded natural. |
| `not_intrusive` | The assistant talked about the right amount (not too much). |
| `use_again` | I would like to use this assistant again. |

## Analysis

```bash
python -m emotion_rl.study --logs data/interactions --questionnaire data/study/questionnaire.csv
```

Friedman test across the three conditions, then pairwise Wilcoxon signed-rank
tests with Holm correction; report medians, effect direction and p-values. With
8–12 people, treat results as **pilot evidence**: report them honestly, including
null results.

`--effects` also prints the real (emotion × action) effect table. Comparing it
with the simulator's user types (`simulator.py`) tells you whether the simulator
is realistic, which is a good discussion section.

## Consent text (template)

> You are invited to try an experimental emotion-aware assistant. During the
> session your webcam is used to estimate your facial expression in real time.
> **No video or images are saved**; only estimated emotion probabilities, the
> assistant's replies, any key presses and (if you use push-to-talk) the
> transcript of what you say are stored, under a participant code instead of
> your name. The assistant is a research prototype, not a mental-health service.
> You can stop at any time without giving a reason, and ask for your data to be
> deleted. Do you agree to take part? ☐ Yes ☐ No

## Data handling

- Logs are keyed by participant code (P1…Pn); keep the code↔name list separately.
- `data/` is git-ignored: never commit participant data.
- Delete raw logs after analysis if you promised to.
