# v1 (semester version), kept for comparison

These are the original scripts and trained model from the first version of the
project. They are **not used** by v2 (`src/emotion_rl/`), but are kept so the
improvements can be checked and reproduced.

Run `python audit_v1.py` from this folder to reproduce the issues that led to
the redesign:

| # | Finding in v1 | Measured | Fixed in v2 by |
|---|---|---|---|
| 1 | The next emotion ignored the action (`_get_simulated_emotion` only used the current emotion) | identical next-emotion distribution for all 5 actions | `simulator.py`: P(s′ \| s, a, user type) |
| 2 | Reward used `curr + prev` instead of `curr − prev` | happy → sad rewarded **+1.00** | `env.py` reward + regression test |
| 3 | The only action-dependent reward was a +0.2 bonus for matching a hand-written table, so PPO just copied the table | PPO +7.40 vs table +7.99 vs random +4.80 | bonus removed; agents compared with that table as a baseline |
| 4 | Disgust could not start an episode and was rarely reachable | 0% of starts | smoothed transitions, start distribution |
| 5 | Global `np.random` instead of the env's seeded generator | `reset(seed=42)` not reproducible | `self.np_random` everywhere + test |
| 6 | DeepFace received RGB frames but expects BGR | — | `detector.py` + test |
| 7 | `requirements.txt` listed `google-generativeai` but the code imported `google-genai`; `pyttsx3` and `python-dotenv` missing | fresh install failed | `pyproject.toml` + pinned `requirements.txt` |
| 8 | Evaluation: 5 episodes, no baselines, accuracy measured against the same hand-written table | std ≈ 8 per episode | `evaluate.py`: 1000 episodes × 5 seeds, baselines, CIs, paired tests |
