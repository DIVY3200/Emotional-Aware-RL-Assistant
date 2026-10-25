"""Analyse a user study run with the live assistant (protocol: docs/user_study.md).

    python -m emotion_rl.study --logs data/interactions --questionnaire data/study/questionnaire.csv

* Objective measures, per session, from the interaction logs: mean live reward,
  valence change after the assistant spoke, share of turns spoken, feedback.
* Subjective measures from the questionnaire (1–7 Likert).
* Conditions are unblinded with the mapping in configs/default.yaml.
* Within-subject design (every participant saw every condition): Friedman test,
  then pairwise Wilcoxon signed-rank with Holm correction. Otherwise
  Kruskal–Wallis, then Mann–Whitney U.
* --effects prints the empirical response-effect table (mean valence change per
  detected emotion × action), the real-data counterpart of the simulator's
  user types, for calibrating the simulator.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from emotion_rl.config import load_config
from emotion_rl.emotions import ACTIONS, EMOTIONS, LISTEN


def load_logs(log_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    if not records:
        raise SystemExit(f"No interaction logs found in {log_dir}")
    return pd.DataFrame(records)


def session_table(logs, condition_map):
    transitions = logs[logs["event"] == "transition"].copy()
    transitions["spoke"] = transitions["action"] != LISTEN
    transitions["delta_valence"] = transitions["valence_after"] - transitions["valence_before"]
    rows = []
    for session_id, s in transitions.groupby("session_id"):
        spoken = s[s["spoke"]]
        condition = s["condition"].iloc[0]
        rows.append({
            "session_id": session_id,
            "participant": s["user"].iloc[0],
            "condition": condition,
            "policy": condition_map.get(condition, s["policy"].iloc[0]),
            "turns": len(s),
            "mean_reward": s["reward"].mean(),
            "mean_valence": s["valence_after"].mean(),
            "delta_valence_after_speaking": spoken["delta_valence"].mean() if len(spoken) else 0.0,
            "speak_rate": s["spoke"].mean(),
            "thumbs_up": int((s["feedback"] > 0).sum()),
            "thumbs_down": int((s["feedback"] < 0).sum()),
        })
    return pd.DataFrame(rows)


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def compare_conditions(df, measure):
    """Picks within- or between-subject tests depending on the design."""
    pivot = df.pivot_table(index="participant", columns="policy", values=measure, aggfunc="mean")
    policies = list(pivot.columns)
    result = {"measure": measure,
              "means": df.groupby("policy")[measure].mean().to_dict(),
              "n_participants": int(pivot.shape[0])}
    if len(policies) < 2:
        return result
    complete = pivot.dropna()
    within = len(complete) >= 3 and len(complete) == len(pivot)
    pairs = [(a, b) for i, a in enumerate(policies) for b in policies[i + 1:]]
    if within:
        result["design"] = "within-subject"
        if len(policies) > 2:
            result["friedman_p"] = float(stats.friedmanchisquare(
                *[complete[p] for p in policies]).pvalue)
        raw = [float(stats.wilcoxon(complete[a], complete[b]).pvalue) for a, b in pairs]
    else:
        result["design"] = "between-subject"
        groups = [df.loc[df["policy"] == p, measure] for p in policies]
        if len(policies) > 2:
            result["kruskal_p"] = float(stats.kruskal(*groups).pvalue)
        raw = [float(stats.mannwhitneyu(df.loc[df["policy"] == a, measure],
                                        df.loc[df["policy"] == b, measure]).pvalue)
               for a, b in pairs]
    result["pairwise"] = [{"a": a, "b": b, "p": p, "p_holm": float(ph)}
                          for (a, b), p, ph in zip(pairs, raw, holm(np.array(raw)))]
    return result


def effect_table(logs):
    """Mean valence change by (dominant detected emotion before, action), with counts."""
    t = logs[logs["event"] == "transition"].copy()
    t["emotion"] = [EMOTIONS[int(np.argmax(p))] for p in t["prev_probs"]]
    t["action_name"] = [ACTIONS[int(a)] for a in t["action"]]
    t["delta"] = t["valence_after"] - t["valence_before"]
    return t.pivot_table(index="emotion", columns="action_name", values="delta",
                         aggfunc=["mean", "count"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyse user-study logs")
    parser.add_argument("--logs", default="data/interactions")
    parser.add_argument("--questionnaire", default="data/study/questionnaire.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--effects", action="store_true")
    args = parser.parse_args(argv)

    condition_map = load_config(args.config)["study"]["conditions"]
    logs = load_logs(args.logs)
    sessions = session_table(logs, condition_map)
    print(sessions.groupby("policy")[["mean_reward", "delta_valence_after_speaking",
                                      "speak_rate", "thumbs_up", "thumbs_down"]].mean())
    report = {"sessions": sessions.to_dict(orient="records"), "objective": [
        compare_conditions(sessions, m) for m in
        ("mean_reward", "delta_valence_after_speaking", "mean_valence")]}

    if os.path.exists(args.questionnaire):
        q = pd.read_csv(args.questionnaire)
        q["policy"] = q["condition"].map(condition_map).fillna(q["condition"])
        items = [c for c in q.columns if c not in ("participant", "condition", "policy", "order")]
        report["subjective"] = [compare_conditions(q, item) for item in items]
        print(q.groupby("policy")[items].mean())

    if args.effects:
        print("\nEmpirical response effects (mean valence change):")
        print(effect_table(logs))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "study_summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[Study] summary -> {os.path.join(args.out, 'study_summary.json')}")


if __name__ == "__main__":
    main()
