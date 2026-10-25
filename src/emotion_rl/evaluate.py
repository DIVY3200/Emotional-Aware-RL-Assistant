"""Rigorous evaluation of every policy on held-out simulated users.

    python -m emotion_rl.evaluate --runs runs --out results

What it measures (all on evaluation seeds never used in training):
  1. Main comparison: baselines, bandits, trained agents (mean ± 95% CI over
     training seeds) on the four known user types, plus a per-user-type breakdown.
  2. Generalisation: users that blend two user types, never seen in training.
  3. Paired significance test of the main agent against each baseline (every
     policy meets the same users and start emotions: common random numbers).
  4. Robustness to emotion-detector accuracy (agents trained at 65%).
  5. Online personalisation across 10 sessions with the same user (known and blended types).
  6. Learning curves (mean ± std over seeds) from the training logs.

Outputs: results/summary.json, results/*.csv, results/results.md, results/*.png
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from emotion_rl import plots
from emotion_rl.bandits import (
    LinearThompsonPolicy,
    LinUCBPolicy,
    OnlinePersonalizer,
    PersonalizedPolicy,
    train_bandit,
)
from emotion_rl.env import EmotionSupportEnv
from emotion_rl.policies import (
    GreedyOraclePolicy,
    Policy,
    RuleCooldownPolicy,
    SB3Policy,
    baseline_policies,
)
from emotion_rl.simulator import PERSONA_NAMES

EVAL_SEED = 777_000
PERSONALISATION_SEED = 888_000

TAG_NAMES = {
    "ppo_no_memory": "PPO (ours)",
    "ppo": "PPO + 4-turn memory",
    "ppo_top1": "PPO, top-1 label input",
    "ppo_privileged": "PPO told the user type (upper bound)",
    "recurrent_ppo": "Recurrent PPO (LSTM)",
    "a2c": "A2C + memory",
    "dqn": "DQN + memory",
}


# ─── Episode runner ────────────────────────────────────────────────────────────
def run_episodes(policy, env, n_episodes, seed=EVAL_SEED, options=None, learn=False):
    rows = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i, options=options)
        policy.reset()
        total, valence, spoke, done = 0.0, 0.0, 0, False
        while not done:
            action = policy.act(obs, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if learn:
                policy.update(obs, action, reward)
            total += reward
            valence += info["valence"]
            spoke += info["spoke"]
            obs, done = next_obs, terminated or truncated
        rows.append({"episode": i, "return": total, "mean_valence": valence / env.max_steps,
                     "speak_rate": spoke / env.max_steps, "persona": info["persona"]})
    return pd.DataFrame(rows)


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


# ─── Policy registry ───────────────────────────────────────────────────────────
class PolicyEntry:
    """One row of the comparison: a single policy, or one policy per training seed."""

    def __init__(self, name, family, make, seeds=(None,), env_kwargs=None):
        self.name = name
        self.family = family
        self.make = make  # seed -> Policy
        self.seeds = seeds
        self.env_kwargs = env_kwargs or {}


def discover_runs(runs_dir):
    groups = {}
    for config_path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed_*", "config.json"))):
        try:
            with open(config_path) as f:
                config = json.load(f)
        except ValueError:
            print(f"[Eval] skipping unfinished run {os.path.dirname(config_path)}")
            continue
        seed_dir = os.path.dirname(config_path)
        tag = os.path.basename(os.path.dirname(seed_dir))
        if os.path.exists(os.path.join(seed_dir, "best_model.zip")):
            groups.setdefault(tag, []).append((seed_dir, config))
    return groups


def rl_entries(groups):
    entries = []
    for tag, runs in groups.items():
        configs = {seed_dir: cfg for seed_dir, cfg in runs}
        env_kwargs = {k: v for k, v in runs[0][1]["env"].items() if k != "confusion"}

        def make(seed_dir, configs=configs, tag=tag):
            return SB3Policy.from_path(os.path.join(seed_dir, "best_model"),
                                       algo=configs[seed_dir]["algo"], name=tag)

        family = "Oracle" if env_kwargs.get("privileged") else "Deep RL"
        entries.append(PolicyEntry(TAG_NAMES.get(tag, tag), family, make,
                                   seeds=[d for d, _ in runs], env_kwargs=env_kwargs))
    return entries


def bandit_entries(bandit_steps, seeds):
    def make_linucb(seed):
        env = EmotionSupportEnv()
        return train_bandit(LinUCBPolicy(seed=seed), env, bandit_steps, seed=seed)

    def make_thompson(seed):
        env = EmotionSupportEnv()
        return train_bandit(LinearThompsonPolicy(seed=seed), env, bandit_steps, seed=seed)

    return [PolicyEntry("LinUCB (contextual bandit)", "Contextual bandit", make_linucb, seeds),
            PolicyEntry("Thompson sampling (bandit)", "Contextual bandit", make_thompson, seeds)]


def baseline_entries():
    entries = []
    for policy in baseline_policies():
        entries.append(PolicyEntry(policy.name, "Heuristic", lambda _s, p=policy: p))
    entries.append(PolicyEntry("Greedy oracle (sees hidden state)", "Oracle",
                               lambda _s: GreedyOraclePolicy()))
    return entries


def evaluate_entry(entry, n_episodes, env_overrides=None, cache=None):
    """Returns (per-seed list of episode DataFrames)."""
    frames = []
    for seed in entry.seeds:
        key = (entry.name, seed)
        policy = cache[key] if cache is not None and key in cache else entry.make(seed)
        if cache is not None:
            cache[key] = policy
        env = EmotionSupportEnv(**{**entry.env_kwargs, **(env_overrides or {})})
        frames.append(run_episodes(policy, env, n_episodes))
    return frames


def summarise(entry, frames):
    seed_means = [f["return"].mean() for f in frames]
    if len(frames) > 1:
        mean, spread = float(np.mean(seed_means)), ci95(seed_means)
        ci_over = "training seeds"
    else:
        mean, spread = float(frames[0]["return"].mean()), ci95(frames[0]["return"])
        ci_over = "episodes"
    allf = pd.concat(frames)
    return {
        "name": entry.name,
        "family": entry.family,
        "mean": mean,
        "ci95": spread,
        "ci_over": ci_over,
        "seed_std": float(np.std(seed_means, ddof=1)) if len(frames) > 1 else None,
        "n_seeds": len(frames),
        "mean_valence": float(allf["mean_valence"].mean()),
        "speak_rate": float(allf["speak_rate"].mean()),
    }


def paired_test(frames_a, frames_b):
    """Per-episode difference (averaged over seeds), Wilcoxon signed-rank test."""
    a = np.mean([f["return"].to_numpy() for f in frames_a], axis=0)
    b = np.mean([f["return"].to_numpy() for f in frames_b], axis=0)
    diff = a - b
    p_value = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    return {"mean_diff": float(diff.mean()), "ci95": ci95(diff), "p_value": p_value}


# ─── Experiments ───────────────────────────────────────────────────────────────
def comparison(entries, n_episodes, env_overrides=None, cache=None, label=""):
    results, frames_by_name = [], {}
    for entry in entries:
        start = time.time()
        frames = evaluate_entry(entry, n_episodes, env_overrides, cache)
        frames_by_name[entry.name] = frames
        row = summarise(entry, frames)
        results.append(row)
        print(f"[Eval{label}] {entry.name:38s} {row['mean']:+6.2f} +/- {row['ci95']:.2f} "
              f"(spoke {row['speak_rate']:.0%}, {time.time() - start:.0f}s)")
    return results, frames_by_name


def per_persona(frames_by_name, names):
    table = {}
    for name in names:
        allf = pd.concat(frames_by_name[name])
        table[name] = allf.groupby("persona")["return"].mean().reindex(PERSONA_NAMES).tolist()
    return table


def detector_robustness(entries, accuracies, n_episodes, cache):
    curves = {}
    for entry in entries:
        means, cis = [], []
        for acc in accuracies:
            frames = evaluate_entry(entry, n_episodes, {"detector_accuracy": acc}, cache)
            seed_means = [f["return"].mean() for f in frames]
            if len(frames) > 1:
                means.append(float(np.mean(seed_means)))
                cis.append(ci95(seed_means))
            else:
                means.append(float(frames[0]["return"].mean()))
                cis.append(ci95(frames[0]["return"]))
        curves[entry.name] = (means, cis)
        print(f"[Robustness] {entry.name:38s} " + " ".join(f"{m:+.2f}" for m in means))
    return curves


def personalisation(entry, n_users, n_sessions, mix_personas=True):
    """Same unseen user for n_sessions; PPO alone vs PPO + online personaliser."""
    by_kind = {"ppo": np.zeros((0, n_sessions)), "personalised": np.zeros((0, n_sessions))}
    for seed_dir in entry.seeds:
        base = entry.make(seed_dir)
        env = EmotionSupportEnv(**{**entry.env_kwargs, "mix_personas": mix_personas})
        for kind in by_kind:
            curves = []
            for user in range(n_users):
                policy: Policy = base if kind == "ppo" else PersonalizedPolicy(
                    base, OnlinePersonalizer(seed=user))
                env.reset(seed=PERSONALISATION_SEED + user)  # draws the user
                returns = []
                for session in range(n_sessions):
                    options = {"keep_user": True}
                    frame = run_episodes(policy, env, 1,
                                         seed=PERSONALISATION_SEED + 1000 * user + session,
                                         options=options, learn=kind != "ppo")
                    returns.append(frame["return"].iloc[0])
                curves.append(returns)
            by_kind[kind] = np.vstack([by_kind[kind], np.array(curves)])
    summary = {}
    for kind, matrix in by_kind.items():
        summary[kind] = {"mean": matrix.mean(axis=0).tolist(),
                         "ci95": [ci95(matrix[:, s]) for s in range(n_sessions)]}
    late = slice(n_sessions // 2, n_sessions)
    diff = by_kind["personalised"][:, late].mean(axis=1) - by_kind["ppo"][:, late].mean(axis=1)
    summary["late_sessions_gain"] = {"mean": float(diff.mean()), "ci95": ci95(diff),
                                     "p_value": float(stats.wilcoxon(diff).pvalue)}
    return summary


def export_best(groups, tag, out_path):
    """Copy the seed with the best *validation* return (EvalCallback during training, separate
    user seeds) to out_path for the live assistant. Test results are never used for selection."""
    import shutil

    seed_dir, config = max(groups[tag], key=lambda run: run[1]["best_eval_return"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.copyfile(os.path.join(seed_dir, "best_model.zip"), out_path)
    meta = {"algo": config["algo"], "env": config["env"], "source": seed_dir.replace("\\", "/"),
            "validation_return": config["best_eval_return"]}
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Export] {seed_dir} -> {out_path} "
          f"(validation return {config['best_eval_return']:+.2f})")


def learning_curve_data(groups, tags):
    curves = {}
    for tag in tags:
        if tag not in groups:
            continue
        runs = []
        for seed_dir, _ in groups[tag]:
            data = np.load(os.path.join(seed_dir, "evaluations.npz"))
            runs.append((data["timesteps"], data["results"].mean(axis=1)))
        n = min(len(r[1]) for r in runs)
        steps = runs[0][0][:n]
        matrix = np.vstack([r[1][:n] for r in runs])
        curves[TAG_NAMES.get(tag, tag)] = (steps, matrix.mean(axis=0), matrix.std(axis=0))
    return curves


# ─── Report ────────────────────────────────────────────────────────────────────
def markdown_table(rows, baseline_name=None):
    lines = ["| Policy | Mean return (95% CI) | Mean valence | Speaks | CI over |",
             "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["mean"]):
        lines.append(f"| {r['name']} | {r['mean']:+.2f} ± {r['ci95']:.2f} | "
                     f"{r['mean_valence']:+.3f} | {r['speak_rate']:.0%} | {r['ci_over']} |")
    return "\n".join(lines)


def write_report(summary, groups, out):
    """Figures and results.md from summary.json (also used by --plots-only)."""
    known, unseen = summary["known_users"], summary["unseen_users"]
    rule_cd = next(r for r in known if r["name"] == "Rule table + 2-turn cooldown")["mean"]
    oracle = next(r for r in known if r["name"].startswith("Greedy oracle"))["mean"]
    refs = {"Rule table + cooldown": rule_cd, "Greedy oracle": oracle}
    algo_curves = learning_curve_data(groups, ["ppo_no_memory", "recurrent_ppo", "a2c", "dqn"])
    ablation_curves = learning_curve_data(groups, ["ppo_no_memory", "ppo", "ppo_top1",
                                                   "ppo_privileged"])

    plots.policy_comparison(known, os.path.join(out, "policy_comparison.png"),
                            "Policy comparison on held-out sessions (known user types)")
    plots.policy_comparison(unseen, os.path.join(out, "policy_comparison_unseen.png"),
                            "Generalisation to unseen blended users")
    names = list(summary["per_persona"])
    plots.heatmap([summary["per_persona"][n] for n in names], names, list(PERSONA_NAMES),
                  os.path.join(out, "per_persona.png"), "Mean return by hidden user type")
    if algo_curves:
        plots.learning_curves(algo_curves, refs, os.path.join(out, "learning_curves.png"),
                              "Learning curves, mean ± std over seeds")
    if len(ablation_curves) > 1:
        plots.learning_curves(ablation_curves, refs, os.path.join(out, "ablation_state.png"),
                              "State ablation: memory and probability inputs")
    robustness = summary["detector_robustness"]
    acc_labels = [a if isinstance(a, str) else f"{a:.0%}" for a in robustness["accuracies"]]
    acc_labels = [a + "\n(train)" if a == "65%" else a for a in acc_labels]
    x = np.arange(len(acc_labels))
    plots.line_with_bands(x, robustness["curves"], os.path.join(out, "detector_robustness.png"),
                          "Robustness to emotion-detector accuracy",
                          "Simulated detector top-1 accuracy", "Mean return",
                          xticks=x, xticklabels=acc_labels)
    titles = {"known": "Online personalisation: same user, known user types",
              "unseen": "Online personalisation: same user, unseen blended users"}
    for label, pers in summary["personalisation"].items():
        sessions = np.arange(1, len(pers["ppo"]["mean"]) + 1)
        plots.line_with_bands(sessions, {
            "PPO + online personalisation": (pers["personalised"]["mean"],
                                             pers["personalised"]["ci95"]),
            "PPO alone": (pers["ppo"]["mean"], pers["ppo"]["ci95"]),
        }, os.path.join(out, f"personalisation_{label}.png"), titles[label],
            "Session with the same user", "Mean return", xticks=sessions)

    main_name = summary.get("main_policy", "main agent")
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write("## Known user types\n\n" + markdown_table(known) + "\n\n")
        f.write("## Unseen blended users\n\n" + markdown_table(unseen) + "\n\n")
        f.write(f"## Paired test vs {main_name}\n\n| Policy | Δ return (ours − policy) | "
                "95% CI | Wilcoxon p |\n|---|---|---|---|\n")
        for name, t in sorted(summary["paired_vs_main"].items(), key=lambda kv: kv[1]["mean_diff"]):
            f.write(f"| {name} | {t['mean_diff']:+.2f} | ± {t['ci95']:.2f} | "
                    f"{t['p_value']:.1e} |\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate all emotion-support policies")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--bandit-steps", type=int, default=300_000)
    parser.add_argument("--bandit-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--main-tag", default="ppo_no_memory",
                        help="deployed agent (chosen by validation return, see README)")
    parser.add_argument("--users", type=int, default=40, help="users per seed (personalisation)")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="small numbers for smoke tests")
    parser.add_argument("--plots-only", action="store_true",
                        help="redraw figures and results.md from an existing summary.json")
    parser.add_argument("--export", default="models/ppo_best.zip",
                        help="where to copy the main agent for the live assistant ('' to skip)")
    return parser.parse_args(argv)


def main(argv=None):
    from emotion_rl.train import block_tensorflow

    block_tensorflow()
    import torch

    torch.set_num_threads(1)
    args = parse_args(argv)
    if args.quick:
        args.episodes, args.bandit_steps, args.users, args.sessions = 50, 2_000, 3, 4
        args.bandit_seeds = args.bandit_seeds[:2]
    os.makedirs(args.out, exist_ok=True)

    groups = discover_runs(args.runs)
    if args.plots_only:
        with open(os.path.join(args.out, "summary.json")) as f:
            write_report(json.load(f), groups, args.out)
        print(f"[Done] figures redrawn in {args.out}/")
        return
    if args.main_tag not in groups:
        raise SystemExit(f"No trained runs for '{args.main_tag}' in {args.runs}/. "
                         f"Train first: python -m emotion_rl.train --algo ppo")
    rl = rl_entries(groups)
    entries = baseline_entries() + bandit_entries(args.bandit_steps, args.bandit_seeds) + rl
    main_entry = next(e for e in rl if e.name == TAG_NAMES.get(args.main_tag, args.main_tag))
    cache = {}
    summary = {"episodes": args.episodes, "eval_seed": EVAL_SEED}

    # 1-3. known user types, unseen blended users, paired tests
    known, known_frames = comparison(entries, args.episodes, cache=cache, label=":known")
    unseen, unseen_frames = comparison(entries, args.episodes, {"mix_personas": True},
                                       cache=cache, label=":unseen")
    summary["known_users"] = known
    summary["unseen_users"] = unseen
    summary["per_persona"] = per_persona(known_frames, [e.name for e in entries])
    summary["paired_vs_main"] = {
        name: paired_test(known_frames[main_entry.name], frames)
        for name, frames in known_frames.items() if name != main_entry.name
    }

    # 4. detector robustness
    accuracies = [0.45, 0.55, 0.65, 0.75, 0.85, None]
    robust_entries = [main_entry] + [e for e in rl if e.name == TAG_NAMES["ppo_top1"]] + [
        PolicyEntry("Rule table + 2-turn cooldown", "Heuristic", lambda _s: RuleCooldownPolicy())]
    robustness = detector_robustness(robust_entries, accuracies, max(args.episodes // 4, 20),
                                     cache)
    summary["detector_robustness"] = {"accuracies": ["perfect" if a is None else a
                                                     for a in accuracies],
                                      "curves": robustness}

    # 5. personalisation, for known user types and for unseen blended users
    summary["personalisation"] = {}
    for label, mix in (("known", False), ("unseen", True)):
        result = personalisation(main_entry, args.users, args.sessions, mix_personas=mix)
        summary["personalisation"][label] = result
        gain = result["late_sessions_gain"]
        print(f"[Personalisation:{label}] gain in sessions {args.sessions // 2 + 1}-"
              f"{args.sessions}: {gain['mean']:+.2f} +/- {gain['ci95']:.2f} "
              f"(p={gain['p_value']:.3g})")

    summary["main_policy"] = main_entry.name
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(known).to_csv(os.path.join(args.out, "known_users.csv"), index=False)
    pd.DataFrame(unseen).to_csv(os.path.join(args.out, "unseen_users.csv"), index=False)
    write_report(summary, groups, args.out)
    if args.export and not args.quick:
        export_best(groups, args.main_tag, args.export)
    print(f"[Done] results written to {args.out}/")


if __name__ == "__main__":
    main()
