"""Train response policies on the user simulator, over several random seeds.

Examples:
    python -m emotion_rl.train --algo ppo --seeds 0 1 2 3 4
    python -m emotion_rl.train --algo ppo --history-len 1 --tag ppo_no_memory
    python -m emotion_rl.train --algo recurrent_ppo --history-len 1 --timesteps 300000

Each run writes to runs/<tag>/seed_<n>/:
    best_model.zip   best checkpoint on the evaluation users
    final_model.zip  weights at the end of training
    evaluations.npz  learning curve (mean evaluation return vs timesteps)
    config.json      everything needed to reproduce the run
"""

import argparse
import copy
import json
import os
import platform
import sys
import time

import numpy as np

from emotion_rl.env import EmotionSupportEnv


def block_tensorflow():
    """SB3 imports TensorBoard, which loads TensorFlow (~1 GB RAM) when it is installed.
    Training never needs TF, so make the import fail; TensorBoard falls back to its stub."""
    sys.modules.setdefault("tensorflow", None)

ALGO_HYPERPARAMS = {
    "ppo": dict(n_steps=256, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.95,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128])),
    "a2c": dict(n_steps=16, learning_rate=7e-4, gamma=0.95, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128])),
    "dqn": dict(learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000, batch_size=128,
                gamma=0.95, train_freq=1, gradient_steps=2, target_update_interval=1_000,
                exploration_fraction=0.3, exploration_final_eps=0.02,
                policy_kwargs=dict(net_arch=[128, 128])),
    "recurrent_ppo": dict(n_steps=256, batch_size=256, n_epochs=10, learning_rate=3e-4,
                          gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                          policy_kwargs=dict(lstm_hidden_size=64, net_arch=[128])),
}


def make_model(algo, env, seed, tensorboard_log=None):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import A2C, DQN, PPO

    classes = {"ppo": (PPO, "MlpPolicy"), "a2c": (A2C, "MlpPolicy"),
               "dqn": (DQN, "MlpPolicy"), "recurrent_ppo": (RecurrentPPO, "MlpLstmPolicy")}
    cls, policy = classes[algo]
    # deepcopy: SB3's A2C writes its optimizer class into the policy_kwargs dict it is given
    return cls(policy, env, seed=seed, verbose=0, device="cpu",
               tensorboard_log=tensorboard_log, **copy.deepcopy(ALGO_HYPERPARAMS[algo]))


def run_finished(run_dir):
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            json.load(f)
        return os.path.exists(os.path.join(run_dir, "best_model.zip"))
    except (OSError, ValueError):
        return False


def train_one(algo, seed, env_kwargs, timesteps, run_dir, n_envs=8, eval_every=10_000,
              n_eval_episodes=200, tensorboard=False):
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(run_dir, exist_ok=True)
    vec_env = make_vec_env(lambda: EmotionSupportEnv(**env_kwargs), n_envs=n_envs, seed=seed)
    # Evaluation users use a different seed stream from training users
    eval_env = Monitor(EmotionSupportEnv(**env_kwargs))
    eval_env.reset(seed=10_000 + seed)

    model = make_model(algo, vec_env, seed, run_dir if tensorboard else None)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(eval_every // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    start = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=False)
    elapsed = time.time() - start
    model.save(os.path.join(run_dir, "final_model"))

    import stable_baselines3

    config = {
        "algo": algo,
        "seed": seed,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "env": env_kwargs,
        "hyperparams": ALGO_HYPERPARAMS[algo],
        "best_eval_return": float(eval_callback.best_mean_reward),
        "train_seconds": round(elapsed, 1),
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an emotion-support policy")
    parser.add_argument("--algo", default="ppo", choices=list(ALGO_HYPERPARAMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--obs-mode", default="probs", choices=["probs", "onehot"])
    parser.add_argument("--detector-accuracy", type=float, default=0.65)
    parser.add_argument("--max-steps", type=int, default=20, help="turns per session")
    parser.add_argument("--privileged", action="store_true",
                        help="reveal the hidden user type (upper-bound teacher agent only)")
    parser.add_argument("--confusion", default=None,
                        help="optional .npy confusion matrix measured with detector_eval.py")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--tag", default=None, help="run name (default: algo)")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads; 1 is fastest for these small networks")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip seeds that already finished (resume an interrupted sweep)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.tensorboard:
        block_tensorflow()
    import torch

    torch.set_num_threads(args.threads)
    tag = args.tag or args.algo
    env_kwargs = {
        "history_len": args.history_len,
        "observation_mode": args.obs_mode,
        "detector_accuracy": args.detector_accuracy,
        "max_steps": args.max_steps,
        "privileged": args.privileged,
    }
    if args.confusion:
        env_kwargs["confusion"] = np.load(args.confusion).tolist()

    print(f"[Train] {tag}: algo={args.algo} seeds={args.seeds} timesteps={args.timesteps:,}")
    for seed in args.seeds:
        run_dir = os.path.join(args.out, tag, f"seed_{seed}")
        if args.skip_existing and run_finished(run_dir):
            print(f"[Train] {tag} seed {seed}: already finished, skipping")
            continue
        config = train_one(args.algo, seed, env_kwargs, args.timesteps, run_dir,
                           n_envs=args.n_envs, eval_every=args.eval_every,
                           tensorboard=args.tensorboard)
        print(f"[Train] {tag} seed {seed}: best eval return {config['best_eval_return']:+.2f} "
              f"({config['train_seconds']:.0f}s) -> {run_dir}")


if __name__ == "__main__":
    main()
