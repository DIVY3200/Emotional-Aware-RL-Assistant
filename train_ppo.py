import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from rl_environment import EmotionRLEnvironment

MODEL_DIR = "model/ppo_agent"
LOG_DIR = "models/ppo_logs"
PLOT_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 100_000
EPISODE_STEPS = 20
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10

class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []

    def _on_step(self):
        reward = self.locals.get('rewards', [0])[0]
        self.current_rewards.append(reward)

        done = self.locals.get('dones', [False])[0]

        if done:
            total = sum(self.current_rewards)
            self.episode_rewards.append(total)
            self.episode_lengths.append(len(self.current_rewards))
            self.current_rewards = []

            if len(self.episode_rewards) % 50 == 0:
                avg = np.mean(self.episode_rewards[-50:])
                print(f"[Callback] Episodes: {len(self.episode_rewards):4d} | "
                      f"Avg Reward (last 50): {avg:+.2f}")
        return True

def make_env():
    env =  EmotionRLEnvironment(use_webcam=False, max_steps=EPISODE_STEPS)
    env = Monitor(env, LOG_DIR)
    return env

def train():
    print("=" * 60)
    print("  Training PPO Agent — Emotion-Aware RL Assistant")
    print("=" * 60)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Steps/episode   : {EPISODE_STEPS}")
    print(f"  Model save path : {MODEL_DIR}")
    print("=" * 60)

    env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print("\n[PPO Model Architecture]")
    print(model.policy)
    print()

    reward_callback = RewardTrackingCallback()
    eval_env = Monitor(EmotionRLEnvironment(use_webcam=False, max_steps=EPISODE_STEPS))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print("[Training Started...]\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[reward_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(MODEL_DIR, "ppo_emotion_final")
    model.save(final_path)
    print(f"\n[Model Saved] → {final_path}.zip")

    plot_rewards(reward_callback.episode_rewards)

    return model, reward_callback.episode_rewards


def plot_rewards(rewards):
    if not rewards:
        print("[Plot] No rewards to plot.")
        return

    plt.figure(figsize=(12, 5))

    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.4, color='steelblue', label='Episode Reward')
    plt.title('Episode Rewards During Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    # Rolling average
    plt.subplot(1, 2, 2)
    window = min(50, len(rewards))
    rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.plot(rolling_avg, color='orange', label=f'Rolling Avg ({window} ep)')
    plt.title('Smoothed Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "training_rewards.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"[Plot Saved] → {plot_path}")

def evaluate(model_path = None):
    print("\n" + "=" * 60)
    print("  Evaluating Trained PPO Agent")
    print("=" * 60)

    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "best_model")
    if not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] Model not found at {model_path}.zip")
        print("Please train first by running: python train_ppo.py")
        return

    model = PPO.load(model_path)
    print(f"[Model Loaded] <- {model_path}.zip\n")

    env = EmotionRLEnvironment(use_webcam=False, max_steps=EPISODE_STEPS)
    all_rewards = []

    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        print(f"\n── Evaluation Episode {ep + 1} ──")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        all_rewards.append(ep_reward)
        print(f"Episode {ep + 1} Total Reward: {ep_reward:.2f}")

    print(f"\n[Evaluation Done]")
    print(f"Average Reward over 5 episodes: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or Evaluate PPO Emotion Agent")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help="'train' to train, 'eval' to evaluate saved model")
    args = parser.parse_args()

    if args.mode == 'train':
        model, rewards = train()
        print("\n[Training Complete!]")
        print("Next step: run_assistant.py for live webcam demo")

    elif args.mode == 'eval':
        evaluate()
