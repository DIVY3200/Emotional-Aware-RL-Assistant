"""Reproduces the problems found in v1 that motivated the v2 redesign.

    cd legacy/v1 && python audit_v1.py

Checks: (1) the action has no effect on the next emotion, (2) the reward sign
bug (happy -> sad is rewarded), (3) PPO vs the hand-written table it was meant
to replace, (4) disgust never starts an episode, (5) seeding does not work.
"""

import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

with contextlib.redirect_stdout(io.StringIO()):
    import rl_environment as R
    from emotion_detector import INDEX_TO_EMOTION
    from stable_baselines3 import PPO

model = PPO.load("model/ppo_agent/best_model")
with contextlib.redirect_stdout(io.StringIO()):
    env = R.EmotionRLEnvironment(use_webcam=False, max_steps=20)

print("(1) Next-emotion distribution from SAD for each action:")
for a in range(5):
    np.random.seed(0)
    nxt = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(20000):
            env.current_emotion_idx = 4
            env.step(a)
            nxt.append(env.current_emotion_idx)
    print(f"    {R.ACTIONS[a]:14s} {np.round(np.bincount(nxt, minlength=7) / len(nxt), 3)}")

print("(2) Reward examples:")
for p, c in [(3, 4), (4, 3), (4, 4)]:
    print(f"    {INDEX_TO_EMOTION[p]:6s} -> {INDEX_TO_EMOTION[c]:6s}: "
          f"{env.calculate_reward(p, c, 2):+.2f}")


def run(policy, n=2000):
    np.random.seed(0)
    returns = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                obs, r, term, trunc, _ = env.step(policy(obs))
                total += r
                done = term or trunc
            returns.append(total)
    return np.mean(returns)


rng = np.random.default_rng(0)
print("(3) Mean return over 2000 episodes:")
print(f"    PPO            {run(lambda o: int(model.predict(o, deterministic=True)[0])):+.2f}")
print(f"    lookup table   {run(lambda o: R.BEST_ACTION_FOR_EMOTION[int(np.argmax(o))]):+.2f}")
print(f"    random         {run(lambda o: int(rng.integers(5))):+.2f}")

starts = []
with contextlib.redirect_stdout(io.StringIO()):
    for _ in range(5000):
        env.reset()
        starts.append(env.current_emotion_idx)
print(f"(4) Share of episodes starting in disgust: {np.mean(np.array(starts) == 1):.3f}")

with contextlib.redirect_stdout(io.StringIO()):
    a = env.reset(seed=42)[0].argmax()
    np.random.seed(None)
    b = env.reset(seed=42)[0].argmax()
print(f"(5) reset(seed=42) twice gives the same start: {a == b}")
