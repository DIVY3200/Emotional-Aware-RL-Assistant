"""Policies that can drive the assistant: baselines, an oracle and trained SB3 agents.

Every policy has the same small interface so `evaluate.py` and the live
assistant can treat them identically:

    policy.reset()                          # new session
    action = policy.act(obs, env=None)      # choose an action index
    policy.update(obs, action, reward)      # optional online learning
"""

import os

import numpy as np

from emotion_rl.emotions import (
    ACTION_TO_INDEX,
    LISTEN,
    NUM_ACTIONS,
    NUM_EMOTIONS,
    V1_RULE_TABLE,
    VALENCE,
)
from emotion_rl.env import FRAME_SIZE


def current_emotion_probs(obs):
    """Detector probabilities of the most recent turn inside a stacked observation."""
    start = len(obs) - 1 - FRAME_SIZE
    return np.asarray(obs[start:start + NUM_EMOTIONS])


class Policy:
    name = "policy"
    learns_online = False

    def reset(self):
        pass

    def act(self, obs, env=None):
        raise NotImplementedError

    def update(self, obs, action, reward):
        pass


# ─── Baselines ─────────────────────────────────────────────────────────────────
class RandomPolicy(Policy):
    name = "Random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env=None):
        return int(self.rng.integers(NUM_ACTIONS))


class FixedActionPolicy(Policy):
    def __init__(self, action):
        self.action = ACTION_TO_INDEX[action] if isinstance(action, str) else int(action)
        self.name = f"Always {list(ACTION_TO_INDEX)[self.action]}"

    def act(self, obs, env=None):
        return self.action


class RuleTablePolicy(Policy):
    """v1 behaviour: hand-written emotion → action table, speaks every turn."""

    name = "v1 rule table"

    def act(self, obs, env=None):
        return V1_RULE_TABLE[int(np.argmax(current_emotion_probs(obs)))]


class RuleCooldownPolicy(Policy):
    """Rule table plus a fixed cooldown: speak once, then listen for `cooldown` turns."""

    def __init__(self, cooldown=2):
        self.cooldown = cooldown
        self.name = f"Rule table + {cooldown}-turn cooldown"
        self.wait = 0

    def reset(self):
        self.wait = 0

    def act(self, obs, env=None):
        if self.wait > 0:
            self.wait -= 1
            return LISTEN
        self.wait = self.cooldown
        return V1_RULE_TABLE[int(np.argmax(current_emotion_probs(obs)))]


class GreedyOraclePolicy(Policy):
    """Cheating reference: sees the true emotion, user type and fatigue, and picks
    the action with the best expected immediate reward. Not achievable in
    practice; it shows how much headroom is left."""

    name = "Greedy oracle (sees hidden state)"

    def act(self, obs, env=None):
        base = env.unwrapped
        sim, cfg = base.sim, base.reward_config
        v_prev = VALENCE[sim.emotion]
        best_action, best_value = LISTEN, -np.inf
        for action in range(NUM_ACTIONS):
            probs = sim.transition_probs(action)
            expected_v = float(probs @ VALENCE)
            value = cfg.w_level * expected_v + cfg.w_delta * (expected_v - v_prev)
            if action != LISTEN:
                value -= cfg.speak_cost
                if action == sim.last_action:
                    value -= cfg.repeat_cost
            if value > best_value:
                best_action, best_value = action, value
        return best_action


# ─── Trained agents ────────────────────────────────────────────────────────────
ALGORITHMS = ("ppo", "a2c", "dqn", "recurrent_ppo")


def load_sb3_model(path, algo=None, device="cpu"):
    """Load a saved SB3 model; the algorithm is read from the run's config if not given."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import A2C, DQN, PPO

    classes = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "recurrent_ppo": RecurrentPPO}
    if algo is None:
        algo = _algo_from_run_config(path) or "ppo"
    return classes[algo].load(path, device=device), algo


def read_model_config(model_path):
    """Training settings saved next to a model: models/<name>.json (exported by evaluate.py)
    or the run's config.json. Observations at inference must use the same history_len."""
    import json

    stem = os.path.splitext(model_path)[0]
    for candidate in (stem + ".json", os.path.join(os.path.dirname(model_path), "config.json")):
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return {"algo": "ppo", "env": {}}


def _algo_from_run_config(model_path):
    return read_model_config(model_path).get("algo")


class SB3Policy(Policy):
    """Wraps a trained Stable-Baselines3 model (handles recurrent LSTM state)."""

    def __init__(self, model, algo="ppo", name=None, deterministic=True):
        self.model = model
        self.algo = algo
        self.name = name or algo.upper()
        self.deterministic = deterministic
        self.lstm_state = None
        self.episode_start = np.ones((1,), dtype=bool)

    @classmethod
    def from_path(cls, path, algo=None, name=None):
        model, algo = load_sb3_model(path, algo)
        return cls(model, algo, name)

    def reset(self):
        self.lstm_state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def act(self, obs, env=None):
        if self.algo == "recurrent_ppo":
            action, self.lstm_state = self.model.predict(
                obs, state=self.lstm_state, episode_start=self.episode_start,
                deterministic=self.deterministic)
            self.episode_start = np.zeros((1,), dtype=bool)
            return int(action)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)

    def action_probs(self, obs):
        """π(a | obs) for actor-critic models (used by the online personalizer)."""
        import torch

        if self.algo not in ("ppo", "a2c"):
            raise ValueError("action_probs is only available for PPO/A2C policies")
        obs_tensor, _ = self.model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
        return dist.distribution.probs.cpu().numpy()[0]


def baseline_policies(seed=0):
    return [
        RandomPolicy(seed),
        FixedActionPolicy("comfort"),
        FixedActionPolicy("listen"),
        RuleTablePolicy(),
        RuleCooldownPolicy(cooldown=2),
    ]
