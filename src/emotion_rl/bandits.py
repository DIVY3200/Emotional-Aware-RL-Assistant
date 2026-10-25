"""Online learners: contextual bandits and per-user personalisation of a trained policy.

Contextual bandits (LinUCB, linear Thompson sampling) learn which response
gives the best *immediate* reward for the current context. They are the right
tool when actions do not affect the future; comparing them with PPO shows
whether long-horizon effects (fatigue, repetition) actually matter here.

OnlinePersonalizer keeps a small Bayesian model per user and nudges a trained
policy towards responses that worked for *this* user:

    π_user(a | s)  ∝  π_ppo(a | s) · exp(η · Ã_user(s, a))

which is the closed-form solution of maximising E[Ã] − (1/η)·KL(π ‖ π_ppo):
stay close to the population policy until there is evidence this user differs.
Ã is a Thompson sample from a hierarchical Bayesian linear model,

    A_user(e, a) = β_a + γ_{e,a},   β ~ N(0, 0.4²),  γ ~ N(0, 0.15²)

β_a is how much this user likes response a in general (shared across emotions)
and γ is a small emotion-specific correction. Sharing matters: a first version
with an independent value per (emotion, action) cell had 35 noisy parameters
and gave no measurable gain; the shared model learns from every turn.
The trained policy still decides *when* to speak; the personaliser only
re-ranks *which* response to give.
"""

import json

import numpy as np

from emotion_rl.emotions import ACTIONS, LISTEN, NUM_ACTIONS, NUM_EMOTIONS, SPEAKING_ACTIONS
from emotion_rl.env import FRAME_SIZE
from emotion_rl.policies import Policy, current_emotion_probs


def bandit_features(obs):
    """Most recent turn of the observation plus a bias term."""
    frame = np.asarray(obs[len(obs) - 1 - FRAME_SIZE:len(obs) - 1], dtype=np.float64)
    return np.append(frame, 1.0)


# ─── Contextual bandits ────────────────────────────────────────────────────────
class _LinearBandit(Policy):
    learns_online = True

    def __init__(self, ridge=1.0, n_features=FRAME_SIZE + 1, seed=0):
        self.A_inv = np.stack([np.eye(n_features) / ridge] * NUM_ACTIONS)
        self.b = np.zeros((NUM_ACTIONS, n_features))
        self.rng = np.random.default_rng(seed)
        self.frozen = False

    def theta(self):
        return np.einsum("aij,aj->ai", self.A_inv, self.b)

    def update(self, obs, action, reward):
        if self.frozen:
            return
        x = bandit_features(obs)
        A_inv = self.A_inv[action]
        Ax = A_inv @ x
        self.A_inv[action] = A_inv - np.outer(Ax, Ax) / (1.0 + x @ Ax)  # Sherman-Morrison
        self.b[action] += reward * x

    def freeze(self):
        """Stop exploring and learning (used for evaluation after training)."""
        self.frozen = True


class LinUCBPolicy(_LinearBandit):
    """Li et al. (2010): pick argmax_a  x·θ_a + α·sqrt(xᵀ A_a⁻¹ x)."""

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.name = "LinUCB (contextual bandit)"

    def act(self, obs, env=None):
        x = bandit_features(obs)
        mean = self.theta() @ x
        if self.frozen:
            return int(np.argmax(mean))
        width = np.sqrt(np.einsum("i,aij,j->a", x, self.A_inv, x))
        return int(np.argmax(mean + self.alpha * width))


class LinearThompsonPolicy(_LinearBandit):
    """Agrawal & Goyal (2013): sample θ_a ~ N(A_a⁻¹ b_a, v² A_a⁻¹) and act greedily."""

    def __init__(self, v=0.5, **kwargs):
        super().__init__(**kwargs)
        self.v = v
        self.name = "Thompson sampling (contextual bandit)"

    def act(self, obs, env=None):
        x = bandit_features(obs)
        theta = self.theta()
        if self.frozen:
            return int(np.argmax(theta @ x))
        scores = np.empty(NUM_ACTIONS)
        for a in range(NUM_ACTIONS):
            # x·θ̃ with θ̃ ~ N(θ, v²A⁻¹) is Gaussian, so sample the score directly
            var = self.v ** 2 * float(x @ self.A_inv[a] @ x)
            scores[a] = theta[a] @ x + np.sqrt(max(var, 0.0)) * self.rng.standard_normal()
        return int(np.argmax(scores))


def train_bandit(policy, env, timesteps, seed=0):
    """Run a bandit online in the simulator, then freeze it for evaluation."""
    obs, _ = env.reset(seed=seed)
    for _ in range(timesteps):
        action = policy.act(obs, env)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        policy.update(obs, action, reward)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    policy.freeze()
    return policy


# ─── Per-user personalisation on top of a trained policy ──────────────────────
N_SPEAKING = len(SPEAKING_ACTIONS)


class OnlinePersonalizer:
    def __init__(self, eta=8.0, shared_std=0.4, residual_std=0.15, noise_std=0.7,
                 baseline_rate=0.1, seed=0):
        """eta=8 was chosen on validation users (seeds disjoint from the evaluation users)."""
        self.eta = eta
        self.noise_var = noise_std ** 2
        self.baseline_rate = baseline_rate
        prior = [1 / shared_std ** 2] * N_SPEAKING + [1 / residual_std ** 2] * (
            NUM_EMOTIONS * N_SPEAKING)
        self.precision = np.diag(prior)  # posterior precision of [β, γ]
        self.b = np.zeros(len(prior))  # precision-weighted sum of targets
        self.value = np.zeros(NUM_EMOTIONS)  # running mean reward per emotion (baseline)
        self.n_updates = 0
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def features(emotion_probs, k):
        """[one-hot(k) | emotion_probs ⊗ one-hot(k)] for speaking action number k."""
        x = np.zeros(N_SPEAKING + NUM_EMOTIONS * N_SPEAKING)
        x[k] = 1.0
        x[N_SPEAKING + np.arange(NUM_EMOTIONS) * N_SPEAKING + k] = emotion_probs
        return x

    @property
    def mean(self):
        return np.linalg.solve(self.precision, self.b)

    def advantage(self, emotion_probs, sample=True):
        """Ã for each speaking action under the current (soft) emotion estimate."""
        theta = self.mean
        if sample:
            cov = np.linalg.inv(self.precision)
            theta = self.rng.multivariate_normal(theta, (cov + cov.T) / 2)
        p = np.asarray(emotion_probs, dtype=np.float64)
        return np.array([self.features(p, k) @ theta for k in range(N_SPEAKING)])

    def choose(self, prior_probs, emotion_probs, sample=True):
        prior_probs = np.asarray(prior_probs)
        if int(np.argmax(prior_probs)) == LISTEN:
            return LISTEN
        speaking = np.array(SPEAKING_ACTIONS)
        logits = np.log(prior_probs[speaking] + 1e-8) + self.eta * self.advantage(
            emotion_probs, sample)
        return int(speaking[np.argmax(logits)])

    def update(self, emotion_probs, action, reward):
        p = np.asarray(emotion_probs, dtype=np.float64)
        x = self.features(p, SPEAKING_ACTIONS.index(action))
        target = reward - float(p @ self.value)  # advantage over this user's usual reward
        self.precision += np.outer(x, x) / self.noise_var
        self.b += x * target / self.noise_var
        self.value += self.baseline_rate * p * (reward - self.value)
        self.n_updates += 1

    def preferences(self):
        """This user's learned general liking of each response (β), for display/debugging."""
        return dict(zip([ACTIONS[a] for a in SPEAKING_ACTIONS], self.mean[:N_SPEAKING].round(3)))

    # ── persistence: one JSON profile per user ──────────────────────────────
    def to_dict(self):
        return {"eta": self.eta, "noise_var": self.noise_var, "baseline_rate": self.baseline_rate,
                "precision": self.precision.tolist(), "b": self.b.tolist(),
                "value": self.value.tolist(), "n_updates": self.n_updates,
                "preferences": {k: float(v) for k, v in self.preferences().items()}}

    @classmethod
    def from_dict(cls, data, seed=0):
        obj = cls(eta=data["eta"], baseline_rate=data["baseline_rate"], seed=seed)
        obj.noise_var = data["noise_var"]
        obj.precision = np.array(data["precision"])
        obj.b = np.array(data["b"])
        obj.value = np.array(data["value"])
        obj.n_updates = data["n_updates"]
        return obj

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


class PersonalizedPolicy(Policy):
    """A trained PPO/A2C policy plus an OnlinePersonalizer for one user.

    reset() starts a new session but keeps what was learned about the user.
    """

    learns_online = True

    def __init__(self, base_policy, personalizer=None, name=None):
        self.base = base_policy
        self.personalizer = personalizer or OnlinePersonalizer()
        self.name = name or f"{base_policy.name} + online personalisation"

    def reset(self):
        self.base.reset()

    def act(self, obs, env=None):
        prior = self.base.action_probs(obs)
        return self.personalizer.choose(prior, current_emotion_probs(obs))

    def update(self, obs, action, reward):
        if action != LISTEN:
            self.personalizer.update(current_emotion_probs(obs), action, reward)
