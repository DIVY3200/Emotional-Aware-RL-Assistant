"""Gymnasium environment: one episode = one support session with a simulated user.

Observation (flat float32 vector), built from the last `history_len` turns:
    per turn : detector probabilities (7) + last action one-hot (6)
               + share of recent turns where the assistant spoke (1)
               + how often the last action was repeated in a row (1)
    plus     : fraction of the session elapsed (1)
With history the agent can infer the hidden user type from how the user reacted
to earlier responses; with history_len=1 it cannot (memoryless ablation).

Reward, computed from the user's *true* emotion s → s':
    r = w_level · v(s') + w_delta · (v(s') − v(s)) − speak_cost·[spoke] − repeat_cost·[repeated]
The level term rewards keeping the user well over the whole session. The delta
term is reward shaping: with w_delta·(γ·v(s') − v(s)) it would be exactly
potential-based (Ng et al., 1999) and could not change the optimal policy; we
use the simpler undiscounted difference, which is close to that for γ near 1.
"""

from collections import deque
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from emotion_rl.emotions import (
    ACTIONS,
    EMOTIONS,
    LISTEN,
    NUM_ACTIONS,
    NUM_EMOTIONS,
    VALENCE,
    one_hot,
)
from emotion_rl.simulator import PERSONA_NAMES, UserSimulator

FRAME_SIZE = NUM_EMOTIONS + NUM_ACTIONS + 2
SPEAK_RATE_WINDOW = 4
MAX_REPEATS = 3


@dataclass
class RewardConfig:
    w_level: float = 1.0
    w_delta: float = 0.5
    speak_cost: float = 0.05
    repeat_cost: float = 0.10


def observation_size(history_len):
    return FRAME_SIZE * history_len + 1


def build_frame(emotion_probs, last_action, recent_spoke, repeat_count):
    """One turn of the observation. Shared by the environment and the live assistant."""
    speak_rate = float(np.mean(recent_spoke)) if len(recent_spoke) else 0.0
    return np.concatenate([
        np.asarray(emotion_probs, dtype=np.float32),
        one_hot(last_action, NUM_ACTIONS),
        np.array([speak_rate, min(repeat_count, MAX_REPEATS) / MAX_REPEATS], dtype=np.float32),
    ])


class ObservationBuilder:
    """Keeps the turn history and stacks it into the agent's observation vector."""

    def __init__(self, history_len=4, observation_mode="probs", max_steps=20,
                 fixed_time_frac=None):
        """fixed_time_frac: live sessions are open-ended, so the assistant shows the agent a
        constant mid-session time instead of a counter that would run past training's range."""
        if observation_mode not in ("probs", "onehot"):
            raise ValueError("observation_mode must be 'probs' or 'onehot'")
        self.history_len = history_len
        self.observation_mode = observation_mode
        self.max_steps = max_steps
        self.fixed_time_frac = fixed_time_frac
        self.reset()

    def reset(self):
        self.frames = deque([np.zeros(FRAME_SIZE, dtype=np.float32)] * self.history_len,
                            maxlen=self.history_len)
        self.recent_spoke = deque(maxlen=SPEAK_RATE_WINDOW)
        self.last_action = LISTEN
        self.repeat_count = 0
        self.step = 0

    def record_action(self, action):
        spoke = action != LISTEN
        if spoke and action == self.last_action:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.recent_spoke.append(float(spoke))
        self.last_action = action
        self.step += 1

    def push(self, emotion_probs):
        probs = np.asarray(emotion_probs, dtype=np.float32)
        if self.observation_mode == "onehot":
            probs = one_hot(int(np.argmax(probs)), NUM_EMOTIONS)
        self.frames.append(build_frame(probs, self.last_action, self.recent_spoke,
                                       self.repeat_count))
        time_frac = self.fixed_time_frac
        if time_frac is None:
            time_frac = min(self.step / max(self.max_steps, 1), 1.0)
        return np.concatenate([*self.frames, np.array([time_frac], dtype=np.float32)])


class EmotionSupportEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_steps=20,
        history_len=4,
        observation_mode="probs",
        personas=PERSONA_NAMES,
        mix_personas=False,
        detector_accuracy=0.65,
        confusion=None,
        reward=None,
        privileged=False,
        render_mode=None,
        **sim_kwargs,
    ):
        """
        privileged=True prepends the hidden user-type weights to the observation.
        Only for training an upper-bound "teacher" agent; never available live.
        """
        super().__init__()
        self.max_steps = max_steps
        self.privileged = privileged
        self.reward_config = reward if reward is not None else RewardConfig()
        if isinstance(self.reward_config, dict):
            self.reward_config = RewardConfig(**self.reward_config)
        self.render_mode = render_mode

        self.sim = UserSimulator(personas=personas, mix_personas=mix_personas,
                                 detector_accuracy=detector_accuracy, confusion=confusion,
                                 **sim_kwargs)
        self.obs_builder = ObservationBuilder(history_len, observation_mode, max_steps)

        size = observation_size(history_len) + (len(PERSONA_NAMES) if privileged else 0)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.step_count = 0
        self.last_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.sim.reset(self.np_random, persona=options.get("persona"),
                       keep_user=options.get("keep_user", False))
        self.obs_builder.reset()
        self.step_count = 0
        observed = self.sim.observe()
        self.last_info = self._info(observed, spoke=False, reward=0.0)
        return self._observation(observed), dict(self.last_info)

    def step(self, action):
        action = int(action)
        prev_emotion = self.sim.emotion
        repeated = action != LISTEN and action == self.sim.last_action
        next_emotion = self.sim.step(action)
        reward = self.transition_reward(prev_emotion, next_emotion, action, repeated)

        self.step_count += 1
        self.obs_builder.record_action(action)
        observed = self.sim.observe()
        obs = self._observation(observed)

        truncated = self.step_count >= self.max_steps
        self.last_info = self._info(observed, spoke=action != LISTEN, reward=reward)
        self.last_info["action"] = ACTIONS[action]
        self.last_info["prev_emotion"] = EMOTIONS[prev_emotion]
        return obs, reward, False, truncated, dict(self.last_info)

    def _observation(self, observed):
        obs = self.obs_builder.push(observed)
        if self.privileged:  # in front, so the most recent turn stays at the end
            obs = np.concatenate([self.sim.persona_weights, obs])
        return obs

    def transition_reward(self, prev_emotion, next_emotion, action, repeated=False):
        cfg = self.reward_config
        v_prev, v_next = float(VALENCE[prev_emotion]), float(VALENCE[next_emotion])
        reward = cfg.w_level * v_next + cfg.w_delta * (v_next - v_prev)
        if action != LISTEN:
            reward -= cfg.speak_cost
        if repeated:
            reward -= cfg.repeat_cost
        return float(reward)

    def _info(self, observed, spoke, reward):
        return {
            "true_emotion": EMOTIONS[self.sim.emotion],
            "observed_emotion": EMOTIONS[int(np.argmax(observed))],
            "persona": self.sim.persona_name,
            "valence": float(VALENCE[self.sim.emotion]),
            "fatigue": float(self.sim.fatigue),
            "spoke": spoke,
            "reward": reward,
            "step": self.step_count,
        }

    def render(self):
        info = self.last_info
        line = (f"[t={self.step_count:2d}] user={info.get('persona')} "
                f"emotion={info.get('true_emotion')} (seen as {info.get('observed_emotion')}) "
                f"action={info.get('action', '-')} reward={info.get('reward', 0.0):+.2f}")
        if self.render_mode == "ansi":
            return line
        print(line)
        return None

    def config(self):
        return {
            "max_steps": self.max_steps,
            "history_len": self.obs_builder.history_len,
            "observation_mode": self.obs_builder.observation_mode,
            "personas": list(self.sim.persona_names),
            "mix_personas": self.sim.mix_personas,
            "detector_accuracy": self.sim.detector_accuracy,
            "privileged": self.privileged,
            "reward": asdict(self.reward_config),
        }
