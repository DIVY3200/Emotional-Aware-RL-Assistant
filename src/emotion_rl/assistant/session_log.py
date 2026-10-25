"""Interaction logging: every live decision becomes one JSON line.

Each record is a transition (observed emotion before, action, observed emotion
after, reward, explicit feedback). The logs feed:
* online personalisation (the per-user profile is updated from them live),
* the user-study analysis (`python -m emotion_rl.study`),
* future offline RL / simulator calibration with real data.

No images are ever stored; only emotion probabilities and the text spoken.
"""

import json
import os
import time
import uuid

import numpy as np

from emotion_rl.emotions import LISTEN, VALENCE


def expected_valence(emotion_probs):
    return float(np.asarray(emotion_probs) @ VALENCE)


def live_reward(prev_probs, next_probs, action, repeated, feedback=0.0, reward_config=None):
    """The simulator's reward, computed from detector probabilities instead of the true emotion,
    plus explicit user feedback (+1 thumbs up, -1 thumbs down)."""
    from emotion_rl.env import RewardConfig

    cfg = reward_config or RewardConfig()
    v_prev, v_next = expected_valence(prev_probs), expected_valence(next_probs)
    reward = cfg.w_level * v_next + cfg.w_delta * (v_next - v_prev)
    if action != LISTEN:
        reward -= cfg.speak_cost
    if repeated:
        reward -= cfg.repeat_cost
    return float(reward + feedback)


class InteractionLogger:
    def __init__(self, log_dir="data/interactions", user="anonymous", condition=None,
                 policy_name=None):
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.path = os.path.join(log_dir, f"{user}_{self.session_id}.jsonl")
        self.meta = {"session_id": self.session_id, "user": user, "condition": condition,
                     "policy": policy_name}
        self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 (kept open)
        self.write("session_start", {})

    def write(self, event, data):
        record = {"t": time.time(), "event": event, **self.meta, **data}
        self._file.write(json.dumps(record, default=_to_builtin) + "\n")
        self._file.flush()

    def transition(self, step, prev_probs, action, next_probs, reward, feedback, response=None):
        self.write("transition", {
            "step": step,
            "prev_probs": np.round(prev_probs, 4).tolist(),
            "action": int(action),
            "next_probs": np.round(next_probs, 4).tolist(),
            "valence_before": expected_valence(prev_probs),
            "valence_after": expected_valence(next_probs),
            "reward": reward,
            "feedback": feedback,
            "response": response,
        })

    def close(self, extra=None):
        self.write("session_end", extra or {})
        self._file.close()


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")
