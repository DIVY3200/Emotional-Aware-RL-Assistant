import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from emotion_rl.emotions import ACTION_TO_INDEX, EMOTION_TO_INDEX, LISTEN, NUM_EMOTIONS
from emotion_rl.env import (
    FRAME_SIZE,
    EmotionSupportEnv,
    ObservationBuilder,
    observation_size,
)
from emotion_rl.policies import current_emotion_probs
from emotion_rl.simulator import PERSONA_NAMES

E = EMOTION_TO_INDEX


def test_gymnasium_api_compliance():
    check_env(EmotionSupportEnv(), skip_render_check=True)


def test_stable_baselines3_env_check():
    from stable_baselines3.common.env_checker import check_env as sb3_check

    sb3_check(EmotionSupportEnv(), warn=True)


@pytest.mark.parametrize("history", [1, 4])
def test_observation_shape_and_bounds(history):
    env = EmotionSupportEnv(history_len=history)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (observation_size(history),)
    for _ in range(25):
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_episode_length():
    env = EmotionSupportEnv(max_steps=7)
    env.reset(seed=0)
    for t in range(7):
        _, _, terminated, truncated, _ = env.step(LISTEN)
        assert not terminated
        assert truncated == (t == 6)


def test_reward_rewards_improvement_not_just_sum():
    """Regression test for v1's `delta = curr + prev` bug (happy -> sad was rewarded +1)."""
    env = EmotionSupportEnv()
    listen = LISTEN
    assert env.transition_reward(E["sad"], E["happy"], listen) > 0
    assert env.transition_reward(E["happy"], E["sad"], listen) < 0
    assert env.transition_reward(E["sad"], E["happy"], listen) > env.transition_reward(
        E["happy"], E["happy"], listen) > env.transition_reward(E["sad"], E["sad"], listen)


def test_talking_and_repeating_cost_something():
    env = EmotionSupportEnv()
    n = E["neutral"]
    comfort = ACTION_TO_INDEX["comfort"]
    silent = env.transition_reward(n, n, LISTEN)
    spoken = env.transition_reward(n, n, comfort)
    repeated = env.transition_reward(n, n, comfort, repeated=True)
    assert silent > spoken > repeated


def test_seeding_is_reproducible():
    a, b = EmotionSupportEnv(), EmotionSupportEnv()
    obs_a, info_a = a.reset(seed=123)
    obs_b, info_b = b.reset(seed=123)
    assert np.allclose(obs_a, obs_b) and info_a["persona"] == info_b["persona"]
    for action in [0, 1, 5, 4, 2]:
        ra = a.step(action)
        rb = b.step(action)
        assert np.allclose(ra[0], rb[0]) and ra[1] == rb[1]


def test_persona_option_and_privileged_observation():
    env = EmotionSupportEnv(privileged=True)
    obs, info = env.reset(seed=0, options={"persona": "achiever"})
    assert info["persona"] == "achiever"
    assert obs.shape == (observation_size(4) + len(PERSONA_NAMES),)
    assert obs[PERSONA_NAMES.index("achiever")] == 1.0
    # the most recent turn is still at the end, so policies can read it
    assert np.isclose(current_emotion_probs(obs).sum(), 1.0, atol=1e-5)


def test_current_emotion_probs_reads_latest_turn():
    builder = ObservationBuilder(history_len=3)
    first = np.eye(NUM_EMOTIONS, dtype=np.float32)[0]
    latest = np.eye(NUM_EMOTIONS, dtype=np.float32)[4]
    builder.push(first)
    builder.record_action(ACTION_TO_INDEX["comfort"])
    obs = builder.push(latest)
    assert np.allclose(current_emotion_probs(obs), latest)
    last_frame = obs[-1 - FRAME_SIZE:-1]
    assert last_frame[NUM_EMOTIONS + ACTION_TO_INDEX["comfort"]] == 1.0


def test_onehot_mode_and_fixed_time():
    builder = ObservationBuilder(history_len=1, observation_mode="onehot", fixed_time_frac=0.5)
    probs = np.array([0.1, 0.0, 0.2, 0.5, 0.1, 0.05, 0.05], dtype=np.float32)
    obs = builder.push(probs)
    assert np.allclose(current_emotion_probs(obs), np.eye(NUM_EMOTIONS)[3])
    assert obs[-1] == 0.5


def test_info_has_hidden_state_for_analysis():
    env = EmotionSupportEnv()
    env.reset(seed=1)
    _, _, _, _, info = env.step(ACTION_TO_INDEX["playful"])
    for key in ("true_emotion", "observed_emotion", "persona", "valence", "spoke", "fatigue"):
        assert key in info
