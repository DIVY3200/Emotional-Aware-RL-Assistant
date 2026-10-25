import numpy as np
import pytest

from emotion_rl.emotions import ACTION_TO_INDEX, EMOTION_TO_INDEX, LISTEN, NUM_ACTIONS, NUM_EMOTIONS
from emotion_rl.simulator import (
    BASE_TRANSITIONS,
    PERSONA_NAMES,
    PERSONAS,
    START_DISTRIBUTION,
    UserSimulator,
    default_confusion,
)

SAD, HAPPY = EMOTION_TO_INDEX["sad"], EMOTION_TO_INDEX["happy"]


def make_sim(persona="comfort_seeker", **kwargs):
    sim = UserSimulator(effect_noise=0.0, **kwargs)
    sim.reset(np.random.default_rng(0), persona=persona)
    return sim


def test_tables_are_valid_distributions():
    assert np.allclose(BASE_TRANSITIONS.sum(axis=1), 1.0)
    assert (BASE_TRANSITIONS > 0).all(), "every emotion should be reachable (v1: disgust was not)"
    assert np.isclose(START_DISTRIBUTION.sum(), 1.0)
    conf = default_confusion(0.65)
    assert np.allclose(conf.sum(axis=1), 1.0)
    assert np.allclose(np.diag(conf), 0.65)


@pytest.mark.parametrize("name", PERSONA_NAMES)
def test_persona_tables(name):
    effects = PERSONAS[name].effects
    assert effects.shape == (NUM_EMOTIONS, NUM_ACTIONS)
    assert np.all(np.abs(effects) <= 1.0)
    assert np.all(effects[:, LISTEN] == 0.0)


def test_actions_change_the_next_emotion():
    """The core v1 flaw: the action had no influence on what happened next."""
    sim = make_sim("comfort_seeker")
    sim.emotion = SAD
    comfort = sim.transition_probs(ACTION_TO_INDEX["comfort"])
    playful = sim.transition_probs(ACTION_TO_INDEX["playful"])
    assert np.isclose(comfort.sum(), 1.0)
    assert comfort[HAPPY] > playful[HAPPY] + 0.05


def test_best_action_differs_between_user_types():
    best = {}
    for name in PERSONA_NAMES:
        sim = make_sim(name)
        sim.emotion = SAD
        best[name] = max(range(LISTEN), key=lambda a, s=sim: s.transition_probs(a)[HAPPY])
    assert len(set(best.values())) >= 3, best


def test_listen_is_neutral_and_recovers_fatigue():
    sim = make_sim()
    sim.emotion = SAD
    assert np.allclose(sim.transition_probs(LISTEN), BASE_TRANSITIONS[SAD])
    for _ in range(5):
        sim.step(ACTION_TO_INDEX["comfort"])
    tired = sim.fatigue
    assert tired > 0
    sim.step(LISTEN)
    assert sim.fatigue < tired


def test_repetition_and_fatigue_reduce_the_effect():
    sim = make_sim()
    sim.emotion = SAD
    comfort = ACTION_TO_INDEX["comfort"]
    fresh = sim.effective_effect(comfort)
    sim.last_action, sim.repeat_count = comfort, 2
    assert sim.effective_effect(comfort) < fresh
    sim.last_action, sim.repeat_count, sim.fatigue = LISTEN, 0, 1.0
    assert sim.effective_effect(comfort) < 0, "talking to an exhausted user backfires"


def test_detector_accuracy_matches_confusion():
    sim = UserSimulator(detector_accuracy=0.65)
    rng = np.random.default_rng(1)
    hits = []
    for _ in range(3000):
        sim.reset(rng)
        hits.append(np.argmax(sim.observe()) == sim.emotion)
    assert 0.6 < np.mean(hits) < 0.75


def test_perfect_detector_is_one_hot():
    sim = make_sim(detector_accuracy=None)
    obs = sim.observe()
    assert obs[sim.emotion] == 1.0 and obs.sum() == 1.0


def test_seeded_reset_is_reproducible():
    a, b = UserSimulator(), UserSimulator()
    a.reset(np.random.default_rng(42))
    b.reset(np.random.default_rng(42))
    assert a.persona_name == b.persona_name and a.emotion == b.emotion
    assert np.allclose(a.effects, b.effects)


def test_keep_user_keeps_the_same_effects():
    sim = UserSimulator()
    sim.reset(np.random.default_rng(3))
    effects, name = sim.effects.copy(), sim.persona_name
    sim.fatigue = 0.7
    sim.reset(np.random.default_rng(4), keep_user=True)
    assert sim.persona_name == name and np.allclose(sim.effects, effects)
    assert sim.fatigue == 0.0


def test_blended_users_have_two_weights():
    sim = UserSimulator(mix_personas=True)
    sim.reset(np.random.default_rng(5))
    assert sim.persona_name.startswith("mix:")
    assert np.isclose(sim.persona_weights.sum(), 1.0)
    assert (sim.persona_weights > 0).sum() == 2
