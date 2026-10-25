import numpy as np

from emotion_rl.bandits import (
    LinearThompsonPolicy,
    LinUCBPolicy,
    OnlinePersonalizer,
    PersonalizedPolicy,
    train_bandit,
)
from emotion_rl.emotions import (
    ACTION_TO_INDEX,
    EMOTION_TO_INDEX,
    LISTEN,
    NUM_ACTIONS,
    NUM_EMOTIONS,
    V1_RULE_TABLE,
    one_hot,
)
from emotion_rl.env import EmotionSupportEnv, ObservationBuilder
from emotion_rl.evaluate import run_episodes
from emotion_rl.policies import (
    GreedyOraclePolicy,
    RandomPolicy,
    RuleCooldownPolicy,
    RuleTablePolicy,
)


def obs_for(emotion):
    return ObservationBuilder().push(one_hot(EMOTION_TO_INDEX[emotion], NUM_EMOTIONS))


def test_rule_table_reproduces_v1():
    policy = RuleTablePolicy()
    for emotion, action in V1_RULE_TABLE.items():
        obs = ObservationBuilder().push(one_hot(emotion, NUM_EMOTIONS))
        assert policy.act(obs) == action


def test_cooldown_alternates_speaking_and_listening():
    policy = RuleCooldownPolicy(cooldown=2)
    actions = [policy.act(obs_for("sad")) for _ in range(6)]
    assert actions == [ACTION_TO_INDEX["comfort"], LISTEN, LISTEN] * 2


def test_oracle_beats_random_on_the_same_users():
    env = EmotionSupportEnv()
    oracle = run_episodes(GreedyOraclePolicy(), env, 60)["return"].mean()
    random = run_episodes(RandomPolicy(0), env, 60)["return"].mean()
    assert oracle > random + 1.0


class _ToyEnv:
    """Contextual bandit: when the user is sad, only 'comfort' pays off."""


def test_linucb_learns_a_simple_rule():
    rng = np.random.default_rng(0)
    policy = LinUCBPolicy(alpha=0.3)
    contexts = {e: obs_for(e) for e in ("sad", "happy")}
    best = {"sad": ACTION_TO_INDEX["comfort"], "happy": ACTION_TO_INDEX["playful"]}
    for _ in range(600):
        emotion = "sad" if rng.random() < 0.5 else "happy"
        action = policy.act(contexts[emotion])
        reward = 1.0 if action == best[emotion] else 0.0
        policy.update(contexts[emotion], action, reward + 0.1 * rng.standard_normal())
    policy.freeze()
    assert policy.act(contexts["sad"]) == best["sad"]
    assert policy.act(contexts["happy"]) == best["happy"]


def test_bandits_train_in_the_simulator():
    env = EmotionSupportEnv()
    for policy in (LinUCBPolicy(), LinearThompsonPolicy()):
        trained = train_bandit(policy, env, 500)
        assert trained.frozen
        assert 0 <= trained.act(env.reset(seed=0)[0]) < NUM_ACTIONS


class _FixedPrior:
    """Stands in for a trained PPO: prefers 'comfort' a little."""

    name = "prior"

    def reset(self):
        pass

    def action_probs(self, obs):
        p = np.full(NUM_ACTIONS, 0.1)
        p[ACTION_TO_INDEX["comfort"]] = 0.5
        return p / p.sum()


def test_personalizer_moves_towards_what_works_for_this_user():
    policy = PersonalizedPolicy(_FixedPrior(), OnlinePersonalizer(seed=0))
    obs = obs_for("sad")
    playful, comfort = ACTION_TO_INDEX["playful"], ACTION_TO_INDEX["comfort"]
    assert policy.personalizer.choose(_FixedPrior().action_probs(obs),
                                      one_hot(4, NUM_EMOTIONS), sample=False) == comfort
    for _ in range(30):  # this user loves jokes and dislikes comfort
        policy.update(obs, playful, 1.0)
        policy.update(obs, comfort, -0.5)
    assert policy.personalizer.choose(_FixedPrior().action_probs(obs),
                                      one_hot(4, NUM_EMOTIONS), sample=False) == playful


def test_personalizer_never_overrides_listen():
    prior = np.full(NUM_ACTIONS, 0.05)
    prior[LISTEN] = 0.75
    p = OnlinePersonalizer()
    for _ in range(50):  # this user strongly prefers comfort...
        p.update(one_hot(4, NUM_EMOTIONS), ACTION_TO_INDEX["comfort"], 3.0)
    assert p.preferences()["comfort"] > 0.3
    assert p.choose(prior, one_hot(4, NUM_EMOTIONS)) == LISTEN  # ...but PPO decides when


def test_personalizer_shares_what_it_learns_across_emotions():
    """Learning that a user likes jokes when sad should carry over to when they are angry."""
    p = OnlinePersonalizer(seed=0)
    for _ in range(30):
        p.update(one_hot(EMOTION_TO_INDEX["sad"], NUM_EMOTIONS), ACTION_TO_INDEX["playful"], 1.0)
    adv = p.advantage(one_hot(EMOTION_TO_INDEX["angry"], NUM_EMOTIONS), sample=False)
    assert int(np.argmax(adv)) == ACTION_TO_INDEX["playful"]


def test_personalizer_profile_roundtrip(tmp_path):
    p = OnlinePersonalizer()
    p.update(one_hot(4, NUM_EMOTIONS), 1, 0.8)
    path = tmp_path / "user.json"
    p.save(path)
    q = OnlinePersonalizer.load(path)
    assert np.allclose(p.mean, q.mean) and q.n_updates == 1
