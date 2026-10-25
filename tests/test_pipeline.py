"""End-to-end smoke tests: training, evaluation helpers, CNN training and study analysis."""

import numpy as np
import pytest

from emotion_rl.assistant.session_log import InteractionLogger
from emotion_rl.emotions import NUM_EMOTIONS, one_hot
from emotion_rl.env import EmotionSupportEnv
from emotion_rl.evaluate import ci95, paired_test, run_episodes
from emotion_rl.policies import SB3Policy
from emotion_rl.train import train_one


def test_ppo_trains_and_reloads(tmp_path):
    env_kwargs = {"history_len": 2}
    config = train_one("ppo", 0, env_kwargs, timesteps=2048, run_dir=str(tmp_path), n_envs=2,
                       eval_every=1024, n_eval_episodes=3)
    assert (tmp_path / "best_model.zip").exists() and (tmp_path / "config.json").exists()
    assert config["algo"] == "ppo"
    policy = SB3Policy.from_path(str(tmp_path / "best_model"))
    frame = run_episodes(policy, EmotionSupportEnv(**env_kwargs), 3)
    assert len(frame) == 3 and np.isfinite(frame["return"]).all()
    probs = policy.action_probs(EmotionSupportEnv(**env_kwargs).reset(seed=0)[0])
    assert np.isclose(probs.sum(), 1.0)


def test_statistics_helpers():
    assert ci95([1.0]) == 0.0
    assert ci95([1.0, 2.0, 3.0]) > 0
    import pandas as pd

    a = [pd.DataFrame({"return": [2.0, 3.0, 4.0, 5.0]})]
    b = [pd.DataFrame({"return": [1.0, 1.5, 2.0, 2.5]})]
    result = paired_test(a, b)
    assert result["mean_diff"] > 0


def test_emotion_cnn_trains_on_a_tiny_dataset(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from emotion_rl.emotion_cnn import EmotionCNN, train

    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for emotion, brightness in (("happy", 200), ("sad", 40)):
            folder = tmp_path / "fer" / split / emotion
            folder.mkdir(parents=True)
            for i in range(12):
                pixels = np.clip(rng.normal(brightness, 20, (48, 48)), 0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(folder / f"{i}.png")
    out = tmp_path / "cnn.pt"
    report = train(str(tmp_path / "fer"), str(out), epochs=2, batch_size=8, width=8)
    assert out.exists() and "test_acc" in report
    probs = EmotionCNN.load(str(out)).predict_batch(np.zeros((2, 48, 48), dtype=np.uint8))
    assert probs.shape == (2, NUM_EMOTIONS) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_study_analysis_on_synthetic_logs(tmp_path):
    from emotion_rl.study import compare_conditions, effect_table, load_logs, session_table

    rng = np.random.default_rng(0)
    gain = {"A": 0.3, "B": 0.1, "C": -0.1}
    for participant in ("P1", "P2", "P3", "P4"):
        for condition, g in gain.items():
            log = InteractionLogger(str(tmp_path), user=participant, condition=condition)
            for step in range(10):
                before = one_hot(4, NUM_EMOTIONS) * 0.6 + 0.4 / NUM_EMOTIONS
                after = before.copy()
                after[3] += max(g + 0.05 * rng.standard_normal(), 0)
                after /= after.sum()
                log.transition(step, before, int(step % 6), after, g, 0.0)
            log.close()
    logs = load_logs(str(tmp_path))
    sessions = session_table(logs, {"A": "ppo_personalised", "B": "rule_cooldown", "C": "random"})
    assert len(sessions) == 12
    result = compare_conditions(sessions, "mean_reward")
    assert result["design"] == "within-subject" and "friedman_p" in result
    assert result["means"]["ppo_personalised"] > result["means"]["random"]
    assert not effect_table(logs).empty
