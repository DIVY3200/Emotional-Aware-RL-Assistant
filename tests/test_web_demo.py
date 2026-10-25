"""Runs the Streamlit demo headless (no browser) and checks the simulator playground renders."""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DEMO = os.path.join(os.path.dirname(__file__), "..", "src", "emotion_rl", "web_demo.py")


def test_playground_runs_with_baselines_only(monkeypatch):
    monkeypatch.setenv("EMOTION_RL_MODEL", "does/not/exist.zip")
    app = AppTest.from_file(DEMO, default_timeout=120).run()
    assert not app.exception
    assert any("No trained model" in w.value for w in app.warning)
    assert len(app.metric) >= 1


def test_playground_with_a_trained_agent(monkeypatch, tmp_path):
    from emotion_rl.train import train_one

    train_one("ppo", 0, {"history_len": 1}, timesteps=1024, run_dir=str(tmp_path), n_envs=2,
              eval_every=1024, n_eval_episodes=2)
    monkeypatch.setenv("EMOTION_RL_MODEL", str(tmp_path / "best_model.zip"))
    app = AppTest.from_file(DEMO, default_timeout=120).run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "PPO (ours)" in labels
    app.selectbox[0].set_value("space_needer").run()
    assert not app.exception
