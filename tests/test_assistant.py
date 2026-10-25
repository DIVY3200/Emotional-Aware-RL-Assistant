import json

import numpy as np
import pytest

from emotion_rl.assistant.latency import LatencyTracker
from emotion_rl.assistant.llm import FALLBACK_RESPONSES, ResponseGenerator, build_prompt
from emotion_rl.assistant.safety import WellbeingMonitor, detect_crisis, sanitize_response
from emotion_rl.assistant.session_log import InteractionLogger, live_reward
from emotion_rl.config import load_config
from emotion_rl.emotions import ACTIONS, EMOTION_TO_INDEX, EMOTIONS, LISTEN, NUM_EMOTIONS, one_hot
from emotion_rl.env import EmotionSupportEnv


@pytest.mark.parametrize("text", [
    "I want to die", "sometimes I think about killing myself", "I've been self-harming",
    "there is no reason to live", "they'd be better off without me",
])
def test_crisis_language_is_detected(text):
    assert detect_crisis(text)


@pytest.mark.parametrize("text", ["", None, "this exam is killing me", "I'm dying to see it",
                                  "I feel a bit sad today"])
def test_everyday_language_is_not_flagged(text):
    assert not detect_crisis(text)


def test_sanitize_response():
    raw = '"As an AI, I understand. You are doing great 😊! Keep going. One more thing."'
    clean = sanitize_response(raw)
    assert "AI" not in clean and "😊" not in clean
    assert clean.count(".") + clean.count("!") <= 2


def test_wellbeing_monitor_fires_once():
    now = [0.0]
    monitor = WellbeingMonitor(seconds=60, clock=lambda: now[0])
    sad = one_hot(EMOTION_TO_INDEX["sad"], NUM_EMOTIONS)
    fired = []
    for _ in range(20):
        fired.append(monitor.update(sad))
        now[0] += 10
    assert sum(fired) == 1
    assert not monitor.update(one_hot(EMOTION_TO_INDEX["happy"], NUM_EMOTIONS))


def test_prompt_uses_natural_instructions():
    prompt = build_prompt("sad", 70, "suggest_break", ["Earlier reply."], user_text="long day")
    assert "suggest_break" not in prompt and "short break" in prompt
    assert "Earlier reply." in prompt and "long day" in prompt


def test_fallback_covers_every_emotion_and_speaking_action():
    gen = ResponseGenerator(enabled=False)
    for emotion in EMOTIONS:
        for action in ACTIONS[:LISTEN]:
            text, source = gen.generate(emotion, 50, action)
            assert text == FALLBACK_RESPONSES[emotion][action] and source == "fallback"


class _FakeTypes:
    class GenerateContentConfig(dict):
        def __init__(self, **kw):
            super().__init__(kw)

    class ThinkingConfig(dict):
        def __init__(self, **kw):
            super().__init__(kw)


class _FakeModels:
    def __init__(self, delay=0.0, reject_thinking=False):
        self.delay, self.reject_thinking, self.calls = delay, reject_thinking, []

    def generate_content(self, model, contents, config=None):
        import time
        from types import SimpleNamespace

        self.calls.append((model, config))
        if self.reject_thinking and config is not None:
            raise ValueError("400 Thinking level MINIMAL is not supported for this model")
        time.sleep(self.delay)
        return SimpleNamespace(text="You are doing better than you think. 😊 Keep going!")


def _generator_with(fake_models, **kwargs):
    from types import SimpleNamespace

    gen = ResponseGenerator(enabled=False, **kwargs)
    gen.client, gen.types = SimpleNamespace(models=fake_models), _FakeTypes
    return gen


def test_llm_reply_is_sanitised_and_model_rejecting_thinking_is_retried():
    fake = _FakeModels(reject_thinking=True)
    gen = _generator_with(fake, models=["m1"])
    text, source = gen.generate("sad", 60, "comfort")
    assert source == "m1" and "😊" not in text
    assert fake.calls[0][1] is not None and fake.calls[1][1] is None
    gen.generate("sad", 60, "comfort")
    assert fake.calls[-1][1] is None, "remembers that this model rejects the setting"


def test_slow_llm_falls_back_to_local_reply():
    import time

    gen = _generator_with(_FakeModels(delay=1.0), models=["slow"], timeout_seconds=0.1)
    start = time.time()
    text, source = gen.generate("sad", 60, "comfort")
    assert source == "fallback" and time.time() - start < 0.5
    assert text == FALLBACK_RESPONSES["sad"]["comfort"]


def test_live_reward_matches_simulator_reward_for_certain_detections():
    env = EmotionSupportEnv()
    for prev, nxt, action in [(4, 3, 0), (3, 4, 5), (6, 6, 2)]:
        expected = env.transition_reward(prev, nxt, action)
        got = live_reward(one_hot(prev, NUM_EMOTIONS), one_hot(nxt, NUM_EMOTIONS), action, False)
        assert np.isclose(got, expected)
    assert live_reward(one_hot(6, 7), one_hot(6, 7), 0, False, feedback=1.0) > live_reward(
        one_hot(6, 7), one_hot(6, 7), 0, False)


def test_interaction_logger_writes_jsonl(tmp_path):
    logger = InteractionLogger(str(tmp_path), user="P1", condition="A", policy_name="ppo")
    logger.transition(1, one_hot(4, 7), 0, one_hot(3, 7), 0.9, 1.0, {"text": "hi"})
    logger.close({"turns": 1})
    with open(logger.path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert [r["event"] for r in lines] == ["session_start", "transition", "session_end"]
    assert lines[1]["user"] == "P1" and lines[1]["valence_after"] > lines[1]["valence_before"]


def test_latency_tracker():
    tracker = LatencyTracker()
    for ms in (10, 20, 30):
        tracker.add("llm", ms)
    with tracker.measure("policy"):
        pass
    summary = tracker.summary()
    assert summary["llm"]["n"] == 3 and summary["llm"]["p50_ms"] == 20
    assert "policy" in summary


def test_default_config_loads():
    cfg = load_config()
    assert cfg["assistant"]["decision_interval"] > 0
    assert set(cfg["study"]["conditions"]) == {"A", "B", "C"}
    assert load_config("does/not/exist.yaml")["assistant"]["camera"] == 0


def test_speech_skips_monitor_audio_by_default():
    from emotion_rl.assistant.speech import choose_output

    names = ["KA242Y P6 (HD Audio Driver for Display Audio)", "Speakers (Realtek(R) Audio)"]
    assert choose_output(names) == 1
    assert choose_output(names, "auto") == 1
    assert choose_output(names, "KA242Y") == 0  # explicit choice wins
    assert choose_output(["Display Audio only"]) == 0  # nothing better: keep it


def test_recording_stops_when_the_user_pauses():
    from emotion_rl.assistant.speech import end_of_utterance

    room, voice = 0.0004, 0.03
    nobody = [room] * 50
    assert end_of_utterance(nobody[:49]) == (False, False)
    assert end_of_utterance(nobody) == (True, False)  # 5 s and nobody spoke
    talk_then_pause = [room] * 5 + [voice] * 20 + [room] * 11
    assert end_of_utterance(talk_then_pause) == (False, True)  # 1.1 s of quiet: keep going
    assert end_of_utterance(talk_then_pause + [room]) == (True, True)  # 1.2 s: done
    starts_at_once = [voice] * 10 + [room] * 12  # noise floor still found afterwards
    assert end_of_utterance(starts_at_once) == (True, True)
    assert end_of_utterance([voice] * 120)[0]  # 12 s cap
    key_click = [room] * 10 + [0.05] + [room] * 39
    assert end_of_utterance(key_click) == (True, False)  # a click is not speech
    zeros_first = [0.0] * 3 + [room] * 47  # some mics send silent blocks first
    assert end_of_utterance(zeros_first) == (True, False)


def test_quiet_recordings_are_normalised():
    from emotion_rl.assistant.speech import normalise

    quiet = np.array([0.0, 0.01, -0.02], dtype=np.float32)
    assert np.isclose(np.abs(normalise(quiet)).max(), 0.9)
    silence = np.zeros(3, dtype=np.float32)
    assert np.all(normalise(silence) == 0)
