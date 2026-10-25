"""Integration test of the live assistant's decision loop, without a webcam, audio or API key."""

import json
import time

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("dotenv")

from emotion_rl.assistant.app import Assistant, wrap_text  # noqa: E402
from emotion_rl.config import load_config  # noqa: E402
from emotion_rl.detector import Detection  # noqa: E402
from emotion_rl.emotions import EMOTIONS, LISTEN, NUM_EMOTIONS, one_hot  # noqa: E402


def detection(emotion, conf=0.8):
    probs = np.full(NUM_EMOTIONS, (1 - conf) / (NUM_EMOTIONS - 1), dtype=np.float32)
    probs[EMOTIONS.index(emotion)] = conf
    return Detection(probs=probs, emotion=emotion, confidence=conf * 100, face_found=True,
                     box=(0, 0, 10, 10), timestamp=time.time())


@pytest.fixture
def assistant(tmp_path):
    cfg = load_config()
    cfg["assistant"].update(policy="rule_cooldown", mute=True, log_dir=str(tmp_path / "logs"),
                            profile_dir=str(tmp_path / "profiles"))
    cfg["llm"]["enabled"] = False
    a = Assistant(cfg, condition="B")
    yield a
    a.worker.stop()


def wait_for_reply(a, timeout=5.0):
    start = time.time()
    while a.thinking and time.time() - start < timeout:
        time.sleep(0.01)


def test_turns_are_logged_as_transitions(assistant):
    for emotion in ["sad", "sad", "neutral", "happy"]:
        assistant.decide(detection(emotion))
        wait_for_reply(assistant)
    assistant.logger.close()
    with open(assistant.logger.path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    transitions = [e for e in events if e["event"] == "transition"]
    assert len(transitions) == 3, "each turn after the first closes the previous one"
    assert transitions[0]["action"] != LISTEN and transitions[1]["action"] == LISTEN
    assert transitions[0]["response"]["source"] == "fallback"
    assert all(e["condition"] == "B" for e in events)


def test_feedback_key_changes_the_reward(assistant):
    assistant.decide(detection("neutral"))
    wait_for_reply(assistant)
    assistant.handle_key("G")  # Caps Lock on: must still count
    assistant.decide(detection("neutral"))
    assistant.logger.close()
    with open(assistant.logger.path, encoding="utf-8") as f:
        transition = [json.loads(x) for x in f if '"transition"' in x][0]
    assert transition["feedback"] == 1.0 and transition["reward"] > 0.5


def test_crisis_text_overrides_the_policy(assistant):
    assistant.user_text = "honestly I want to die"
    assistant.decide(detection("sad"))
    assert assistant.banner is not None and "14416" in assistant.response_text
    assert assistant.pending is None, "safety messages are not learned from"


def test_wrap_text_fits_width():
    lines = wrap_text("one two three four five six seven eight nine ten " * 3, 200, 0.5)
    assert len(lines) > 1 and all(line for line in lines)


def test_live_observation_matches_training_layout(assistant):
    obs = assistant.obs_builder.push(one_hot(4, NUM_EMOTIONS))
    assert obs.shape[0] == 15 * assistant.obs_builder.history_len + 1
    assert obs[-1] == 0.5


def test_every_input_gives_visible_feedback(assistant):
    assistant.handle_key("g")  # nothing to rate yet
    assert "Nothing to rate yet" in assistant.toast[0] and assistant.feedback == 0.0
    assistant.handle_key("t")  # voice input is off in this fixture
    assert "Voice input is off" in assistant.toast[0]
    assistant.handle_key("p")
    assert assistant.paused and assistant.toast[0] == "Paused"


def test_clicking_a_button_acts_like_its_key(assistant):
    import cv2

    assistant.buttons = [(10, 400, 110, 430, "g"), (120, 400, 220, 430, "b")]
    assistant._on_mouse(cv2.EVENT_LBUTTONDOWN, 150, 410, 0, None)
    assert assistant.clicked == "b"
    assistant.clicked = None
    assistant._on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 10, 0, None)  # not on a button
    assert assistant.clicked is None


def test_the_assistant_always_answers_when_the_user_speaks(assistant):
    assistant.decide(detection("sad"))  # rule + cooldown speaks now...
    wait_for_reply(assistant)
    assistant.user_text = "I had a long day at college"
    assistant.decide(detection("sad"))  # ...and would listen now, but the user spoke
    assert assistant.current_action != "listen"


class _BusyRecognizer:
    busy, listening, waiting = True, True, False


def test_no_turns_while_the_user_is_talking(assistant):
    assistant.recognizer = _BusyRecognizer()
    assistant.last_decision = 0.0
    assistant._maybe_decide(detection("fear"))  # a talking mouth often reads as fear
    assert assistant.turn == 0
    assistant.recognizer = None
    assistant._maybe_decide(detection("neutral"))
    assert assistant.turn == 1
