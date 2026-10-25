"""Detector logic with a fake DeepFace, so these tests need neither TensorFlow nor a webcam."""

import numpy as np

from emotion_rl.detector import EmotionDetector, deepface_probs
from emotion_rl.emotions import EMOTIONS


def fake_result(emotion, confidence=0.9, face_confidence=0.95, region=(10, 20, 100, 100)):
    scores = {e: (100 - confidence * 100) / 6 for e in EMOTIONS}
    scores[emotion] = confidence * 100
    x, y, w, h = region
    return {"emotion": scores, "dominant_emotion": emotion, "face_confidence": face_confidence,
            "region": {"x": x, "y": y, "w": w, "h": h}}


class FakeAnalyze:
    def __init__(self, results):
        self.results = list(results)
        self.frames = []

    def __call__(self, frame, **kwargs):
        self.frames.append(frame)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, list) else [result]


FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def test_frames_are_passed_as_bgr_unchanged():
    fake = FakeAnalyze([fake_result("happy")])
    EmotionDetector(analyze_fn=fake).detect(FRAME)
    assert fake.frames[0] is FRAME, "DeepFace expects BGR: the frame must not be converted"


def test_probabilities_follow_our_index_order():
    probs = deepface_probs(fake_result("sad", 0.7))
    assert np.isclose(probs.sum(), 1.0)
    assert EMOTIONS[int(np.argmax(probs))] == "sad"


def test_no_face_keeps_state_and_says_so():
    fake = FakeAnalyze([fake_result("happy")] * 3 + [fake_result("angry", face_confidence=0,
                                                                 region=(0, 0, 320, 240))])
    det = EmotionDetector(analyze_fn=fake, smoothing=1.0)
    for _ in range(3):
        det.detect(FRAME)
    result = det.detect(FRAME)
    assert not result.face_found
    assert result.emotion == "happy", "a whole-frame 'face' must not change the label"
    assert result.box == (10, 20, 100, 100), "the full-frame region must not enter the box"


def test_hysteresis_ignores_single_frame_flicker():
    seq = [fake_result("neutral")] * 5 + [fake_result("surprise")] + [fake_result("neutral")] * 2
    det = EmotionDetector(analyze_fn=FakeAnalyze(seq), smoothing=0.35, hysteresis=0.1)
    labels = [det.detect(FRAME).emotion for _ in seq]
    assert set(labels) == {"neutral"}


def test_sustained_change_switches_label_and_confidence_matches():
    seq = [fake_result("neutral")] * 3 + [fake_result("sad")] * 8
    det = EmotionDetector(analyze_fn=FakeAnalyze(seq))
    results = [det.detect(FRAME) for _ in seq]
    assert results[-1].emotion == "sad"
    last = results[-1]
    assert np.isclose(last.confidence, last.probs[EMOTIONS.index("sad")] * 100)


def test_largest_face_is_used():
    small = fake_result("sad", region=(0, 0, 20, 20))
    large = fake_result("happy", region=(50, 50, 150, 150))
    det = EmotionDetector(analyze_fn=FakeAnalyze([[small, large]]), smoothing=1.0)
    assert det.detect(FRAME).emotion == "happy"


def test_errors_are_reported_and_last_state_kept(capsys):
    det = EmotionDetector(analyze_fn=FakeAnalyze([fake_result("happy"), RuntimeError("boom")]),
                          smoothing=1.0)
    first = det.detect(FRAME)
    second = det.detect(FRAME)
    assert second is first
    assert "detection failed: boom" in capsys.readouterr().out
