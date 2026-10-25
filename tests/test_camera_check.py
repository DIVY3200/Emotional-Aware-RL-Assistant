import numpy as np

from emotion_rl.camera_check import (
    NATURAL,
    PHASE_SECONDS,
    PHASES,
    advice,
    model_comparison,
    print_comparison,
    print_report,
    summarise,
)
from emotion_rl.emotions import EMOTIONS

AWAY = next(i for i, (_, target) in enumerate(PHASES) if target is None)


def as_emotion(target):
    """Natural talking should read as neutral; nobody in view has no emotion."""
    return "neutral" if target in (None, NATURAL) else target


def probs_for(label, peak=0.7):
    p = np.full(7, 0.05)
    p[EMOTIONS.index(label)] = peak
    return p / p.sum()


def synthetic_session(recognise=True):
    rng = np.random.default_rng(0)
    records, windows, t = [], [], 0.0
    for i, (_, target) in enumerate(PHASES):
        windows.append((t, t + PHASE_SECONDS))
        for k in range(int(PHASE_SECONDS * 5)):
            label = as_emotion(target) if recognise else "neutral"
            records.append({"t": t + k / 5, "phase": i, "face_found": target is not None,
                            "emotion": label, "confidence": 70.0, "raw_top": label,
                            "raw_probs": probs_for(label).tolist(),
                            "latency_ms": float(rng.uniform(60, 120))})
        t += PHASE_SECONDS
    return records, windows


def test_summary_of_a_good_session(capsys):
    records, windows = synthetic_session()
    summary = summarise(records, windows, [120.0] * 50, [1280, 720])
    assert summary["expressions_recognised"] == summary["expressions_total"] == 6
    assert summary["no_face_correctly_reported"] == 1.0
    assert summary["face_found_rate_when_present"] == 1.0
    assert summary["neutral_label_switches_per_minute"] == 0
    talking = next(p for p in summary["phases"] if p["target"] == NATURAL)
    assert talking["false_alarm_rate"] == 0.0
    assert advice(summary) == ["Camera, lighting and detector look good."]
    print_report(summary)
    assert "expressions recognised: 6/6" in capsys.readouterr().out


def test_advice_for_a_dark_room_and_unrecognised_expressions():
    records, windows = synthetic_session(recognise=False)
    summary = summarise(records, windows, [40.0] * 50, [640, 480])
    tips = " ".join(advice(summary))
    assert "dark" in tips and "posed expressions" in tips


def test_summary_of_a_session_stopped_early():
    records, windows = synthetic_session()
    kept = windows[:4]  # neutral, talking, happy, sad
    summary = summarise([r for r in records if r["t"] < kept[-1][1]], kept, [100.0], [1280, 720])
    assert summary["no_face_correctly_reported"] is None
    assert summary["expressions_total"] == 2
    advice(summary)
    print_report(summary)


def test_phantom_face_when_user_left_is_flagged():
    records, windows = synthetic_session()
    for r in records:
        if r["phase"] == AWAY:  # "move out of view", but a face is still reported
            r["face_found"] = True
    summary = summarise(records, windows, [100.0], [1280, 720])
    assert summary["no_face_correctly_reported"] == 0.0
    assert any("when you left" in tip for tip in advice(summary))


def _with_models(records, make):
    rng = np.random.default_rng(1)
    for r in records:
        noise = rng.uniform(0.9, 1.1, 7)
        r["models"] = {}
        for name, p in make(PHASES[r["phase"]][1]).items():
            q = p * noise
            r["models"][name] = (q / q.sum()).tolist()


def test_model_comparison_prefers_the_model_that_reads_this_face(capsys):
    """Model A reads sad as (slightly more) neutral; B reads everything right.
    B must win, and calibration must help A recover sad without hurting talking."""
    def make(target):
        bad = probs_for(as_emotion(target))
        if target == "sad":
            bad = np.full(7, 0.05)
            bad[EMOTIONS.index("neutral")], bad[EMOTIONS.index("sad")] = 0.45, 0.35
        return {"A": bad, "B": probs_for(as_emotion(target))}

    records, windows = synthetic_session()
    _with_models(records, make)
    results, best = model_comparison(records, windows)
    assert best == "B" and not results["B"]["calibration_enabled"]  # nothing to gain
    assert results["A"]["per_emotion_raw"]["sad"] == 0.0
    assert results["A"]["per_emotion_calibrated"]["sad"] > 0.5
    assert results["A"]["natural_false_alarms_calibrated"] == 0.0
    assert results["A"]["calibration_enabled"]
    print_comparison(results, best)
    assert "Best for you: B WITHOUT calibration" in capsys.readouterr().out


def test_calibration_that_causes_false_alarms_is_rejected():
    """The failure seen in the first live test: posed disgust reads as borderline neutral,
    so calibration boosts disgust, which then misreads natural talking as disgust."""
    def make(target):
        if target in ("disgust", NATURAL):
            p = np.full(7, 0.05)
            p[EMOTIONS.index("neutral")], p[EMOTIONS.index("disgust")] = 0.42, 0.36
            return {"A": p}
        return {"A": probs_for(as_emotion(target))}

    records, windows = synthetic_session()
    _with_models(records, make)
    results, best = model_comparison(records, windows)
    a = results["A"]
    assert a["per_emotion_calibrated"]["disgust"] > a["per_emotion_raw"]["disgust"]
    assert a["natural_false_alarms_calibrated"] > 0.15
    assert not a["calibration_enabled"]
    assert a["score"] == a["heldout_macro_recall_raw"]


def test_calibration_fit_and_apply():
    from emotion_rl.emotion_models import accuracy, apply_calibration, fit_calibration

    sad, neutral = EMOTIONS.index("sad"), EMOTIONS.index("neutral")
    probs = []
    for _ in range(20):
        p = np.full(7, 0.05)
        p[neutral], p[sad] = 0.45, 0.35
        probs.append(p / p.sum())
    targets = [sad] * 10 + [neutral] * 10
    neutral_probs = [np.eye(7)[neutral] * 0.8 + 0.2 / 7 for _ in range(10)]
    bias = fit_calibration(probs[:10] + neutral_probs, targets)
    assert accuracy(probs[:10], [sad] * 10, bias) == 1.0
    assert accuracy(neutral_probs, [neutral] * 10, bias) == 1.0
    assert np.isclose(apply_calibration(probs[0], bias).sum(), 1.0)
