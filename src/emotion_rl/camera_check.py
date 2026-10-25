"""Guided camera and emotion-detector check (about 2 minutes).

    python -m emotion_rl.camera_check
    python -m emotion_rl.camera_check --camera 1 --mute

The window (and a voice) asks you to make a sequence of expressions, records
what the detector sees during each one, and prints how well it matched, plus
face-detection rate, latency, label flicker and lighting. Use it to check a new
webcam, lighting or detector backend before running the assistant or a study
session. Results go to data/camera_check/ (no images are saved).
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from emotion_rl.emotion_models import (
    DeepFaceCropClassifier,
    accuracy,
    fit_calibration,
    make_classifier,
    save_detector_profile,
)
from emotion_rl.emotions import EMOTION_COLORS, EMOTIONS

# Targets: an emotion, None (nobody in view) or NATURAL (talking normally: should read as
# mostly neutral; used to measure false alarms, which posed expressions cannot reveal).
NATURAL = "natural"
PHASES = [
    ("Relax your face and look at the camera", "neutral"),
    ("Talk normally: count slowly from one to ten out loud", NATURAL),
    ("Give a big smile", "happy"),
    ("Look sad: let your face drop, inner eyebrows up, look down", "sad"),
    ("Look surprised: eyebrows up, jaw dropped into an O", "surprise"),
    ("Look angry: frown hard and press your lips together", "angry"),
    ("Look scared: eyes very wide, brows pulled together, lips stretched sideways", "fear"),
    ("Look disgusted: wrinkle your nose as if something smells bad", "disgust"),
    ("Move out of the camera's view", None),
    ("Come back, relax and look at the camera", "neutral"),
]
CALIBRATION_MIN_GAIN = 0.10  # held-out macro recall it must add...
CALIBRATION_MAX_FALSE_ALARMS = 0.15  # ...without misreading natural talking more than this
PHASE_SECONDS = 8.0
SETTLE_SECONDS = 2.0  # reaction time at the start of each phase, not scored
READY_TIMEOUT = 600.0


def phase_metrics(records, target):
    if not records:
        return {"target": target, "n": 0}
    found = [r for r in records if r["face_found"]]
    out = {"target": target or "no face", "n": len(records),
           "face_found_rate": len(found) / len(records)}
    if target is None:
        return out
    if target == NATURAL:
        if found:
            out["false_alarm_rate"] = float(np.mean([r["emotion"] != "neutral" for r in found]))
            out["labels_seen"] = dict(Counter(r["emotion"] for r in found).most_common(3))
        return out
    if found:
        out["stable_label_match"] = float(np.mean([r["emotion"] == target for r in found]))
        out["raw_top1_match"] = float(np.mean([r["raw_top"] == target for r in found]))
        out["mean_target_prob"] = float(np.mean([r["raw_probs"][EMOTIONS.index(target)]
                                                 for r in found]))
        out["labels_seen"] = dict(Counter(r["emotion"] for r in found).most_common(3))
    return out


def summarise(records, phase_windows, brightness, frame_size):
    phases = []
    for (instruction, target), (start, end) in zip(PHASES, phase_windows):
        window = [r for r in records if start + SETTLE_SECONDS <= r["t"] < end]
        phases.append({"instruction": instruction, **phase_metrics(window, target)})
    expression = [p for p in phases
                  if p.get("target") not in (None, "no face", "neutral", NATURAL)]
    latencies = [r["latency_ms"] for r in records]
    neutral_windows = [(s, e) for (s, e), (_, tgt) in zip(phase_windows, PHASES)
                       if tgt == "neutral"]
    neutral = [r for r in records for s, e in neutral_windows
               if s + SETTLE_SECONDS <= r["t"] < e and r["face_found"]]
    switches = sum(a["emotion"] != b["emotion"] for a, b in zip(neutral, neutral[1:]))
    neutral_seconds = max(len(neutral_windows) * (PHASE_SECONDS - SETTLE_SECONDS), 1e-9)
    away = next((p for p in phases if p["target"] == "no face"), None)  # None if stopped early
    present = [p["face_found_rate"] for p in phases if p.get("target") != "no face" and p["n"]]
    return {
        "frame_size": frame_size,
        "mean_brightness": float(np.mean(brightness)) if brightness else None,
        "detections_per_second": len(records) / max(phase_windows[-1][1] - phase_windows[0][0], 1),
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "face_found_rate_when_present": float(np.mean(present)) if present else 0.0,
        "no_face_correctly_reported": (1.0 - away.get("face_found_rate", 0.0)) if away else None,
        "expressions_recognised": sum(p.get("stable_label_match", 0) >= 0.5 for p in expression),
        "expressions_total": len(expression),
        "neutral_label_switches_per_minute": switches / neutral_seconds * 60,
        "phases": phases,
    }


def model_comparison(records, phase_windows):
    """Per emotion model: macro recall on held-out data, raw and after calibration, and
    whether the calibration is safe to use.

    Within each expression the first half of the detections fits the per-user
    calibration and the second half tests it. Posed expressions alone are not enough:
    in the first live test a calibration that lifted posed "sad" from 0% to 80% also
    turned 54% of the user's neutral moments into false "disgust". So calibration is
    only enabled if it adds CALIBRATION_MIN_GAIN macro recall AND keeps false alarms
    during natural talking at most CALIBRATION_MAX_FALSE_ALARMS (or no worse than
    +5 points over the uncalibrated model).
    """
    rows, natural = {}, {}  # model -> [(is_test, probs, target index)] / [probs]
    for (_, target), (start, end) in zip(PHASES, phase_windows):
        if target is None:
            continue
        phase = [r for r in records if start + SETTLE_SECONDS <= r["t"] < end and r["face_found"]]
        half = len(phase) // 2
        for k, r in enumerate(phase):
            for model, probs in r["models"].items():
                if target == NATURAL:
                    natural.setdefault(model, []).append(probs)
                else:
                    rows.setdefault(model, []).append((k >= half, probs, EMOTIONS.index(target)))

    def macro_recall(pairs, bias):
        per_emotion = {}
        for e in sorted({y for _, y in pairs}):
            subset = [(p, y) for p, y in pairs if y == e]
            per_emotion[EMOTIONS[e]] = accuracy([p for p, _ in subset], [y for _, y in subset],
                                                bias)
        return float(np.mean(list(per_emotion.values()))), per_emotion

    results = {}
    for model, data in rows.items():
        train = [(p, y) for is_test, p, y in data if not is_test]
        test = [(p, y) for is_test, p, y in data if is_test]
        if len(train) < 5 or len(test) < 5:
            continue
        bias = fit_calibration([p for p, _ in train], [y for _, y in train])
        raw_score, raw_per = macro_recall(test, None)
        cal_score, cal_per = macro_recall(test, bias)
        neutral = EMOTIONS.index("neutral")
        talk = natural.get(model, [])
        raw_fa = (1 - accuracy(talk, [neutral] * len(talk))) if talk else None
        cal_fa = (1 - accuracy(talk, [neutral] * len(talk), bias)) if talk else None
        safe = cal_fa is not None and cal_fa <= max(CALIBRATION_MAX_FALSE_ALARMS, raw_fa + 0.05)
        enabled = safe and cal_score - raw_score >= CALIBRATION_MIN_GAIN
        results[model] = {
            "heldout_macro_recall_raw": raw_score, "heldout_macro_recall_calibrated": cal_score,
            "per_emotion_raw": raw_per, "per_emotion_calibrated": cal_per,
            "natural_false_alarms_raw": raw_fa, "natural_false_alarms_calibrated": cal_fa,
            "calibration_enabled": bool(enabled),
            "score": cal_score if enabled else raw_score,
            "bias_all_data": fit_calibration([p for _, p, _ in data],
                                             [y for _, _, y in data]).tolist(),
        }
    # On a tie prefer the model that needs no calibration: simpler and less fragile
    best = max(results, key=lambda m: (round(results[m]["score"], 3),
                                       not results[m]["calibration_enabled"]), default=None)
    return results, best


def print_comparison(results, best):
    print("\n=== Emotion models on your face (held-out half of each expression) ===")
    emotions = list(next(iter(results.values()))["per_emotion_raw"]) if results else []
    header = " ".join(f"{e[:7]:>9s}" for e in emotions)
    print(f"{'model':10s} {'raw':>5s} {'calib.':>6s} {'false alarms':>13s}   {header}")
    for model, r in results.items():
        cells = " ".join(f"{r['per_emotion_raw'][e]:4.0%}>{r['per_emotion_calibrated'][e]:4.0%}"
                         for e in emotions)
        fa = ("    n/a     " if r["natural_false_alarms_raw"] is None else
              f"{r['natural_false_alarms_raw']:4.0%}>{r['natural_false_alarms_calibrated']:4.0%}")
        print(f"{model:10s} {r['heldout_macro_recall_raw']:5.0%} "
              f"{r['heldout_macro_recall_calibrated']:6.0%} {fa:>13s}   {cells}")
    print("(before > after calibration; false alarms = talking misread as an emotion)")
    if best:
        r = results[best]
        print(f"Best for you: {best} "
              + ("WITH calibration" if r["calibration_enabled"] else
                 "WITHOUT calibration (it did not add enough, or caused false alarms)"))


def advice(summary):
    tips = []
    if summary["mean_brightness"] is not None and summary["mean_brightness"] < 70:
        tips.append("The image is dark: face a window or a lamp (light in front of you, "
                    "not behind).")
    if summary["face_found_rate_when_present"] < 0.9:
        tips.append("Your face was often not found: sit 50-80 cm from the camera, centred, "
                    "with nothing covering your face.")
    no_face = summary["no_face_correctly_reported"]  # None if the session stopped before it
    if no_face is not None and no_face < 0.8:
        tips.append("A 'face' was reported when you left: check for posters or photos "
                    "in view, or try --detector-backend retinaface.")
    if summary["expressions_recognised"] < summary["expressions_total"] / 2:
        tips.append("Few posed expressions were recognised. That is common for fear and "
                    "disgust; exaggerate expressions, or train your own CNN on FER2013.")
    return tips or ["Camera, lighting and detector look good."]


def print_report(summary):
    print("\n=== Camera check ===")
    print(f"frame {summary['frame_size']}, brightness {summary['mean_brightness']:.0f}/255, "
          f"{summary['detections_per_second']:.1f} detections/s, latency p50 "
          f"{summary['latency_ms_p50']:.0f} ms / p95 {summary['latency_ms_p95']:.0f} ms")
    print(f"{'asked for':10s} {'face':>5s} {'stable':>7s} {'raw':>5s} {'p(target)':>9s}  seen")
    for p in summary["phases"]:
        if not p["n"]:
            print(f"{p['target']:10s} (no detections)")
            continue
        if p["target"] == NATURAL:
            print(f"{'talking':10s} {p['face_found_rate']:5.0%}  misread as an emotion "
                  f"{p.get('false_alarm_rate', 0):.0%} of the time  {p.get('labels_seen', {})}")
            continue
        if p["target"] == "no face":
            print(f"{'no face':10s} {p['face_found_rate']:5.0%}  (should be ~0%)")
            continue
        print(f"{p['target']:10s} {p['face_found_rate']:5.0%} "
              f"{p.get('stable_label_match', 0):7.0%} "
              f"{p.get('raw_top1_match', 0):5.0%} {p.get('mean_target_prob', 0):9.2f}  "
              f"{p.get('labels_seen', {})}")
    print(f"expressions recognised: {summary['expressions_recognised']}/"
          f"{summary['expressions_total']} (stable label correct >= 50% of the time)")
    print(f"label switches while neutral: {summary['neutral_label_switches_per_minute']:.1f}/min")
    for tip in advice(summary):
        print("-", tip)


# ─── Live run ──────────────────────────────────────────────────────────────────
def _draw(frame, lines, progress, detection, banner_color=(40, 40, 40)):
    import cv2

    from emotion_rl.detector import draw_detection

    h, w = frame.shape[:2]
    frame = draw_detection(frame, detection)
    cv2.rectangle(frame, (0, 0), (w, 34 + 34 * len(lines)), banner_color, -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (14, 38 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
    if progress is not None:
        cv2.rectangle(frame, (0, h - 12), (int(w * progress), h), (0, 200, 255), -1)
    if detection is not None:
        x0, y0 = w - 190, 60 + 34 * len(lines)
        for i, name in enumerate(EMOTIONS):
            y = y0 + i * 20
            cv2.putText(frame, name, (x0, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            cv2.rectangle(frame, (x0 + 70, y), (x0 + 70 + int(110 * detection.probs[i]), y + 13),
                          EMOTION_COLORS[name], -1)
        if not detection.face_found:
            cv2.putText(frame, "no face", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
    return frame


def run(camera=0, backend="auto", detector_backend="yunet", mute=False, out_dir=None,
        user="anonymous", model="hsemotion", compare=("deepface",), speaker="auto"):
    import cv2

    from emotion_rl.assistant.speech import SilentSpeech, SpeechEngine
    from emotion_rl.detector import DetectionWorker, EmotionDetector, open_camera

    cap = open_camera(camera, backend)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open camera {camera}")
    speech = SilentSpeech() if mute else SpeechEngine(output=speaker)
    try:
        primary = make_classifier(model)
    except Exception as exc:  # e.g. hsemotion-onnx not installed
        print(f"[Camera check] model '{model}' unavailable ({exc}); using deepface")
        model, primary = "deepface", None
    extra = {}
    for name in compare:
        if name == model:
            continue
        try:
            extra[name] = (make_classifier(name) if name != "deepface"
                           else DeepFaceCropClassifier())
        except Exception as exc:
            print(f"[Camera check] skipping model '{name}': {exc}")
    worker = DetectionWorker(EmotionDetector(detector_backend=detector_backend,
                                             classifier=primary,
                                             extra_classifiers=extra)).start()
    window = "STEP 1 of 2 - camera check (starts by itself)"
    face_since, countdown_from = None, None
    records, brightness, frame_size = [], [], None
    last_ts, state, ready_at, phase, phase_start, windows = 0.0, "loading", None, -1, 0.0, []
    started, shown, partial = time.time(), False, False

    def log(message):
        print(f"[Camera check {time.time() - started:6.1f}s] {message}", flush=True)

    log(f"camera {camera} opened at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}; loading models")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("[ERROR] Camera stopped sending frames")
            frame = cv2.flip(frame, 1)
            frame_size = frame_size or [int(frame.shape[1]), int(frame.shape[0])]
            worker.submit(frame)
            det = worker.latest
            now = time.time()
            key = cv2.waitKey(1) & 0xFF
            closed = shown and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            if key == ord("q") or closed:
                log(f"stopped by user ({'window closed' if closed else 'Q'}) during {state}"
                    + (f", phase {phase + 1}/{len(PHASES)}" if state == "running" else ""))
                if state == "running" and windows:
                    partial = True  # keep the phases that were completed
                    break
                return None

            if state == "loading":
                if det is not None:
                    state, ready_at = "ready", now
                    log("models loaded; waiting for a steady face (or SPACE)")
                    speech.speak("Ready. Look at the camera and the check will start by itself.")
                lines, progress = ["Loading face and emotion models...", "(about 20 s)"], None
            elif state == "ready":
                # Start by itself once a face has been visible for 2 s (no key needed), with a
                # spoken 3-2-1 countdown; SPACE starts at once.
                face_now = det is not None and det.face_found
                face_since = (face_since or now) if face_now else None
                if countdown_from is None and face_since and now - face_since >= 2.0:
                    countdown_from = now
                    speech.speak("Starting in three, two, one.")
                if not face_now:
                    countdown_from = None
                if key == ord(" ") or (countdown_from and now - countdown_from >= 3.5):
                    state, phase, phase_start = "running", 0, now
                    log("started " + ("with SPACE" if key == ord(" ") else "automatically"))
                    speech.speak(PHASES[0][0])
                elif now - ready_at > READY_TIMEOUT:
                    log("no face seen and SPACE not pressed; exiting")
                    return None
                if countdown_from:
                    lines = [f"Starting in {max(3 - int(now - countdown_from), 1)}...",
                             "Hold each expression until the bar fills"]
                else:
                    lines = ["Sit 50-80 cm away, face the light.",
                             "Starts by itself when your face is seen (Q quits)"]
                progress = None
            else:
                if det is not None and det.timestamp != last_ts:
                    last_ts = det.timestamp
                    records.append({
                        "t": det.timestamp, "phase": phase, "face_found": det.face_found,
                        "emotion": det.emotion, "confidence": det.confidence,
                        "raw_top": EMOTIONS[int(np.argmax(det.raw_probs))],
                        "raw_probs": [round(float(p), 4) for p in det.raw_probs],
                        "latency_ms": det.latency_ms,
                        "models": {model: [round(float(p), 4) for p in det.raw_probs],
                                   **{k: [round(float(p), 4) for p in v]
                                      for k, v in det.extra_probs.items()}}})
                if int(now * 2) % 2 == 0:
                    brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
                if now - phase_start >= PHASE_SECONDS:
                    windows.append((phase_start, now))
                    phase += 1
                    if phase == len(PHASES):
                        break
                    phase_start = now
                    log(f"phase {phase + 1}: {PHASES[phase][0]}")
                    speech.speak(PHASES[phase][0])
                lines = [f"{phase + 1}/{len(PHASES)}: {PHASES[phase][0]}"]
                progress = (now - phase_start) / PHASE_SECONDS
            cv2.imshow(window, _draw(frame, lines, progress, det))
            if not shown:  # keep it in front: it is launched from a background process
                cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
            shown = True
        speech.speak("Stopped early." if partial else "Done. Thank you!")
        time.sleep(1.5)
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()

    out_dir = out_dir or os.path.join("data", "camera_check")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime("check_%Y%m%d-%H%M%S.json"))
    raw = {"records": records, "windows": windows, "brightness": brightness,
           "frame_size": frame_size}
    with open(path, "w") as f:  # raw data first, so a session is never lost
        json.dump(raw, f)
    summary = summarise(records, windows, brightness, frame_size)
    summary.update({"camera": camera, "detector_backend": detector_backend, "partial": partial,
                    "total_seconds": round(time.time() - started, 1)})
    comparison, best = model_comparison(records, windows)
    summary["model_comparison"], summary["best_model"] = comparison, best
    with open(path, "w") as f:
        json.dump({"summary": summary, **raw}, f, indent=1)
    print_report(summary)
    if comparison:
        print_comparison(comparison, best)
        chosen = comparison[best]
        bias = (np.array(chosen["bias_all_data"]) if chosen["calibration_enabled"]
                else np.zeros(len(EMOTIONS)))
        profile = save_detector_profile(user, best, bias, chosen)
        print(f"[Camera check] your detector profile -> {profile} "
              f"(used by the assistant with --user {user})")
    print(f"[Camera check] saved -> {path}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Guided webcam and emotion-detector check")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", default="auto", help="auto, dshow, msmf, v4l2, any")
    parser.add_argument("--detector-backend", default="yunet")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--user", default="anonymous", help="profile name for the calibration")
    parser.add_argument("--model", default="hsemotion", help="emotion model shown on screen")
    parser.add_argument("--compare", nargs="*", default=["deepface"],
                        help="other emotion models scored on the same frames")
    parser.add_argument("--speaker", default="auto", help="audio output name (or part of it)")
    args = parser.parse_args(argv)
    if run(args.camera, args.backend, args.detector_backend, args.mute, user=args.user,
           model=args.model, compare=args.compare, speaker=args.speaker) is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
