"""Measure the emotion detector instead of quoting a benchmark.

1. FER2013 test set (needs data/fer2013/test/<emotion>/*.jpg):
       python -m emotion_rl.detector_eval --fer data/fer2013/test --backends skip opencv
       python -m emotion_rl.detector_eval --custom-model models/emotion_cnn.pt
   Reports accuracy, macro-F1, per-class recall and the confusion matrix, and saves
   results/detector_<name>_confusion.npy, which the simulator can load:
       python -m emotion_rl.train --confusion results/detector_deepface-skip_confusion.npy

   FER2013 images are already 48x48 face crops, so backend "skip" (no face
   detection) measures the emotion model alone; "opencv" also shows how often
   the face detector fails on small faces.

2. Smoothing ablation on a recorded video or the webcam (no labels needed):
       python -m emotion_rl.detector_eval --video my_session.mp4
       python -m emotion_rl.detector_eval --camera 0 --seconds 30
   Counts label switches per minute ("flicker") and average label dwell time
   for no smoothing, EMA only, and EMA + hysteresis, on the same frames.
"""

import argparse
import json
import os
import time

import numpy as np

from emotion_rl.emotions import EMOTION_TO_INDEX, EMOTIONS, NUM_EMOTIONS


def classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    labels = list(range(NUM_EMOTIONS))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                      zero_division=0)),
        "per_class_recall": dict(zip(EMOTIONS, recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0).tolist())),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def list_fer_images(fer_dir, max_per_class):
    items = []
    for folder in sorted(os.listdir(fer_dir)):
        label = EMOTION_TO_INDEX.get(folder.lower())
        path = os.path.join(fer_dir, folder)
        if label is None or not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path))[:max_per_class]:
            items.append((os.path.join(path, name), label))
    if not items:
        raise SystemExit(f"No FER2013 class folders found in {fer_dir}")
    return items


def evaluate_deepface(items, backend):
    import cv2
    from deepface import DeepFace

    from emotion_rl.detector import _largest_face, deepface_probs

    y_true, y_pred, no_face, start = [], [], 0, time.time()
    for i, (path, label) in enumerate(items):
        img = cv2.imread(path)
        if img is None:
            continue
        results = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False,
                                   detector_backend=backend, silent=True)
        face = _largest_face(results)
        if backend != "skip" and float(face.get("face_confidence", 0.0)) <= 0:
            no_face += 1
        y_true.append(label)
        y_pred.append(int(np.argmax(deepface_probs(face))))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)} images")
    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    metrics["no_face_rate"] = no_face / max(len(y_true), 1)
    metrics["ms_per_image"] = (time.time() - start) * 1000 / max(len(y_true), 1)
    return metrics


def evaluate_custom(items, model_path):
    import cv2

    from emotion_rl.emotion_cnn import EmotionCNN

    model = EmotionCNN.load(model_path)
    images = np.stack([cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (48, 48))
                       for p, _ in items])
    start = time.time()
    probs = np.concatenate([model.predict_batch(images[i:i + 512])
                            for i in range(0, len(images), 512)])
    metrics = classification_metrics(np.array([lbl for _, lbl in items]), probs.argmax(1))
    metrics["ms_per_image"] = (time.time() - start) * 1000 / len(images)
    return metrics


def save_metrics(name, metrics, out_dir):
    from emotion_rl import plots

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"detector_{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(out_dir, f"detector_{name}_confusion.npy"),
            np.array(metrics["confusion"], dtype=np.float64))
    plots.confusion(metrics["confusion"], list(EMOTIONS),
                    os.path.join(out_dir, f"detector_{name}_confusion.png"),
                    f"{name} on FER2013 test (accuracy {metrics['accuracy']:.1%})")
    print(f"[{name}] accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
          f"({metrics['ms_per_image']:.0f} ms/image)")


# ─── Smoothing ablation ────────────────────────────────────────────────────────
SMOOTHING_CONFIGS = {
    "none": dict(smoothing=1.0, hysteresis=0.0),
    "ema": dict(smoothing=0.35, hysteresis=0.0),
    "ema+hysteresis": dict(smoothing=0.35, hysteresis=0.10),
}


def read_frames(video=None, camera=None, seconds=30, every=3):
    import cv2

    from emotion_rl.detector import open_camera

    cap = cv2.VideoCapture(video) if video else open_camera(camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video ' + video if video else f'camera {camera}'}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, start, i = [], time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok or (video is None and time.time() - start > seconds):
            break
        if i % every == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames, fps / every


def flicker_stats(labels, rate_hz):
    switches = sum(a != b for a, b in zip(labels, labels[1:]))
    minutes = len(labels) / rate_hz / 60
    return {"switches_per_minute": switches / max(minutes, 1e-9),
            "mean_dwell_seconds": len(labels) / rate_hz / (switches + 1)}


def smoothing_ablation(frames, rate_hz, backend):
    from deepface import DeepFace

    from emotion_rl.detector import EmotionDetector

    cached = [DeepFace.analyze(f, actions=["emotion"], enforce_detection=False,
                               detector_backend=backend, silent=True) for f in frames]
    results = {}
    for name, cfg in SMOOTHING_CONFIGS.items():
        replay = iter(cached)  # same DeepFace outputs for every smoothing setting
        detector = EmotionDetector(analyze_fn=lambda *_a, _r=replay, **_k: next(_r), **cfg)
        labels = [detector.detect(f).emotion for f in frames]
        results[name] = flicker_stats(labels, rate_hz)
        print(f"[Smoothing] {name:15s} {results[name]['switches_per_minute']:6.1f} switches/min, "
              f"dwell {results[name]['mean_dwell_seconds']:.1f}s")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the emotion detector")
    parser.add_argument("--fer", default="data/fer2013/test", help="FER2013 test folder")
    parser.add_argument("--backends", nargs="+", default=["skip", "opencv"])
    parser.add_argument("--custom-model", default=None, help="path to emotion_cnn.pt")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    if args.video or args.camera is not None:
        frames, rate = read_frames(args.video, args.camera, args.seconds)
        results = smoothing_ablation(frames, rate, args.backends[-1])
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "smoothing_ablation.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    if not os.path.isdir(args.fer):
        raise SystemExit(f"FER2013 test set not found at {args.fer}. Download it from "
                         "https://www.kaggle.com/datasets/msambare/fer2013 "
                         "and unzip it to data/fer2013/")
    items = list_fer_images(args.fer, args.max_per_class)
    print(f"[Detector eval] {len(items)} FER2013 test images")
    for backend in args.backends:
        save_metrics(f"deepface-{backend}", evaluate_deepface(items, backend), args.out)
    if args.custom_model:
        save_metrics("emotion-cnn", evaluate_custom(items, args.custom_model), args.out)


if __name__ == "__main__":
    main()
