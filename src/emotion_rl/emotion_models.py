"""Interchangeable facial-emotion classifiers and per-user calibration.

Every classifier takes a BGR face crop and returns probabilities in our
EMOTIONS order (angry, disgust, fear, happy, sad, surprise, neutral).

* "deepface"  : DeepFace's FER2013 model (the v1 default). Weak on fear and sad.
* "hsemotion" : HSEmotion EfficientNet-B2 trained on AffectNet (Savchenko, 2022),
                a much larger in-the-wild dataset; ~30 ms per face on CPU.
* path/to/emotion_cnn.pt : your own FER2013 CNN (emotion_cnn.py).

Calibration: posed expressions differ between people, and a model can
systematically under-predict a class for one person (e.g. sad read as neutral).
fit_calibration learns one bias per class from a short camera check,

    p'(e)  ∝  p(e) · exp(b_e)

(a prior-shift correction with L2 regularisation), which the detector applies
before smoothing. 7 numbers per user: cheap, robust and easy to inspect.
"""

import json
import os

import numpy as np

from emotion_rl.emotions import EMOTIONS, NUM_EMOTIONS

_HSE_TO_OURS = {"Anger": "angry", "Disgust": "disgust", "Fear": "fear", "Happiness": "happy",
                "Sadness": "sad", "Surprise": "surprise", "Neutral": "neutral"}


class HSEmotionClassifier:
    name = "hsemotion"

    def __init__(self, model_name="enet_b2_7"):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self.model = HSEmotionRecognizer(model_name=model_name)
        classes = [self.model.idx_to_class[i] for i in range(len(self.model.idx_to_class))]
        # "Contempt" (8-class models) has no FER counterpart and is dropped
        self.index = [classes.index(next(k for k, v in _HSE_TO_OURS.items() if v == e))
                      for e in EMOTIONS]

    def predict_proba(self, face_bgr):
        import cv2

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        _, scores = self.model.predict_emotions(rgb, logits=False)
        probs = np.asarray(scores, dtype=np.float64)[self.index]
        return (probs / probs.sum()).astype(np.float32)


class DeepFaceCropClassifier:
    """DeepFace's emotion model on an already cropped face (no second face detection)."""

    name = "deepface"

    def predict_proba(self, face_bgr):
        from deepface import DeepFace

        from emotion_rl.detector import deepface_probs

        result = DeepFace.analyze(face_bgr, actions=["emotion"], enforce_detection=False,
                                  detector_backend="skip", silent=True)
        return deepface_probs(result[0])


def make_classifier(name):
    """None means "use DeepFace.analyze directly" (detection + emotion in one call)."""
    if name in (None, "", "deepface"):
        return None
    if name == "hsemotion":
        return HSEmotionClassifier()
    if str(name).endswith(".pt"):
        from emotion_rl.emotion_cnn import EmotionCNN

        return EmotionCNN.load(name)
    raise ValueError(f"Unknown emotion model '{name}' (deepface, hsemotion or a .pt file)")


# ─── Per-user calibration ──────────────────────────────────────────────────────
def apply_calibration(probs, bias):
    if bias is None:
        return probs
    logits = np.log(np.asarray(probs, dtype=np.float64) + 1e-6) + np.asarray(bias)
    z = np.exp(logits - logits.max())
    return (z / z.sum()).astype(np.float32)


def fit_calibration(probs, targets, l2=0.01, steps=800, lr=0.5):
    """Bias b minimising cross-entropy of softmax(log p + b) on (probs, target index) pairs."""
    log_p = np.log(np.asarray(probs, dtype=np.float64) + 1e-6)
    y = np.eye(NUM_EMOTIONS)[np.asarray(targets)]
    b = np.zeros(NUM_EMOTIONS)
    for _ in range(steps):
        logits = log_p + b
        q = np.exp(logits - logits.max(axis=1, keepdims=True))
        q /= q.sum(axis=1, keepdims=True)
        grad = (q - y).mean(axis=0) + 2 * l2 * b
        b -= lr * grad
    return b - b.mean()  # only differences between classes matter


def accuracy(probs, targets, bias=None):
    if len(targets) == 0:
        return float("nan")
    preds = [int(np.argmax(apply_calibration(p, bias))) for p in probs]
    return float(np.mean(np.asarray(preds) == np.asarray(targets)))


def profile_path(user, profile_dir="data/profiles"):
    return os.path.join(profile_dir, f"{user}_detector.json")


def save_detector_profile(user, emotion_model, bias, report, profile_dir="data/profiles"):
    os.makedirs(profile_dir, exist_ok=True)
    path = profile_path(user, profile_dir)
    with open(path, "w") as f:
        json.dump({"emotion_model": emotion_model, "bias": np.round(bias, 4).tolist(),
                   "bias_by_emotion": dict(zip(EMOTIONS, np.round(bias, 3).tolist())),
                   "report": report}, f, indent=2)
    return path


def load_detector_profile(user, profile_dir="data/profiles"):
    path = profile_path(user, profile_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
