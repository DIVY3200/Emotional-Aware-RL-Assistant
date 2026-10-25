"""Facial emotion detection with temporal smoothing and explicit "no face" handling.

Fixes over v1:
* DeepFace expects BGR images; v1 converted frames to RGB first.
* With enforce_detection=False DeepFace returns the whole frame and an emotion
  even when nobody is there (face_confidence = 0). We now report face_found=False
  so the assistant can stay quiet instead of reacting to a meaningless label.
* Smoothing works on the full probability vector (exponential moving average)
  with hysteresis, and the confidence shown belongs to the label shown.
* Detection can run in a background thread (DetectionWorker) so the webcam
  preview never freezes while the model runs.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from emotion_rl.emotion_models import apply_calibration
from emotion_rl.emotions import EMOTION_COLORS, EMOTION_TO_INDEX, EMOTIONS, NEUTRAL, NUM_EMOTIONS


@dataclass
class Detection:
    probs: np.ndarray  # smoothed probabilities over EMOTIONS (sums to 1)
    emotion: str  # stable label (with hysteresis)
    confidence: float  # smoothed probability of `emotion`, in %
    raw_probs: np.ndarray = field(default_factory=lambda: _neutral_probs())  # before calibration
    extra_probs: dict = field(default_factory=dict)  # other models on the same face (comparison)
    face_found: bool = False
    box: tuple | None = None  # smoothed (x, y, w, h)
    latency_ms: float = 0.0
    timestamp: float = 0.0

    @property
    def index(self):
        return EMOTION_TO_INDEX[self.emotion]


def _neutral_probs():
    probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
    probs[NEUTRAL] = 1.0
    return probs


def deepface_probs(result):
    """DeepFace returns percentages keyed by emotion name; convert to our index order."""
    scores = result["emotion"]
    probs = np.array([float(scores.get(name, 0.0)) for name in EMOTIONS], dtype=np.float32)
    total = probs.sum()
    return probs / total if total > 0 else _neutral_probs()


def _largest_face(results):
    return max(results, key=lambda r: r["region"]["w"] * r["region"]["h"])


class EmotionDetector:
    def __init__(
        self,
        detector_backend="yunet",
        smoothing=0.35,
        hysteresis=0.10,
        min_face_confidence=0.0,
        box_window=5,
        classifier=None,
        calibration=None,
        extra_classifiers=None,
        analyze_fn=None,
    ):
        """
        Args:
            detector_backend    : DeepFace face detector (opencv, retinaface, mtcnn, ssd, ...)
            smoothing           : EMA weight of the newest frame (1.0 = no smoothing)
            hysteresis          : the new top emotion must beat the current label by this
                                  much smoothed probability before the label switches
            min_face_confidence : faces at or below this detector confidence are ignored
            box_window          : frames averaged for the drawn bounding box
            classifier          : optional object with predict_proba(face_bgr) -> probs;
                                  replaces DeepFace's emotion model (see emotion_models.py)
            calibration         : optional per-class bias for this user (emotion_models.py)
            extra_classifiers   : {name: classifier} also run on the same face, for comparison
            analyze_fn          : injectable stand-in for DeepFace.analyze (tests)
        """
        self.detector_backend = detector_backend
        self.smoothing = smoothing
        self.hysteresis = hysteresis
        self.min_face_confidence = min_face_confidence
        self.classifier = classifier
        self.calibration = None if calibration is None else np.asarray(calibration)
        self.extra_classifiers = extra_classifiers or {}
        self._analyze = analyze_fn
        self.box_buffer = deque(maxlen=box_window)
        self.reset()

    def reset(self):
        self.smoothed = _neutral_probs()
        self.stable_emotion = "neutral"
        self.box_buffer.clear()
        self.frames_without_face = 0
        self.last = Detection(self.smoothed.copy(), "neutral", 100.0)

    # ── model calls ─────────────────────────────────────────────────────────
    def _load_backend(self):
        """Builds the face detector once; falls back to OpenCV if its weights cannot be fetched."""
        from deepface import DeepFace

        try:
            DeepFace.build_model(self.detector_backend, task="face_detector")
        except Exception as exc:
            print(f"[EmotionDetector] '{self.detector_backend}' face detector unavailable "
                  f"({str(exc)[:160]}).")
            if self.detector_backend == "yunet":
                print("  Download it manually:\n  curl -L -o ~/.deepface/weights/"
                      "face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/"
                      "raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
            print("  Falling back to 'opencv' (fast, but prone to false faces).")
            self.detector_backend = "opencv"
        self._analyze = DeepFace.analyze

    def _run_model(self, frame_bgr):
        """Returns (raw_probs, face_found, region) for the largest face in the frame."""
        if self._analyze is None:
            self._load_backend()
        if self.classifier is not None:
            return self._run_custom_classifier(frame_bgr)
        results = self._analyze(frame_bgr, actions=["emotion"], enforce_detection=False,
                                detector_backend=self.detector_backend, silent=True)
        face = _largest_face(results)
        found = float(face.get("face_confidence", 0.0)) > self.min_face_confidence
        return deepface_probs(face), found, face["region"]

    def _run_custom_classifier(self, frame_bgr):
        from deepface import DeepFace

        faces = DeepFace.extract_faces(frame_bgr, detector_backend=self.detector_backend,
                                       enforce_detection=False, align=True)
        face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
        found = float(face.get("confidence", 0.0)) > self.min_face_confidence
        region = face["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        crop = frame_bgr[y:y + h, x:x + w] if found else frame_bgr
        return self.classifier.predict_proba(crop), found, region

    # ── public API ──────────────────────────────────────────────────────────
    def detect(self, frame_bgr):
        start = time.perf_counter()
        try:
            raw, found, region = self._run_model(frame_bgr)
        except Exception as exc:  # keep the last state, but never fail silently
            print(f"[EmotionDetector] detection failed: {exc}")
            return self.last

        extra = {}
        if found:
            self.frames_without_face = 0
            self.box_buffer.append((region["x"], region["y"], region["w"], region["h"]))
            if self.extra_classifiers:
                x, y, w, h = (max(int(region[k]), 0) for k in "xywh")
                crop = frame_bgr[y:y + h, x:x + w]
                extra = {name: clf.predict_proba(crop)
                         for name, clf in self.extra_classifiers.items()}
            calibrated = apply_calibration(raw, self.calibration)
            self.smoothed = (1 - self.smoothing) * self.smoothed + self.smoothing * calibrated
            self.smoothed /= self.smoothed.sum()
            self._update_stable_label()
        else:
            self.frames_without_face += 1
            if self.frames_without_face >= self.box_buffer.maxlen:
                self.box_buffer.clear()

        box = tuple(int(v) for v in np.mean(self.box_buffer, axis=0)) if self.box_buffer else None
        self.last = Detection(
            probs=self.smoothed.copy(),
            emotion=self.stable_emotion,
            confidence=float(self.smoothed[EMOTION_TO_INDEX[self.stable_emotion]] * 100),
            raw_probs=raw,
            extra_probs=extra,
            face_found=found,
            box=box,
            latency_ms=(time.perf_counter() - start) * 1000,
            timestamp=time.time(),
        )
        return self.last

    def _update_stable_label(self):
        top = int(np.argmax(self.smoothed))
        current = EMOTION_TO_INDEX[self.stable_emotion]
        if self.smoothed[top] - self.smoothed[current] > self.hysteresis or self.hysteresis <= 0:
            self.stable_emotion = EMOTIONS[top]

    def warm_up(self):
        """Load the face and emotion models now (~20 s the first time) instead of on the
        first real frame."""
        if self._analyze is None:
            self._load_backend()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self._run_model(blank)
        for clf in self.extra_classifiers.values():
            clf.predict_proba(blank[:96, :96])
        self.reset()

    def detect_from_image(self, image_path):
        import cv2

        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        self.reset()
        self.smoothing, saved = 1.0, self.smoothing  # a single image needs no smoothing
        try:
            return self.detect(frame)
        finally:
            self.smoothing = saved


CAMERA_BACKENDS = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "v4l2": "CAP_V4L2", "any": "CAP_ANY"}


def open_camera(index=0, backend="auto", width=1280, height=720):
    """Opens a webcam. On Windows "auto" uses DirectShow: with OpenCV's default (Media
    Foundation) a Logitech Brio 100 took over two minutes to open; DirectShow takes ~3 s.
    Note the two backends can number the same cameras differently."""
    import sys

    import cv2

    if backend == "auto":
        backend = "dshow" if sys.platform == "win32" else "any"
    cap = cv2.VideoCapture(index, getattr(cv2, CAMERA_BACKENDS[backend]))
    if cap.isOpened() and width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_detection(frame, detection):
    """Draws the smoothed face box and label. Safe to call on every frame."""
    import cv2

    if detection is None or detection.box is None:
        return frame
    x, y, w, h = detection.box
    color = EMOTION_COLORS.get(detection.emotion, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{detection.emotion.upper()}  {detection.confidence:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), color, -1)
    cv2.putText(frame, label, (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


class DetectionWorker:
    """Runs the detector on the newest frame in a background thread.

    The UI thread calls submit(frame) every frame and reads `latest`; frames that
    arrive while the model is busy simply replace each other (no queue build-up).
    """

    def __init__(self, detector, min_interval=0.0):
        self.detector = detector
        self.min_interval = min_interval
        self.latest = None
        self._frame = None
        self._lock = threading.Lock()
        self._has_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def submit(self, frame):
        with self._lock:
            self._frame = frame.copy()
        self._has_frame.set()

    def _loop(self):
        while not self._stop.is_set():
            if not self._has_frame.wait(timeout=0.2):
                continue
            with self._lock:
                frame, self._frame = self._frame, None
                self._has_frame.clear()
            if frame is None:
                continue
            started = time.time()
            self.latest = self.detector.detect(frame)
            wait = self.min_interval - (time.time() - started)
            if wait > 0:
                self._stop.wait(wait)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ─── Main: webcam test ─────────────────────────────────────────────────────────
def main(argv=None):
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="Webcam emotion detector test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", default="yunet")
    args = parser.parse_args(argv)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open camera {args.camera}; try --camera 1")
    worker = DetectionWorker(EmotionDetector(detector_backend=args.backend)).start()
    print("Press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        worker.submit(frame)
        det = worker.latest
        frame = draw_detection(frame, det)
        if det is not None and not det.face_found:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        cv2.imshow("Emotion detector  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    worker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
