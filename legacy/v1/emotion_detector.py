import cv2
from deepface import DeepFace
import numpy as np
from collections import deque

# ─── FER2013 Correct 7-Class Mapping ──────────────────────────────────────────
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOTIONS_TO_INDEX = {
    'angry':    0,
    'disgust':  1,
    'fear':     2,
    'happy':    3,
    'sad':      4,
    'surprise': 5,
    'neutral':  6,
}

INDEX_TO_EMOTION = {v: k for k, v in EMOTIONS_TO_INDEX.items()}

NUM_EMOTIONS = 7

# BGR colors for each emotion
EMOTION_COLORS = {
    'angry':    (0, 0, 255),
    'disgust':  (0, 128, 128),
    'fear':     (128, 0, 128),
    'happy':    (0, 255, 0),
    'sad':      (255, 0, 0),
    'surprise': (0, 165, 255),
    'neutral':  (255, 255, 0),
}


class EmotionDetector:
    def __init__(self, smoothing_window=5, confidence_threshold=30.0):
        """
        Args:
            smoothing_window     : number of frames to average box position over
            confidence_threshold : minimum confidence % to update emotion label
        """
        # OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # ── Smoothing Buffers ──────────────────────────────────────────────────
        self.box_buffer     = deque(maxlen=smoothing_window)  # stores last N box positions
        self.emotion_buffer = deque(maxlen=smoothing_window)  # stores last N emotions

        # ── Stable State (only updates when confident) ─────────────────────────
        self.stable_emotion     = 'neutral'
        self.stable_emotion_idx = 6
        self.stable_box         = None  # (x, y, w, h)

        # ── Current Raw State (updates every detection) ────────────────────────
        self.current_emotion     = 'neutral'
        self.current_emotion_idx = 6
        self.confidence          = 0.0

        self.confidence_threshold = confidence_threshold

        print("[EmotionDetector] Initialized with stable bounding box!")
        print(f"[EmotionDetector] Smoothing window : {smoothing_window} frames")
        print(f"[EmotionDetector] Min confidence   : {confidence_threshold}%")

    def detect_from_frame(self, frame):
        """
        Detects emotion from webcam frame.
        Uses smoothing to keep bounding box stable.
        Returns: (emotion_str, emotion_idx, confidence, annotated_frame)
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = DeepFace.analyze(
                rgb_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            emotion    = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]
            region     = result[0].get('region', {})

            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)

            # ── Step 1: Add box to buffer if face is valid ─────────────────────
            if w > 0 and h > 0:
                self.box_buffer.append((x, y, w, h))

            # ── Step 2: Average box over last N frames ─────────────────────────
            if len(self.box_buffer) > 0:
                avg_x = int(np.mean([b[0] for b in self.box_buffer]))
                avg_y = int(np.mean([b[1] for b in self.box_buffer]))
                avg_w = int(np.mean([b[2] for b in self.box_buffer]))
                avg_h = int(np.mean([b[3] for b in self.box_buffer]))
                self.stable_box = (avg_x, avg_y, avg_w, avg_h)

            # ── Step 3: Add emotion to buffer only if confident enough ─────────
            if confidence >= self.confidence_threshold:
                self.emotion_buffer.append(emotion)

            # ── Step 4: Majority vote — pick most frequent emotion in buffer ───
            if len(self.emotion_buffer) > 0:
                stable_emotion          = max(set(self.emotion_buffer),
                                              key=list(self.emotion_buffer).count)
                self.stable_emotion     = stable_emotion
                self.stable_emotion_idx = EMOTIONS_TO_INDEX.get(stable_emotion, 6)

            # ── Step 5: Update raw current state ──────────────────────────────
            self.current_emotion     = emotion
            self.current_emotion_idx = EMOTIONS_TO_INDEX.get(emotion, 6)
            self.confidence          = confidence

        except Exception:
            pass  # keep last stable state on any error

        # Draw stable box on frame
        frame = self._draw_stable_box(frame)

        return (
            self.stable_emotion,
            self.stable_emotion_idx,
            self.confidence,
            frame
        )

    def _draw_stable_box(self, frame):
        """
        Draws smoothed/averaged bounding box and emotion label.
        Called every frame — even between detections — so box never disappears.
        """
        if self.stable_box is None:
            return frame

        x, y, w, h = self.stable_box
        emotion     = self.stable_emotion
        confidence  = self.confidence
        color       = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label with colored background for readability
        label      = f"{emotion.upper()}  {confidence:.1f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame,
                      (x, y - label_size[1] - 15),
                      (x + label_size[0] + 5, y),
                      color, -1)  # filled rectangle as background

        # White text on colored background
        cv2.putText(frame, label,
                    (x + 2, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # FER2013 index shown below box
        index_label = f"FER idx: {self.stable_emotion_idx}"
        cv2.putText(frame, index_label,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1)

        return frame

    def get_emotion_vector(self):
        """
        Returns one-hot vector of size 7 for RL state space.
        Uses STABLE emotion (not raw current).
        Example: happy (idx=3) → [0, 0, 0, 1, 0, 0, 0]
        """
        vector = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        vector[self.stable_emotion_idx] = 1.0
        return vector

    def get_current_state(self):
        """Returns full stable state as dictionary."""
        return {
            'emotion':    self.stable_emotion,
            'index':      self.stable_emotion_idx,
            'confidence': self.confidence,
            'vector':     self.get_emotion_vector()
        }

    def detect_from_image(self, image_path):
        """Detect emotion from a saved image file (useful for testing)."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[EmotionDetector] Could not load image: {image_path}")
            return 'neutral', 6, 0.0, None
        return self.detect_from_frame(frame)


# ─── Main: Webcam Test ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Emotion Detector - Stable Bounding Box Version")
    print("=" * 55)
    print("Press 'q' to quit\n")

    detector = EmotionDetector(
        smoothing_window=5,        # average box over 5 frames
        confidence_threshold=30.0  # only update emotion if >30% confident
    )

    cap = cv2.VideoCapture(1)  # change to 0 if webcam not found

    if not cap.isOpened():
        print("[ERROR] Could not open webcam! Try changing VideoCapture(1) to VideoCapture(0)")
        exit()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame!")
            break

        # Run DeepFace detection every 5 frames
        if frame_count % 5 == 0:
            emotion, idx, confidence, frame = detector.detect_from_frame(frame)
            state = detector.get_current_state()
            print(f"Emotion: {emotion:10s} | "
                  f"FER Index: {idx} | "
                  f"Confidence: {confidence:.1f}% | "
                  f"Vector: {state['vector']}")
        else:
            # On non-detection frames: still draw last stable box so it never disappears
            frame = detector._draw_stable_box(frame)

        cv2.imshow('Emotion Detector - Stable | Press Q to quit', frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Done] Emotion detection stopped.")
