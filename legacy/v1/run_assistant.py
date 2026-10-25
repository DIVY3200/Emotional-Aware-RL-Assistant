import cv2
import os
import time
import threading
import queue
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from google import genai
from stable_baselines3 import PPO
from emotion_detector import EmotionDetector, INDEX_TO_EMOTION
from rl_environment import EmotionRLEnvironment, ACTIONS

# ─── Load API Key ──────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("[Error] GEMINI_API_KEY not found! Check your .env file")

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH        = "model/ppo_agent/best_model"
WEBCAM_INDEX      = 1
DETECT_EVERY      = 5
RESPONSE_COOLDOWN = 8

# ─── Emotion colors for history bar ───────────────────────────────────────────
EMOTION_COLORS = {
    'angry':    (0, 0, 255),
    'disgust':  (0, 128, 128),
    'fear':     (128, 0, 128),
    'happy':    (0, 255, 0),
    'sad':      (255, 0, 0),
    'surprise': (0, 165, 255),
    'neutral':  (255, 255, 0),
}

# ─── Emotion-aware fallback responses (used when Gemini fails) ─────────────────
EMOTION_FALLBACKS = {
    'angry': {
        'comfort':       "I can feel your frustration. Take a deep breath — I am right here with you.",
        'motivate':      "Channel that energy into something powerful. You have more strength than you know!",
        'neutral':       "I notice you seem upset. I am here whenever you want to talk.",
        'suggest_break': "It might help to step away for a moment. A short break can calm the storm.",
        'playful':       "Let us shake off that anger — want to try something fun together?",
    },
    'disgust': {
        'comfort':       "Something seems to be bothering you deeply. I am here to listen.",
        'motivate':      "Do not let this bring you down. You are bigger than whatever is troubling you.",
        'neutral':       "I am here for you. Take your time.",
        'suggest_break': "Sometimes stepping back gives us a fresh perspective. Take a breather.",
        'playful':       "Let us find something more enjoyable to focus on!",
    },
    'fear': {
        'comfort':       "You are safe here. Whatever you are facing, you do not have to face it alone.",
        'motivate':      "Courage is not the absence of fear — it is taking the next step anyway. You have got this!",
        'neutral':       "I am right here with you. Everything will be okay.",
        'suggest_break': "Take a slow, deep breath. Give yourself a moment to reset.",
        'playful':       "Hey, let us smile through this together — you are braver than you think!",
    },
    'happy': {
        'comfort':       "You are glowing today! That positivity is truly wonderful.",
        'motivate':      "You are on fire! Keep riding this amazing wave of energy!",
        'neutral':       "It is great to see you in such good spirits!",
        'suggest_break': "You deserve to enjoy this moment — take a happy little pause!",
        'playful':       "Love that energy! Let us make today even more fun!",
    },
    'sad': {
        'comfort':       "I am sorry you are feeling this way. It is okay to feel sad — I am here with you.",
        'motivate':      "Even on your darkest days, you are still moving forward. That takes real strength.",
        'neutral':       "I am here. You do not have to go through this alone.",
        'suggest_break': "Be gentle with yourself. Take a small break and breathe.",
        'playful':       "Let us find one tiny thing to smile about right now — you deserve it.",
    },
    'surprise': {
        'comfort':       "That caught you off guard! Take a moment to process — I am here.",
        'motivate':      "Surprises can be the start of something amazing. Embrace it!",
        'neutral':       "Wow, something unexpected happened! How are you feeling about it?",
        'suggest_break': "Take a breath and let the surprise settle before you react.",
        'playful':       "Life is full of surprises — and so are you! Let us enjoy this moment!",
    },
    'neutral': {
        'comfort':       "I am here for you. How can I make your day better?",
        'motivate':      "Every moment is a fresh start. What would you like to accomplish today?",
        'neutral':       "I am here for you. How can I help you today?",
        'suggest_break': "A small pause can do wonders. Feel free to take a moment.",
        'playful':       "Let us liven things up a little! What sounds fun to you?",
    },
}


# ─── Speech Engine (Queue-based) ───────────────────────────────────────────────
class SpeechEngine:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking  = False
        self._stop_flag   = False

        self._thread = threading.Thread(target=self._run_engine, daemon=True)
        self._thread.start()
        print("[Speech] Initialized — queue-based engine ready")

    def _run_engine(self):
        """Dedicated thread that owns the pyttsx3 engine."""
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)

        voices = engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break

        while not self._stop_flag:
            try:
                text = self.speech_queue.get(timeout=0.5)
                self.is_speaking = True
                print(f"[Speech] Speaking: {text[:60]}...")
                engine.say(text)
                engine.runAndWait()
                self.is_speaking = False
                self.speech_queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text):
        """Add text to speech queue — non-blocking."""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        self.speech_queue.put(text)

    def stop(self):
        self._stop_flag = True


# ─── Gemini Response Generator ─────────────────────────────────────────────────
class GeminiAssistant:

    # ── Try models in order until one works ───────────────────────────────────
    MODELS_TO_TRY = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
    ]

    def __init__(self, api_key):
        self.client        = genai.Client(api_key=api_key)
        self.working_model = None
        self._find_working_model()

    def _find_working_model(self):
        """Auto-detect the first available Gemini model."""
        test_prompt = "Say hi in one word."
        for model_name in self.MODELS_TO_TRY:
            try:
                self.client.models.generate_content(
                    model    = model_name,
                    contents = test_prompt
                )
                self.working_model = model_name
                print(f"[Gemini] Initialized with model: {model_name}")
                return
            except Exception as e:
                print(f"[Gemini] Model {model_name} unavailable: {e}")

        print("[Gemini] WARNING — No working model found. Using fallback responses only.")
        self.working_model = None

    def generate_response(self, emotion, action):
        """Generate emotion + action aware response. Falls back gracefully."""

        # ── Try Gemini if a working model was found ────────────────────────────
        if self.working_model:
            prompt = f"""You are an empathetic AI assistant helping someone who is feeling {emotion}.
Your task is to {action} them with a warm, genuine response.

Rules:
- Maximum 2 short sentences
- Be natural, warm and human
- Do NOT use emojis
- Do NOT say I am an AI
- Match the tone to the emotion: {emotion}
- Action type: {action}

Generate only the response text, nothing else."""
            try:
                response = self.client.models.generate_content(
                    model    = self.working_model,
                    contents = prompt
                )
                text = response.text.strip()
                if text:
                    return text
            except Exception as e:
                print(f"[Gemini ERROR] {e}")

        # ── Emotion + Action aware fallback ───────────────────────────────────
        emotion_responses = EMOTION_FALLBACKS.get(emotion, EMOTION_FALLBACKS['neutral'])
        fallback_text     = emotion_responses.get(action, "I am here to support you.")
        print(f"[Fallback] Using local response for emotion={emotion}, action={action}")
        return fallback_text


# ─── Main Assistant ────────────────────────────────────────────────────────────
class EmotionAssistant:
    def __init__(self):
        print("\n" + "=" * 60)
        print("  Emotion-Aware RL Assistant — Starting Up")
        print("=" * 60)

        self.detector  = EmotionDetector(smoothing_window=5, confidence_threshold=30.0)
        self.speech    = SpeechEngine()
        self.gemini    = GeminiAssistant(GEMINI_API_KEY)

        if not os.path.exists(MODEL_PATH + ".zip"):
            raise FileNotFoundError(
                f"[ERROR] PPO model not found at {MODEL_PATH}.zip\n"
                f"Please run: python train_ppo.py --mode train"
            )
        self.ppo_model = PPO.load(MODEL_PATH)
        print(f"[PPO] Model loaded from {MODEL_PATH}.zip")

        self.last_response_time = time.time()
        self.last_emotion       = 'neutral'
        self.current_response   = ""
        self.frame_count        = 0
        self.ppo_step_count     = 0
        self.emotion_history    = []
        self.current_action     = "neutral"

        print("\n[Assistant] All systems ready!")
        print("[Assistant] Press Q to quit\n")

    def get_ppo_action(self, emotion_idx):
        state              = np.zeros(7, dtype=np.float32)
        state[emotion_idx] = 1.0
        action, _          = self.ppo_model.predict(state, deterministic=True)
        self.ppo_step_count += 1
        return int(action)

    def should_respond(self, emotion):
        now             = time.time()
        time_passed     = now - self.last_response_time
        emotion_changed = emotion != self.last_emotion
        return time_passed >= RESPONSE_COOLDOWN or emotion_changed

    def draw_ui(self, frame, emotion, action_name, response):
        h, w = frame.shape[:2]

        # ── Bottom panel ───────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame,
                    f"PPO Action : {action_name.upper()}",
                    (10, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if response:
            words = response.split()
            line1 = ' '.join(words[:9])
            line2 = ' '.join(words[9:18]) if len(words) > 9 else ''
            cv2.putText(frame, line1, (10, h - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            if line2:
                cv2.putText(frame, line2, (10, h - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        # ── Top bar ────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
        cv2.putText(frame,
                    "Emotion-Aware RL Assistant  |  Press Q to quit",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── PPO Step Counter ───────────────────────────────────────────────────
        cv2.putText(frame,
                    f"PPO Steps: {self.ppo_step_count}",
                    (w - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

        # ── Speaking indicator ─────────────────────────────────────────────────
        if self.speech.is_speaking:
            cv2.circle(frame, (w - 20, 55), 8, (0, 0, 255), -1)
            cv2.putText(frame, "Speaking",
                        (w - 100, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # ── Emotion History Bar ────────────────────────────────────────────────
        self._draw_emotion_history(frame, w, h)

        return frame

    def _draw_emotion_history(self, frame, w, h):
        """Shows last 6 detected emotions as colored boxes on right side."""
        if not self.emotion_history:
            return

        box_h   = 28
        box_w   = 110
        start_y = 40
        start_x = w - box_w - 10

        cv2.putText(frame, "Emotion Log",
                    (start_x, start_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        for i, em in enumerate(self.emotion_history[-6:]):
            y     = start_y + i * (box_h + 4)
            color = EMOTION_COLORS.get(em, (200, 200, 200))

            thickness  = -1 if i == len(self.emotion_history[-6:]) - 1 else 1
            cv2.rectangle(frame,
                          (start_x, y),
                          (start_x + box_w, y + box_h),
                          color, thickness)

            text_color = (0, 0, 0) if thickness == -1 else color
            cv2.putText(frame, em.upper(),
                        (start_x + 5, y + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    def run(self):
        cap = cv2.VideoCapture(WEBCAM_INDEX)

        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam {WEBCAM_INDEX}!")
            print("Try changing WEBCAM_INDEX to 0")
            return

        action_name = "neutral"

        # Greeting
        self.speech.speak(
            "Hello! I am your emotion aware assistant. "
            "I will support you based on how you feel."
        )
        self.last_response_time = time.time()
        print("[Running] Webcam started. Detecting emotions...\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read webcam frame!")
                break

            frame = cv2.flip(frame, 1)

            if self.frame_count % DETECT_EVERY == 0:

                # Detect emotion
                emotion, emotion_idx, confidence, frame = \
                    self.detector.detect_from_frame(frame)

                # PPO decides action
                action      = self.get_ppo_action(emotion_idx)
                action_name = ACTIONS.get(action, 'neutral')
                self.current_action = action_name

                # Update emotion history log
                if not self.emotion_history or self.emotion_history[-1] != emotion:
                    self.emotion_history.append(emotion)
                    if len(self.emotion_history) > 10:
                        self.emotion_history.pop(0)

                # Generate response + speak
                if self.should_respond(emotion) and not self.speech.is_speaking:
                    self.last_emotion       = emotion
                    self.last_response_time = time.time()

                    def generate_and_speak(em=emotion, ac=action_name):
                        text = self.gemini.generate_response(em, ac)
                        self.current_response = text
                        self.speech.speak(text)

                    thread = threading.Thread(target=generate_and_speak, daemon=True)
                    thread.start()

                    print(f"[PPO Step {self.ppo_step_count:4d}] "
                          f"Emotion: {emotion:10s} | "
                          f"Action: {action_name:15s} | "
                          f"Generating response...")

            else:
                frame = self.detector._draw_stable_box(frame)

            frame = self.draw_ui(frame, self.last_emotion,
                                 action_name, self.current_response)

            cv2.imshow("Emotion-Aware RL Assistant", frame)
            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[Quit] Stopping assistant...")
                break

        self.speech.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[Done] Emotion Assistant stopped.")


if __name__ == "__main__":
    assistant = EmotionAssistant()
    assistant.run()