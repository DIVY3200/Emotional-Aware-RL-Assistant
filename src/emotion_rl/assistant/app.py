"""Live emotion-aware assistant: webcam → emotion → RL policy → LLM reply → speech.

    python -m emotion_rl.assistant.app                        # defaults from configs/default.yaml
    python -m emotion_rl.assistant.app --user divyansh        # personalise to one user
    python -m emotion_rl.assistant.app --condition B          # blinded user-study condition
    python -m emotion_rl.assistant.app --no-llm --mute        # offline demo

Every `decision_interval` seconds is one turn, exactly like a simulator step:
    1. close the previous turn: reward = change in the user's (detected) valence
       + any explicit feedback key, log the transition, update the user's profile
    2. build the observation (same ObservationBuilder as training)
    3. the policy picks an action; "listen" means stay quiet
    4. speaking actions are turned into words by the LLM in a background thread

Keys: G = that helped, B = that did not help, T = talk (voice input), P = pause, Q = quit
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from emotion_rl.assistant.latency import LatencyTracker
from emotion_rl.assistant.llm import ResponseGenerator
from emotion_rl.assistant.safety import (
    DEFAULT_CHECK_IN_MESSAGE,
    DEFAULT_HELPLINE_MESSAGE,
    WellbeingMonitor,
    detect_crisis,
)
from emotion_rl.assistant.session_log import InteractionLogger, live_reward
from emotion_rl.assistant.speech import SilentSpeech, SpeechEngine, SpeechRecognizer
from emotion_rl.bandits import OnlinePersonalizer, PersonalizedPolicy
from emotion_rl.config import load_config
from emotion_rl.detector import DetectionWorker, EmotionDetector, draw_detection, open_camera
from emotion_rl.emotion_models import load_detector_profile, make_classifier
from emotion_rl.emotions import ACTIONS, EMOTION_COLORS, EMOTIONS, LISTEN, V1_RULE_TABLE
from emotion_rl.env import ObservationBuilder
from emotion_rl.policies import (
    RandomPolicy,
    RuleCooldownPolicy,
    RuleTablePolicy,
    SB3Policy,
    current_emotion_probs,
    read_model_config,
)

LIVE_TIME_FRAC = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "STEP 2 of 2 - live assistant (click the buttons or press G/B/T/P/Q)"
BUTTONS = [("Helped (G)", "g", (0, 140, 0)), ("Didn't help (B)", "b", (0, 0, 170)),
           ("Talk (T)", "t", (150, 90, 0)), ("Pause (P)", "p", (90, 90, 90)),
           ("Quit (Q)", "q", (60, 60, 60))]


# ─── Policy construction ───────────────────────────────────────────────────────
def make_policy(name, model_path, user, profile_dir):
    """Returns (policy, observation builder, profile path or None)."""
    env_cfg = {"history_len": 4, "observation_mode": "probs", "max_steps": 20}
    profile_path = None
    if name in ("ppo", "ppo_personalised"):
        if not os.path.exists(model_path):
            raise SystemExit(f"[ERROR] Trained model not found at {model_path}. "
                             "Train first (README > Reproduce) or pass --model.")
        model_cfg = read_model_config(model_path)
        env_cfg.update({k: v for k, v in model_cfg.get("env", {}).items() if k in env_cfg})
        policy = SB3Policy.from_path(model_path, algo=model_cfg.get("algo", "ppo"), name="PPO")
        if name == "ppo_personalised":
            os.makedirs(profile_dir, exist_ok=True)
            profile_path = os.path.join(profile_dir, f"{user}.json")
            personalizer = (OnlinePersonalizer.load(profile_path) if os.path.exists(profile_path)
                            else OnlinePersonalizer())
            print(f"[Policy] personalisation profile for '{user}': "
                  f"{personalizer.n_updates} past updates")
            policy = PersonalizedPolicy(policy, personalizer)
    elif name == "rule_cooldown":
        policy = RuleCooldownPolicy(cooldown=2)
    elif name == "rule":
        policy = RuleTablePolicy()
    elif name == "random":
        policy = RandomPolicy(seed=int(time.time()))
    else:
        raise SystemExit(f"Unknown policy '{name}'")
    builder = ObservationBuilder(env_cfg["history_len"], env_cfg["observation_mode"],
                                 env_cfg["max_steps"], fixed_time_frac=LIVE_TIME_FRAC)
    return policy, builder, profile_path


# ─── Assistant ─────────────────────────────────────────────────────────────────
class Assistant:
    def __init__(self, cfg, condition=None):
        a = cfg["assistant"]
        self.cfg = cfg
        self.decision_interval = float(a["decision_interval"])
        self.condition = condition

        model_name, calibration = a.get("emotion_model") or "auto", None
        if model_name == "auto":  # the user's camera-check profile, if there is one
            profile = load_detector_profile(a["user"], a["profile_dir"])
            if profile:
                model_name, calibration = profile["emotion_model"], profile["bias"]
                print(f"[Detector] using {model_name} with {a['user']}'s calibration "
                      f"{profile['bias_by_emotion']}")
            else:
                model_name = "hsemotion"  # AffectNet model: better than DeepFace on fear/sad
                print(f"[Detector] no profile for '{a['user']}': run "
                      f"python -m emotion_rl.camera_check --user {a['user']} to calibrate")
        try:
            classifier = make_classifier(model_name)
        except Exception as exc:
            print(f"[Detector] {model_name} unavailable ({exc}); using deepface")
            classifier, calibration = None, None
        self.detector = EmotionDetector(detector_backend=a["detector_backend"],
                                        classifier=classifier, calibration=calibration)
        self.worker = DetectionWorker(self.detector).start()

        self.speech = SilentSpeech() if a["mute"] else SpeechEngine(
            rate=cfg["speech"]["rate"], voice_hint=cfg["speech"]["voice_hint"],
            output=cfg["speech"].get("output", "auto"))
        self.llm = ResponseGenerator(models=cfg["llm"]["models"],
                                     history_size=cfg["llm"]["history_size"],
                                     timeout_seconds=float(cfg["llm"]["timeout_seconds"]),
                                     enabled=cfg["llm"]["enabled"])
        self.recognizer = None
        if a["voice_input"]:
            try:
                self.recognizer = SpeechRecognizer(
                    model_size=a.get("stt_model", "small.en"), device=a.get("microphone"),
                    save_dir=os.path.join("data", "audio") if a.get("save_audio") else None)
            except Exception as exc:
                print(f"[Speech input] disabled: {exc}")

        self.policy, self.obs_builder, self.profile_path = make_policy(
            a["policy"], a["model"], a["user"], a["profile_dir"])
        self.policy_label = f"condition {condition}" if condition else a["policy"]
        self.logger = InteractionLogger(a["log_dir"], user=a["user"], condition=condition,
                                        policy_name=a["policy"])
        self.latency = LatencyTracker()
        safety = cfg["safety"]
        self.helpline_message = safety.get("helpline_message") or DEFAULT_HELPLINE_MESSAGE
        self.monitor = WellbeingMonitor(seconds=float(safety["low_mood_seconds"]))

        self.turn = 0
        self.pending = None  # (obs, probs, action, repeated, response) awaiting its reward
        self.feedback = 0.0
        self.user_text = None
        self.response_text = ""
        self.banner = None
        self.thinking = False
        self.paused = False
        self.last_decision = 0.0
        self.last_detection_time = 0.0
        self.last_face_time = time.time()
        self.current_action = "listen"
        self.toast = ("", 0.0)  # (message, show until): confirms every key press / click
        self.buttons = []  # (x0, y0, x1, y1, key), refreshed every frame
        self.clicked = None

    # ── one turn ────────────────────────────────────────────────────────────
    def decide(self, detection):
        probs = detection.probs
        if self.pending is not None:
            obs, prev_probs, action, repeated, response = self.pending
            reward = live_reward(prev_probs, probs, action, repeated, self.feedback)
            self.policy.update(obs, action, reward)
            self.logger.transition(self.turn, prev_probs, action, probs, reward, self.feedback,
                                   response)
            self.feedback = 0.0
            self.pending = None

        obs = self.obs_builder.push(probs)
        user_text, self.user_text = self.user_text, None

        if user_text and detect_crisis(user_text):
            self._safety_override("crisis_language", self.helpline_message, user_text)
            return
        if self.monitor.update(probs):
            self._safety_override("sustained_low_mood", DEFAULT_CHECK_IN_MESSAGE, user_text)
            return

        with self.latency.measure("policy"):
            action = self.policy.act(obs)
        if user_text and action == LISTEN:
            action = self._speaking_action(obs)  # the user spoke to us: always answer
        if action != LISTEN and (self.speech.busy or self.thinking or self.user_talking):
            action = LISTEN  # someone is talking: what the user experiences is silence
        repeated = action != LISTEN and action == self.obs_builder.last_action
        self.obs_builder.record_action(action)
        self.current_action = ACTIONS[action]
        self.turn += 1

        response_holder = {}
        if action != LISTEN:
            self._respond(detection, ACTIONS[action], user_text, response_holder)
        self.pending = (obs, probs, action, repeated, response_holder)
        print(f"[Turn {self.turn:3d}] {detection.emotion:8s} {detection.confidence:3.0f}% -> "
              f"{ACTIONS[action]}")

    def _respond(self, detection, action_name, user_text, holder):
        self.thinking = True
        started = time.perf_counter()

        def work():
            try:
                text, source = self.llm.generate(detection.emotion, detection.confidence,
                                                 action_name, user_text)
                self.latency.add("llm", (time.perf_counter() - started) * 1000)
                self.latency.add("end_to_end", (time.time() - detection.timestamp) * 1000)
                holder.update(text=text, source=source)
                self.response_text = text
                if not self.user_talking:  # never talk over the user (or into the mic)
                    self.speech.speak(text)
            finally:
                self.thinking = False

        threading.Thread(target=work, daemon=True).start()

    def _safety_override(self, reason, message, user_text):
        self.logger.write("safety", {"reason": reason, "user_text": user_text})
        self.banner = message
        self.response_text = message
        self.speech.speak(message)
        self.obs_builder.record_action(LISTEN)  # not a policy decision: excluded from learning
        self.current_action = "safety message"

    # ── input ───────────────────────────────────────────────────────────────
    @property
    def user_talking(self):
        return self.recognizer is not None and self.recognizer.busy

    def _speaking_action(self, obs):
        """Best speaking response when staying quiet is not an option: the policy's own
        preference among speaking actions if it has one, otherwise the v1 rule table."""
        base = getattr(self.policy, "base", self.policy)
        if hasattr(base, "action_probs"):
            probs = np.array(base.action_probs(obs), dtype=float)
            probs[LISTEN] = -1.0
            return int(np.argmax(probs))
        return V1_RULE_TABLE[int(np.argmax(current_emotion_probs(obs)))]

    def notify(self, message, seconds=2.5):
        self.toast = (message, time.time() + seconds)
        print(f"[UI] {message}")

    def handle_key(self, key):
        """key: a character such as "g" (case-insensitive, so Caps Lock does not matter)."""
        key = key.lower()
        if key in ("g", "b"):
            if self.pending is None:
                self.notify("Nothing to rate yet - wait for the assistant to respond")
                return
            self.feedback = 1.0 if key == "g" else -1.0
            self.logger.write("feedback", {"value": int(self.feedback)})
            self.notify("Thanks - noted that it HELPED" if key == "g"
                        else "Noted - that did NOT help")
        elif key == "p":
            self.paused = not self.paused
            self.pending = None
            self.notify("Paused" if self.paused else "Resumed")
        elif key == "t":
            if self.recognizer is None:
                self.notify("Voice input is off - start the app with --voice-input", 4)
            elif self.recognizer.listen_async(self._on_user_text,
                                              wait_until_quiet=lambda: self.speech.busy):
                self.speech.interrupt()  # stop talking so the mic hears only the user
                self.notify("Listening - speak now, I stop when you pause", 3)
            else:
                self.notify("Still listening...")

    def _on_user_text(self, text):
        print(f"[You said] {text}")
        self.user_text = text
        self.logger.write("user_speech", {"text": text})
        self.notify(f'You said: "{text}"' if text else "Didn't catch that - try again", 5)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, key in self.buttons:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.clicked = key

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self, duration=None):
        a = self.cfg["assistant"]
        print("[Assistant] loading face and emotion models...")
        self.detector.warm_up()
        cap = open_camera(a["camera"], a["camera_backend"], a["camera_width"], a["camera_height"])
        if not cap.isOpened():
            raise SystemExit(f"[ERROR] Could not open camera {a['camera']}; try --camera 1")
        self.speech.speak("Hello! I am here with you. Click Helped or Didn't help any time "
                          "to tell me whether something helped.")
        print("[Assistant] running. Buttons or keys: G good, B bad, T talk, P pause, Q quit")
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        shown = False
        started = time.time()
        try:
            while duration is None or time.time() - started < duration:
                ok, frame = cap.read()
                if not ok:
                    print("[ERROR] Could not read a frame from the camera")
                    break
                frame = cv2.flip(frame, 1)
                self.worker.submit(frame)
                detection = self.worker.latest
                if detection is not None:
                    if detection.timestamp != self.last_detection_time:
                        self.last_detection_time = detection.timestamp
                        self.latency.add("detector", detection.latency_ms)
                    if detection.face_found:
                        self.last_face_time = time.time()
                    self._maybe_decide(detection)
                frame = draw_detection(frame, detection)
                cv2.imshow(WINDOW, self.draw_ui(frame, detection))
                if not shown:  # launched from a terminal or background process: stay in front
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
                    shown = True
                code = cv2.waitKey(1) & 0xFF
                key = self.clicked or (chr(code).lower() if code != 255 else None)
                self.clicked = None
                if key == "q" or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key:
                    self.handle_key(key)
        finally:
            self.shutdown(cap)

    def _maybe_decide(self, detection):
        now = time.time()
        if self.paused or self.user_talking or now - self.last_decision < self.decision_interval:
            return  # no turns while the user talks: a moving mouth is misread as emotion
        if not detection.face_found:
            if now - self.last_face_time > 2 * self.decision_interval:
                self.pending = None  # user left: do not learn from a stale turn
            return
        self.last_decision = now
        self.decide(detection)

    def shutdown(self, cap):
        self.worker.stop()
        self.speech.stop()
        cap.release()
        cv2.destroyAllWindows()
        summary = self.latency.summary()
        if self.profile_path and isinstance(self.policy, PersonalizedPolicy):
            self.policy.personalizer.save(self.profile_path)
            print(f"[Policy] saved profile -> {self.profile_path}")
        self.logger.close({"turns": self.turn, "latency": summary})
        print(f"[Assistant] {self.turn} turns, log -> {self.logger.path}")
        print("[Latency]\n" + self.latency.report())

    # ── overlay ─────────────────────────────────────────────────────────────
    def draw_ui(self, frame, detection):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (30, 30, 30), -1)
        cv2.putText(frame, f"Emotion-aware RL assistant | {self.policy_label} | turn {self.turn}",
                    (10, 20), FONT, 0.5, (220, 220, 220), 1)
        status = "PAUSED" if self.paused else (
            "listening to you" if self.recognizer and self.recognizer.listening else
            "one moment..." if self.recognizer and self.recognizer.waiting else
            "thinking" if self.thinking else "speaking" if self.speech.is_speaking else "")
        if status:
            cv2.putText(frame, status, (w - 150, 20), FONT, 0.5, (0, 200, 255), 1)

        if detection is not None:
            self._draw_probability_bars(frame, detection)
            if not detection.face_found:
                cv2.putText(frame, "No face detected - waiting", (10, 60), FONT, 0.6,
                            (0, 0, 255), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 135), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"Action: {self.current_action}", (10, h - 112), FONT, 0.55,
                    (0, 255, 255), 1)
        for i, line in enumerate(wrap_text(self.response_text, w - 20, 0.5)[:2]):
            cv2.putText(frame, line, (10, h - 86 + 24 * i), FONT, 0.5, (255, 255, 255), 1)
        self._draw_buttons(frame)
        message, until = self.toast
        if message and time.time() < until:
            (tw, th), _ = cv2.getTextSize(message, FONT, 0.8, 2)
            x, y = (w - tw) // 2, h // 2
            cv2.rectangle(frame, (x - 16, y - th - 16), (x + tw + 16, y + 16), (20, 20, 20), -1)
            cv2.putText(frame, message, (x, y), FONT, 0.8, (255, 255, 255), 2)
        if self.banner:
            cv2.rectangle(frame, (0, 32), (w, 56), (0, 0, 160), -1)
            cv2.putText(frame, "Support resources shown below - you are not alone", (10, 50),
                        FONT, 0.5, (255, 255, 255), 1)
        return frame

    def _draw_buttons(self, frame):
        h, w = frame.shape[:2]
        self.buttons, x = [], 10
        for label, key, color in BUTTONS:
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            x1, y0, y1 = x + tw + 20, h - 34, h - 6
            active = (key == "p" and self.paused) or (
                key == "t" and self.recognizer is not None and self.recognizer.listening)
            cv2.rectangle(frame, (x, y0), (x1, y1), (0, 200, 255) if active else color, -1)
            cv2.putText(frame, label, (x + 10, y1 - 9), FONT, 0.5, (255, 255, 255), 1)
            self.buttons.append((x, y0, x1, y1, key))
            x = x1 + 8

    def _draw_probability_bars(self, frame, detection):
        x0, y0, bar_w = frame.shape[1] - 170, 70, 100
        for i, name in enumerate(EMOTIONS):
            y = y0 + i * 18
            p = float(detection.probs[i])
            cv2.putText(frame, name[:7], (x0, y + 10), FONT, 0.38, (220, 220, 220), 1)
            cv2.rectangle(frame, (x0 + 58, y), (x0 + 58 + bar_w, y + 11), (60, 60, 60), 1)
            cv2.rectangle(frame, (x0 + 58, y), (x0 + 58 + int(bar_w * p), y + 11),
                          EMOTION_COLORS[name], -1)


def wrap_text(text, max_width, scale):
    lines, current = [], ""
    for word in (text or "").split():
        candidate = f"{current} {word}".strip()
        if cv2.getTextSize(candidate, FONT, scale, 1)[0][0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ─── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the live emotion-aware assistant")
    p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    p.add_argument("--camera", type=int)
    p.add_argument("--policy", choices=["ppo", "ppo_personalised", "rule_cooldown", "rule",
                                        "random"])
    p.add_argument("--model")
    p.add_argument("--user")
    p.add_argument("--condition", help="blinded study condition code (see configs/default.yaml)")
    p.add_argument("--decision-interval", type=float)
    p.add_argument("--detector-backend")
    p.add_argument("--emotion-model", help="path to your own emotion_cnn.pt")
    p.add_argument("--voice-input", action="store_true")
    p.add_argument("--mute", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--duration", type=float, default=None,
                   help="stop after this many seconds (timed test sessions)")
    return p.parse_args(argv)


def main(argv=None):
    import torch

    torch.set_num_threads(1)
    load_dotenv()
    args = parse_args(argv)
    cfg = load_config(args.config)
    a = cfg["assistant"]
    for key in ("camera", "policy", "model", "user", "decision_interval", "detector_backend",
                "emotion_model"):
        value = getattr(args, key)
        if value is not None:
            a[key] = value
    a["voice_input"] = a["voice_input"] or args.voice_input
    a["mute"] = a["mute"] or args.mute
    if args.no_llm:
        cfg["llm"]["enabled"] = False
    if args.condition:
        conditions = cfg["study"]["conditions"]
        if args.condition not in conditions:
            raise SystemExit(f"Unknown condition '{args.condition}'. Known: {list(conditions)}")
        a["policy"] = conditions[args.condition]
    np.set_printoptions(precision=2, suppress=True)
    Assistant(cfg, condition=args.condition).run(duration=args.duration)


if __name__ == "__main__":
    main()
