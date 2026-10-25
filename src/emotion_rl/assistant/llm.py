"""Turns (detected emotion, chosen action, what the user said) into one short reply.

Changes from v1:
* Actions become natural instructions ("kindly suggest they take a short
  break") instead of "Your task is to suggest_break them".
* The last few exchanges go into the prompt, so replies do not repeat.
* No test call per model at start-up (it cost quota and seconds). Models are
  tried in the configured order only when a request fails.
* "-latest" aliases instead of pinned versions: v1's pinned models (and the
  2.x models first used here) were retired and silently fell back every time.
* A live assistant needs fast replies: the lite model with minimal thinking
  answered in ~1.5 s in testing, the full flash model took 38-250 s. Requests
  time out after `timeout_seconds` and the local reply is used instead.
* Output passes through safety.sanitize_response; offline or on errors, an
  emotion- and action-specific local reply is used.
"""

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from emotion_rl.assistant.safety import sanitize_response
from emotion_rl.emotions import ACTION_INSTRUCTIONS

DEFAULT_MODELS = ("gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-flash-latest")

FALLBACK_RESPONSES = {
    "angry": {
        "comfort": "I can feel your frustration. Take a slow breath, I am right here with you.",
        "motivate": "Channel that energy into the next small step. You are stronger than this.",
        "acknowledge": "It looks like something is bothering you. That is completely okay.",
        "suggest_break": "It might help to step away for a minute. A short break calms the storm.",
        "playful": "Let us shake that off together. Even storms run out of rain eventually.",
    },
    "disgust": {
        "comfort": "Something seems to be bothering you. I am here if you want to talk about it.",
        "motivate": "Do not let this drag you down. You are bigger than whatever this is.",
        "acknowledge": "That does not look pleasant. Take your time.",
        "suggest_break": "Sometimes stepping back gives a fresh perspective. Take a breather.",
        "playful": "Let us find something nicer to look at, preferably with fewer surprises.",
    },
    "fear": {
        "comfort": "You are safe right now. Whatever you are facing, you do not face it alone.",
        "motivate": "Courage is taking the next step anyway, and you are already doing it.",
        "acknowledge": "This seems a bit unsettling. It is okay to feel that way.",
        "suggest_break": "Try one slow, deep breath with me. Give yourself a moment to reset.",
        "playful": "You are braver than you think, probably braver than me on a Monday.",
    },
    "happy": {
        "comfort": "It is lovely to see you smiling. Hold on to this feeling.",
        "motivate": "You are on a roll. Ride this energy into whatever comes next.",
        "acknowledge": "You seem to be in good spirits. That is great to see.",
        "suggest_break": "You have earned a happy little pause. Enjoy the moment.",
        "playful": "That smile is contagious. Careful, I might start grinning too.",
    },
    "sad": {
        "comfort": "I am sorry you are feeling low. It is okay to feel this way, I am here.",
        "motivate": "Even on hard days you keep going, and that takes real strength.",
        "acknowledge": "You seem a little down. You do not have to explain anything.",
        "suggest_break": "Be gentle with yourself. A short break and some water might help.",
        "playful": "Let us find one tiny thing to smile about. You deserve it.",
    },
    "surprise": {
        "comfort": "That caught you off guard. Take a moment to settle, I am here.",
        "motivate": "Surprises can be the start of something good. Lean into it.",
        "acknowledge": "Something unexpected happened. How are you feeling about it?",
        "suggest_break": "Take a breath and let the surprise settle before you react.",
        "playful": "Plot twist! Life likes to keep us on our toes.",
    },
    "neutral": {
        "comfort": "I am here with you. Let me know if there is anything on your mind.",
        "motivate": "Every moment is a fresh start. What would you like to get done?",
        "acknowledge": "Just checking in. I am here whenever you need me.",
        "suggest_break": "A small pause can do wonders. Stretch for a moment if you like.",
        "playful": "Let us liven things up a little. What sounds fun right now?",
    },
}


def build_prompt(emotion, confidence, action, history, user_text=None):
    instruction = ACTION_INSTRUCTIONS[action]
    lines = [
        "You are a warm, supportive companion talking with someone at their computer.",
        f"Their facial expression currently reads as {emotion} "
        f"({confidence:.0f}% confidence; the reading can be wrong, so never state their "
        "emotion as a fact).",
        f"Your goal for this reply: {instruction}.",
    ]
    if user_text:
        lines.append(f'They just said: "{user_text}"')
    if history:
        lines.append("Your recent replies (do not repeat them):")
        lines.extend(f"- {h}" for h in history)
    lines += [
        "Rules: at most two short sentences; sound natural and kind; no emojis; do not mention "
        "being an AI, the camera or emotion detection; no medical or diagnostic advice.",
        "Reply with the spoken text only.",
    ]
    return "\n".join(lines)


class ResponseGenerator:
    def __init__(self, api_key=None, models=DEFAULT_MODELS, history_size=4, enabled=True,
                 timeout_seconds=8.0):
        self.models = list(models)
        self.history = deque(maxlen=history_size)
        self.timeout_seconds = timeout_seconds
        self.client = None
        self.no_thinking_config = set()  # models that reject the thinking setting
        # The API rejects short server-side deadlines, so the timeout is enforced here:
        # a slow request keeps running in this pool while the local reply is used.
        self.pool = ThreadPoolExecutor(max_workers=2)
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if enabled and api_key:
            from google import genai
            from google.genai import types

            self.types = types
            self.client = genai.Client(api_key=api_key)
        elif enabled:
            print("[LLM] GEMINI_API_KEY not set; using local fallback replies")

    def _request(self, model, prompt):
        config = None
        if model not in self.no_thinking_config:
            config = self.types.GenerateContentConfig(
                thinking_config=self.types.ThinkingConfig(thinking_level="minimal"))
        try:
            return self.client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as exc:
            if config is not None and "thinking" in str(exc).lower():
                self.no_thinking_config.add(model)
                return self.client.models.generate_content(model=model, contents=prompt)
            raise

    def generate(self, emotion, confidence, action, user_text=None):
        """Returns (text, source) where source is the model name or 'fallback'."""
        if self.client is not None:
            prompt = build_prompt(emotion, confidence, action, list(self.history), user_text)
            deadline = time.monotonic() + self.timeout_seconds
            for model in list(self.models):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    response = self.pool.submit(self._request, model, prompt).result(remaining)
                    text = sanitize_response(response.text or "")
                    if text:
                        self._remember(text)
                        if model != self.models[0]:  # stick with the model that works
                            self.models.remove(model)
                            self.models.insert(0, model)
                        return text, model
                except FutureTimeout:
                    print(f"[LLM] {model} took over {self.timeout_seconds:.0f}s; using local reply")
                    break
                except Exception as exc:
                    print(f"[LLM] {model} failed: {str(exc)[:120]}")
        text = FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"])[action]
        self._remember(text)
        return text, "fallback"

    def _remember(self, text):
        self.history.append(text)
