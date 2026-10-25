"""Safety guardrails. This assistant is a demo, not a mental-health tool.

* detect_crisis(text): spoken/typed text that suggests self-harm or suicidal
  thoughts overrides the policy and shows a helpline message instead.
* WellbeingMonitor: if the detected emotion stays clearly negative for a long
  time, the assistant gently suggests talking to someone (once per session).
* sanitize_response(text): post-filter for LLM output (length, emojis, AI talk).

The helpline text is configurable (configs/default.yaml); check that the
numbers are right for where you deploy.
"""

import re
import time

import numpy as np

from emotion_rl.emotions import VALENCE

CRISIS_PATTERNS = [
    r"\bkill(ing)? myself\b", r"\bsuicid", r"\bend (it all|my life)\b", r"\bwant(ed)? to die\b",
    r"\bself[- ]?harm", r"\bhurt(ing)? myself\b", r"\bno reason to live\b",
    r"\bbetter off (dead|without me)\b", r"\bcan'?t go on\b", r"\bcut(ting)? myself\b",
]
_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

DEFAULT_HELPLINE_MESSAGE = (
    "It sounds like things are really hard right now, and you deserve support from a real "
    "person. Please reach out to someone you trust or a helpline. In India you can call "
    "Tele-MANAS on 14416; elsewhere, findahelpline.com lists free services near you."
)
DEFAULT_CHECK_IN_MESSAGE = (
    "You have seemed low for a while. Talking to a friend, family member or counsellor can "
    "really help, and you do not have to go through it alone."
)

_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")
_AI_TALK_RE = re.compile(r"\b(as an ai|i am an ai|i'm an ai|language model)\b[^.!?]*[.!?]?",
                         re.IGNORECASE)


def detect_crisis(text):
    return bool(text) and _CRISIS_RE.search(text) is not None


def sanitize_response(text, max_sentences=2, max_chars=240):
    text = _EMOJI_RE.sub("", text or "")
    text = _AI_TALK_RE.sub("", text)
    text = text.replace("*", "").strip().strip('"').strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(s for s in sentences[:max_sentences] if s).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return text


class WellbeingMonitor:
    """Fires once when expected valence stays below `threshold` for `seconds`."""

    def __init__(self, threshold=-0.35, seconds=180.0, clock=time.time):
        self.threshold = threshold
        self.seconds = seconds
        self.clock = clock
        self.low_since = None
        self.fired = False

    def update(self, emotion_probs):
        valence = float(np.asarray(emotion_probs) @ VALENCE)
        now = self.clock()
        if valence >= self.threshold:
            self.low_since = None
            return False
        if self.low_since is None:
            self.low_since = now
        if not self.fired and now - self.low_since >= self.seconds:
            self.fired = True
            return True
        return False
