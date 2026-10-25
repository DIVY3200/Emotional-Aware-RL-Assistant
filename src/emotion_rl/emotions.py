"""Emotion and action vocabularies shared by every module.

Emotion indices follow the FER2013 convention used by DeepFace:
0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral.
"""

import numpy as np

# ─── Emotions ──────────────────────────────────────────────────────────────────
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
EMOTION_TO_INDEX = {name: i for i, name in enumerate(EMOTIONS)}
NUM_EMOTIONS = len(EMOTIONS)
NEUTRAL = EMOTION_TO_INDEX["neutral"]
NEGATIVE_EMOTIONS = (0, 1, 2, 4)

# Approximate (valence, arousal) coordinates on Russell's circumplex of affect
# (Russell, 1980, "A circumplex model of affect"). These are hand-set modelling
# assumptions in [-1, 1], not measured values. Valence drives the reward.
VALENCE = np.array([-0.55, -0.60, -0.65, 0.80, -0.70, 0.30, 0.00], dtype=np.float32)
AROUSAL = np.array([0.60, 0.35, 0.60, 0.50, -0.40, 0.70, 0.00], dtype=np.float32)

# BGR colours for OpenCV overlays
EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 128),
    "fear": (128, 0, 128),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
    "neutral": (255, 255, 0),
}

# ─── Actions ───────────────────────────────────────────────────────────────────
# "listen" means the assistant stays quiet this turn. Learning *when not to
# speak* is part of the policy, instead of a hard-coded cooldown.
ACTIONS = ("comfort", "motivate", "acknowledge", "suggest_break", "playful", "listen")
ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)
LISTEN = ACTION_TO_INDEX["listen"]
SPEAKING_ACTIONS = tuple(i for i in range(NUM_ACTIONS) if i != LISTEN)

# Natural-language instruction for the LLM, one per speaking action
ACTION_INSTRUCTIONS = {
    "comfort": "comfort them with warmth and empathy",
    "motivate": "encourage them and give them a little motivation",
    "acknowledge": "gently acknowledge how they seem to feel, without pushing advice",
    "suggest_break": "kindly suggest they take a short break to reset",
    "playful": "lift their mood with light, kind humour",
}

# The hand-written emotion → action table from v1 of this project. It is kept
# only as a baseline policy so the learned agent can be compared against it.
V1_RULE_TABLE = {
    0: ACTION_TO_INDEX["comfort"],
    1: ACTION_TO_INDEX["comfort"],
    2: ACTION_TO_INDEX["suggest_break"],
    3: ACTION_TO_INDEX["playful"],
    4: ACTION_TO_INDEX["comfort"],
    5: ACTION_TO_INDEX["motivate"],
    6: ACTION_TO_INDEX["acknowledge"],
}


def one_hot(index, size):
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector
