"""Emotion-aware assistant with a reinforcement-learned response policy."""

import os

# SB3 imports TensorBoard, which loads TensorFlow when it is installed; keep its C++ logs quiet.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__version__ = "2.0.0"
