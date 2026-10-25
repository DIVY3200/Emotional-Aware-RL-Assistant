"""Loads configs/default.yaml (or another YAML file) into a nested dict."""

import copy
import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join("configs", "default.yaml")

FALLBACK_CONFIG = {
    "assistant": {"camera": 0, "camera_backend": "auto", "camera_width": 1280,
                  "camera_height": 720, "microphone": None, "policy": "ppo_personalised",
                  "model": "models/ppo_best.zip",
                  "decision_interval": 4.0, "detector_backend": "yunet", "emotion_model": "auto",
                  "user": "anonymous", "log_dir": "data/interactions",
                  "profile_dir": "data/profiles", "mute": False, "voice_input": False,
                  "stt_model": "small.en", "save_audio": False},
    "llm": {"enabled": True, "models": ["gemini-flash-lite-latest", "gemini-3.5-flash-lite",
                                        "gemini-flash-latest"], "history_size": 4,
            "timeout_seconds": 8},
    "speech": {"rate": 165, "voice_hint": "zira", "output": "auto"},
    "safety": {"helpline_message": None, "low_mood_seconds": 180},
    "study": {"conditions": {}},
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None):
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return copy.deepcopy(FALLBACK_CONFIG)
    with open(path, encoding="utf-8") as f:
        return _merge(FALLBACK_CONFIG, yaml.safe_load(f) or {})
