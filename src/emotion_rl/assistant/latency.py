"""Per-stage latency profiling for the live pipeline (detector, policy, LLM, TTS)."""

import json
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class LatencyTracker:
    def __init__(self):
        self.samples = defaultdict(list)

    @contextmanager
    def measure(self, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - start) * 1000)

    def add(self, stage, ms):
        self.samples[stage].append(float(ms))

    def summary(self):
        out = {}
        for stage, values in self.samples.items():
            v = np.asarray(values)
            out[stage] = {"n": int(len(v)), "mean_ms": float(v.mean()),
                          "p50_ms": float(np.percentile(v, 50)),
                          "p95_ms": float(np.percentile(v, 95)), "max_ms": float(v.max())}
        return out

    def last(self, stage):
        values = self.samples.get(stage)
        return values[-1] if values else None

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def report(self):
        lines = [f"  {'stage':12s} {'n':>5s} {'p50':>8s} {'p95':>8s}"]
        for stage, s in self.summary().items():
            lines.append(f"  {stage:12s} {s['n']:5d} {s['p50_ms']:7.0f}ms {s['p95_ms']:7.0f}ms")
        return "\n".join(lines)
