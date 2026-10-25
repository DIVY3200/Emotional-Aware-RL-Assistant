"""Stochastic user simulator used to train and evaluate response policies.

Why a simulator: v1 sampled the next emotion from a fixed matrix that ignored
the assistant's action, so no policy could change the outcome. Here the next
emotion depends on (current emotion, action, hidden user type, fatigue):

    P(s' | s, a, u)  ∝  T0[s, s'] · exp(β · e_eff(u, s, a) · valence(s'))

* T0 is the natural emotional drift (v1's matrix, smoothed so every emotion
  is reachable).
* e(u, s, a) ∈ [-1, 1] is how much action a helps user type u in emotion s.
  e > 0 tilts probability mass towards positive-valence emotions, e < 0
  towards negative ones. Different user types want different responses, so
  no single emotion → action table is optimal for everyone.
* e_eff decays when the same action is repeated and turns negative when the
  user is fatigued by too much talking; staying quiet ("listen") lets fatigue
  recover.
* The agent never sees the true emotion: it sees a noisy probability vector
  produced by a simulated emotion detector (confusion matrix + Dirichlet noise).

All numbers below are modelling assumptions, documented so they can be
replaced by values fitted from real interaction logs (see `study.py`).
"""

from dataclasses import dataclass

import numpy as np

from emotion_rl.emotions import EMOTIONS, LISTEN, NUM_ACTIONS, NUM_EMOTIONS, VALENCE


# ─── User types ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    effects: np.ndarray  # (NUM_EMOTIONS, NUM_ACTIONS), "listen" column is 0
    talk_sensitivity: float  # fatigue added every time the assistant speaks


def _effects(rows):
    """rows: emotion -> [comfort, motivate, acknowledge, suggest_break, playful]."""
    table = np.zeros((NUM_EMOTIONS, NUM_ACTIONS), dtype=np.float32)
    for emotion, values in rows.items():
        table[EMOTIONS.index(emotion), :LISTEN] = values
    return table


PERSONAS = {
    "comfort_seeker": Persona(
        "comfort_seeker",
        "Wants empathy first; jokes feel dismissive when upset.",
        _effects({
            "angry":    [0.8, -0.2, 0.2, 0.3, -0.6],
            "disgust":  [0.6, 0.0, 0.2, 0.4, -0.3],
            "fear":     [0.9, 0.1, 0.2, 0.3, -0.5],
            "happy":    [0.2, 0.2, 0.2, -0.2, 0.5],
            "sad":      [0.9, -0.1, 0.2, 0.2, -0.4],
            "surprise": [0.3, 0.3, 0.3, 0.0, 0.2],
            "neutral":  [0.3, 0.2, 0.2, 0.0, 0.3],
        }),
        talk_sensitivity=0.08,
    ),
    "humor_lover": Persona(
        "humor_lover",
        "Light humour works best, even when sad or annoyed.",
        _effects({
            "angry":    [0.1, 0.0, 0.0, 0.2, 0.7],
            "disgust":  [0.1, 0.0, 0.1, 0.2, 0.6],
            "fear":     [0.4, 0.1, 0.1, 0.1, 0.5],
            "happy":    [-0.2, 0.2, 0.1, -0.3, 0.9],
            "sad":      [0.2, 0.1, 0.1, 0.0, 0.8],
            "surprise": [0.0, 0.3, 0.2, -0.1, 0.6],
            "neutral":  [-0.1, 0.2, 0.1, -0.1, 0.6],
        }),
        talk_sensitivity=0.08,
    ),
    "space_needer": Persona(
        "space_needer",
        "Prefers space: breaks and silence help, chatter backfires.",
        _effects({
            "angry":    [-0.5, -0.6, 0.0, 0.9, -0.7],
            "disgust":  [-0.3, -0.3, 0.0, 0.7, -0.5],
            "fear":     [0.2, -0.3, 0.1, 0.7, -0.4],
            "happy":    [-0.2, 0.0, 0.2, 0.0, 0.2],
            "sad":      [0.1, -0.4, 0.2, 0.6, -0.5],
            "surprise": [-0.1, 0.0, 0.3, 0.2, 0.0],
            "neutral":  [-0.2, 0.0, 0.2, 0.1, 0.0],
        }),
        talk_sensitivity=0.20,
    ),
    "achiever": Persona(
        "achiever",
        "Goal-driven; responds to motivation, finds comfort patronising.",
        _effects({
            "angry":    [-0.1, 0.7, 0.1, 0.2, -0.3],
            "disgust":  [0.0, 0.5, 0.1, 0.2, -0.2],
            "fear":     [0.3, 0.8, 0.1, 0.0, -0.2],
            "happy":    [-0.1, 0.6, 0.1, -0.3, 0.4],
            "sad":      [0.3, 0.7, 0.1, -0.1, -0.1],
            "surprise": [0.0, 0.8, 0.2, -0.1, 0.2],
            "neutral":  [-0.1, 0.6, 0.1, -0.1, 0.2],
        }),
        talk_sensitivity=0.08,
    ),
}
PERSONA_NAMES = tuple(PERSONAS)

# ─── Natural emotional drift (v1 matrix, smoothed so every state is reachable) ─
_V1_TRANSITIONS = np.array([
    [0.40, 0.10, 0.10, 0.10, 0.10, 0.00, 0.20],
    [0.10, 0.40, 0.10, 0.10, 0.10, 0.00, 0.20],
    [0.10, 0.00, 0.40, 0.10, 0.10, 0.10, 0.20],
    [0.00, 0.00, 0.00, 0.50, 0.00, 0.20, 0.30],
    [0.10, 0.00, 0.10, 0.10, 0.40, 0.00, 0.30],
    [0.00, 0.00, 0.00, 0.30, 0.00, 0.40, 0.30],
    [0.10, 0.00, 0.10, 0.20, 0.10, 0.10, 0.40],
])
BASE_TRANSITIONS = 0.9 * _V1_TRANSITIONS + 0.1 / NUM_EMOTIONS

# Episodes start in a mix of emotions, weighted towards the ones that matter
START_DISTRIBUTION = np.array([0.15, 0.07, 0.15, 0.15, 0.20, 0.10, 0.18])


def default_confusion(accuracy=0.65):
    """Simulated detector confusion matrix: row = true emotion, column = predicted.

    Off-diagonal mass is biased towards pairs that FER models commonly mix up.
    Replace with a measured matrix via `detector_eval.py` when FER2013 is available.
    """
    weights = np.ones((NUM_EMOTIONS, NUM_EMOTIONS))
    confusable = [("fear", "surprise"), ("sad", "neutral"), ("angry", "disgust"),
                  ("fear", "sad"), ("angry", "sad")]
    for a, b in confusable:
        i, j = EMOTIONS.index(a), EMOTIONS.index(b)
        weights[i, j] = weights[j, i] = 3.0
    np.fill_diagonal(weights, 0.0)
    confusion = (1.0 - accuracy) * weights / weights.sum(axis=1, keepdims=True)
    np.fill_diagonal(confusion, accuracy)
    return confusion


def _softmax(logits):
    z = np.exp(logits - logits.max())
    return z / z.sum()


# ─── Simulator ─────────────────────────────────────────────────────────────────
class UserSimulator:
    def __init__(
        self,
        personas=PERSONA_NAMES,
        mix_personas=False,
        effect_noise=0.15,
        effect_strength=2.5,
        repeat_decay=0.75,
        fatigue_penalty=0.35,
        fatigue_recovery=0.25,
        detector_accuracy=0.65,
        detector_concentration=10.0,
        confusion=None,
    ):
        """
        Args:
            personas            : user types sampled uniformly per episode
            mix_personas        : blend two random user types per episode ("unseen" users)
            effect_noise        : per-episode Gaussian noise on the effect table
            effect_strength     : β, how strongly a response tilts the next emotion
            repeat_decay        : effect multiplier for each consecutive repeat
            fatigue_penalty     : effect lost at full fatigue (can go negative)
            fatigue_recovery    : fatigue removed by one "listen" turn
            detector_accuracy   : diagonal of the simulated confusion matrix;
                                  None gives the agent the true emotion (one-hot)
            detector_concentration : Dirichlet concentration, higher = less noise
            confusion           : optional measured (7, 7) confusion matrix
        """
        self.persona_names = tuple(personas)
        self.mix_personas = mix_personas
        self.effect_noise = effect_noise
        self.effect_strength = effect_strength
        self.repeat_decay = repeat_decay
        self.fatigue_penalty = fatigue_penalty
        self.fatigue_recovery = fatigue_recovery
        self.detector_accuracy = detector_accuracy
        self.detector_concentration = detector_concentration
        if confusion is not None:
            confusion = np.asarray(confusion, dtype=np.float64)
            confusion = confusion / confusion.sum(axis=1, keepdims=True)
        elif detector_accuracy is not None:
            confusion = default_confusion(detector_accuracy)
        self.confusion = confusion

        self.rng = np.random.default_rng()
        self.persona_name = self.persona_names[0]
        self.effects = PERSONAS[self.persona_name].effects
        self.talk_sensitivity = PERSONAS[self.persona_name].talk_sensitivity
        self.persona_weights = np.zeros(len(PERSONA_NAMES), dtype=np.float32)
        self.emotion = 6
        self.fatigue = 0.0
        self.last_action = LISTEN
        self.repeat_count = 0

    def reset(self, rng, persona=None, keep_user=False):
        """Start a new session. Returns the true emotion.

        keep_user=True starts another session with the *same* user (same effect
        table), which is what online personalisation across sessions needs.
        """
        self.rng = rng
        if keep_user:
            self._start_session()
            return self.emotion

        if self.mix_personas and persona is None:
            a, b = rng.choice(len(PERSONA_NAMES), size=2, replace=False)
            w = rng.uniform(0.2, 0.8)
            pa, pb = PERSONAS[PERSONA_NAMES[a]], PERSONAS[PERSONA_NAMES[b]]
            effects = w * pa.effects + (1 - w) * pb.effects
            self.talk_sensitivity = w * pa.talk_sensitivity + (1 - w) * pb.talk_sensitivity
            self.persona_name = f"mix:{pa.name}+{pb.name}"
            self.persona_weights = np.zeros(len(PERSONA_NAMES), dtype=np.float32)
            self.persona_weights[[a, b]] = [w, 1 - w]
        else:
            name = persona or self.persona_names[rng.integers(len(self.persona_names))]
            effects = PERSONAS[name].effects
            self.talk_sensitivity = PERSONAS[name].talk_sensitivity
            self.persona_name = name
            self.persona_weights = np.zeros(len(PERSONA_NAMES), dtype=np.float32)
            self.persona_weights[PERSONA_NAMES.index(name)] = 1.0

        noise = rng.normal(0.0, self.effect_noise, size=effects.shape)
        noise[:, LISTEN] = 0.0
        self.effects = np.clip(effects + noise, -1.0, 1.0)
        self._start_session()
        return self.emotion

    def _start_session(self):
        self.emotion = int(self.rng.choice(NUM_EMOTIONS, p=START_DISTRIBUTION))
        self.fatigue = 0.0
        self.last_action = LISTEN
        self.repeat_count = 0

    def effective_effect(self, action):
        if action == LISTEN:
            return 0.0
        repeats = self.repeat_count + 1 if action == self.last_action else 0
        effect = self.effects[self.emotion, action] * self.repeat_decay ** repeats
        return effect * (1.0 - self.fatigue) - self.fatigue_penalty * self.fatigue

    def transition_probs(self, action):
        logits = np.log(BASE_TRANSITIONS[self.emotion])
        logits = logits + self.effect_strength * self.effective_effect(action) * VALENCE
        return _softmax(logits)

    def step(self, action):
        """Apply the assistant's action. Returns the user's next true emotion."""
        probs = self.transition_probs(action)
        if action != LISTEN and action == self.last_action:
            self.repeat_count += 1
        else:
            self.repeat_count = 0

        if action == LISTEN:
            self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery)
        else:
            self.fatigue = min(1.0, self.fatigue + self.talk_sensitivity)

        self.last_action = action
        self.emotion = int(self.rng.choice(NUM_EMOTIONS, p=probs))
        return self.emotion

    def observe(self):
        """What the emotion detector would report: a probability vector over emotions.

        The top-1 label is drawn from the confusion row, so argmax accuracy matches
        the confusion matrix. The rest of the vector still leans towards the true
        emotion, like a real classifier whose second guess is often right.
        """
        if self.confusion is None:
            probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
            probs[self.emotion] = 1.0
            return probs
        row = self.confusion[self.emotion]
        predicted = self.rng.choice(NUM_EMOTIONS, p=row)
        mean = 0.5 * row
        mean[predicted] += 0.5
        alpha = self.detector_concentration * mean + 0.05
        return self.rng.dirichlet(alpha).astype(np.float32)
