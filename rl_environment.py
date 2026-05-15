import gymnasium as gym
from gymnasium import spaces
import numpy as np
from emotion_detector import (
    EmotionDetector,
    EMOTIONS_TO_INDEX,
    INDEX_TO_EMOTION,
    NUM_EMOTIONS
)

ACTIONS = {
    0: "comfort",
    1: "motivate",
    2: "neutral",
    3: "suggest_break",
    4: "playful",
}

NUM_ACTIONS = len(ACTIONS)

EMOTION_POSITIVITY = {
    0: -2,
    1: -2,
    2: -1,
    3: +2,
    4: -1,
    5: +1,
    6: 0,
}

BEST_ACTION_FOR_EMOTION = {
    0: 0,
    1: 0,
    2: 3,
    3: 4,
    4: 0,
    5: 1,
    6: 2,
}


class EmotionRLEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, use_webcam=False, max_steps=20):

        super(EmotionRLEnvironment, self).__init__()
        self.use_webcam = use_webcam
        self.max_steps = max_steps
        self.step_count = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_EMOTIONS,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.detector = EmotionDetector()

        self.current_emotion_idx = 6
        self.previous_emotion_idx = 6

        self.episode_history = []
        self.total_reward = 0.0

        print(f"\n[Environment] Initialized!")
        print(f"[Environment] Mode       : {'Webcam' if use_webcam else 'Simulation'}")
        print(f"[Environment] State space: {NUM_EMOTIONS} emotions")
        print(f"[Environment] Actions    : {NUM_ACTIONS} response types")
        print(f"[Environment] Max steps  : {self.max_steps}")
        print(f"[Environment] Actions    : {ACTIONS}\n")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_history = []
        self.previous_emotion_idx = 6
        self.current_emotion_idx = 6
        if self.use_webcam:
            state = self._get_emotion_from_webcam()
        else:
            state = self._get_simulated_emotion()
        info = {
            'emotion': INDEX_TO_EMOTION.get(self.current_emotion_idx, 'neutral'),
            'step': 0
        }
        print(f"[Episode Start] Initial emotion: "
              f"{INDEX_TO_EMOTION.get(self.current_emotion_idx, 'neutral').upper()}")
        return state, info

    def step(self, action):
        self.step_count += 1
        self.previous_emotion_idx = self.current_emotion_idx
        action_name = ACTIONS.get(action, 'neutral')

        if self.use_webcam:
            new_state = self._get_emotion_from_webcam()
        else:
            new_state = self._get_simulated_emotion()

        reward = self.calculate_reward(self.previous_emotion_idx, self.current_emotion_idx, action)

        self.total_reward += reward
        terminated = False
        truncated = self.step_count >= self.max_steps

        step_log = {
            'step': self.step_count,
            'action': action_name,
            'prev_emotion': INDEX_TO_EMOTION.get(self.previous_emotion_idx, 'neutral'),
            'curr_emotion': INDEX_TO_EMOTION.get(self.current_emotion_idx, 'neutral'),
            'reward': reward,
            'cumalative_reward': self.total_reward,
        }
        self.episode_history.append(step_log)

        print(f"[Step {self.step_count:2d}] "
              f"Action: {action_name:15s} | "
              f"Emotion: {step_log['prev_emotion']:10s} → "
              f"{step_log['curr_emotion']:10s} | "
              f"Reward: {reward:+.2f} | "
              f"Total: {self.total_reward:+.2f}")

        info = {
            'action': action_name,
            'prev_emotion': step_log['prev_emotion'],
            'curr_emotion': step_log['curr_emotion'],
            'reward': reward,
            'step': self.step_count,
        }

        return new_state, reward, terminated, truncated, info

    def calculate_reward(self, prev_idx, curr_idx, action):
        prev_score = EMOTION_POSITIVITY.get(prev_idx, 0)
        curr_score = EMOTION_POSITIVITY.get(curr_idx, 0)
        delta = curr_score + prev_score

        if delta > 0:
            reward = +1.0
        elif delta < 0:
            reward = -1.0
        else:
            reward = -0.1

        if prev_idx in [0, 1, 2, 4] and curr_idx == 3:
            reward += 1.0

        best_action = BEST_ACTION_FOR_EMOTION.get(prev_idx, 2)
        if action == best_action:
            reward += 0.2
        return round(reward, 2)

    def _get_emotion_from_webcam(self):
        import cv2
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        cap.release()

        if ret:
            _, idx, _, _ = self.detector.detect_from_frame(frame)
            self.current_emotion_idx = idx
            return self.detector.get_emotion_vector()
        else:
            return self._get_neutral_state()

    def _get_simulated_emotion(self):
        transition_matrix = np.array([
            [0.40, 0.10, 0.10, 0.10, 0.10, 0.00, 0.20],
            [0.10, 0.40, 0.10, 0.10, 0.10, 0.00, 0.20],
            [0.10, 0.00, 0.40, 0.10, 0.10, 0.10, 0.20],
            [0.00, 0.00, 0.00, 0.50, 0.00, 0.20, 0.30],
            [0.10, 0.00, 0.10, 0.10, 0.40, 0.00, 0.30],
            [0.00, 0.00, 0.00, 0.30, 0.00, 0.40, 0.30],
            [0.10, 0.00, 0.10, 0.20, 0.10, 0.10, 0.40],
        ])

        probs = transition_matrix[self.current_emotion_idx]
        new_idx = np.random.choice(NUM_EMOTIONS, p=probs)
        self.current_emotion_idx = new_idx

        vector = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        vector[new_idx] = 1.0
        return vector

    def _get_neutral_state(self):
        vector = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        vector[6] = 1.0
        return vector

    def get_action_name(self, action):
        return ACTIONS.get(action, 'unknown')

    def get_episode_summary(self):
        if not self.episode_history:
            return {}
        return {
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.step_count, 1),
            'history': self.episode_history,
        }

    def render(self):
        emotion = INDEX_TO_EMOTION.get(self.current_emotion_idx, 'neutral')
        print(f"[Render] Step: {self.step_count} | "
              f"Emotion: {emotion} | "
              f"Total Reward: {self.total_reward:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing Emotion RL Environment — Simulation Mode")
    print("=" * 60)

    env = EmotionRLEnvironment(use_webcam=False, max_steps=10)

    obs, info = env.reset()
    print(f"\nInitial Observation : {obs}")
    print(f"Initial Emotion     : {info['emotion']}\n")
    print("Running 10 steps with random actions...")
    print("-" * 60)

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("-" * 60)
    summary = env.get_episode_summary()
    print(f"\n[Episode Summary]")
    print(f"Total Steps  : {summary['total_steps']}")
    print(f"Total Reward : {summary['total_reward']:.2f}")
    print(f"Avg Reward   : {summary['avg_reward']:.2f}")
    print("\n[Environment Test Passed!] ✓")
