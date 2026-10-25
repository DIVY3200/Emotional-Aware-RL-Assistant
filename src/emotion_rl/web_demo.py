"""Browser demo: no webcam setup needed.

    streamlit run src/emotion_rl/web_demo.py

Tab 1 "Simulator playground": pick a hidden user type and watch different
policies handle the *same* simulated session, turn by turn.
Tab 2 "Try a photo": take or upload a photo; the detector reads the expression,
the trained agent picks a response type and the reply generator words it.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emotion_rl.bandits import OnlinePersonalizer, PersonalizedPolicy  # noqa: E402
from emotion_rl.emotions import ACTIONS, EMOTIONS, LISTEN  # noqa: E402
from emotion_rl.env import EmotionSupportEnv, ObservationBuilder  # noqa: E402
from emotion_rl.policies import (  # noqa: E402
    GreedyOraclePolicy,
    RandomPolicy,
    RuleCooldownPolicy,
    RuleTablePolicy,
    SB3Policy,
    current_emotion_probs,
    read_model_config,
)
from emotion_rl.simulator import PERSONA_NAMES, PERSONAS  # noqa: E402

DEFAULT_MODEL_PATH = "models/ppo_best.zip"


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def model_path():
    return os.environ.get("EMOTION_RL_MODEL", DEFAULT_MODEL_PATH)




@st.cache_resource
def load_agent(path):
    if not os.path.exists(path):
        return None
    agent = SB3Policy.from_path(path, algo="ppo", name="PPO (ours)")
    agent.history_len = read_model_config(path).get("env", {}).get("history_len", 4)
    return agent


def history_len(agent):
    """Observations must be built with the history length the agent was trained on."""
    return getattr(agent, "history_len", 4)


def make_policy(name, agent):
    return {
        "PPO (ours)": lambda: agent,
        "PPO + online personalisation": lambda: PersonalizedPolicy(agent, OnlinePersonalizer()),
        "v1 rule table": RuleTablePolicy,
        "Rule table + cooldown": RuleCooldownPolicy,
        "Random": lambda: RandomPolicy(seed=0),
        "Greedy oracle (cheats)": GreedyOraclePolicy,
    }[name]()


def run_session(policy, persona, accuracy, seed, history=4):
    env = EmotionSupportEnv(history_len=history, detector_accuracy=accuracy,
                            mix_personas=persona == "blended (unseen)")
    options = None if persona in ("random", "blended (unseen)") else {"persona": persona}
    obs, info = env.reset(seed=seed, options=options)
    policy.reset()
    rows, done, total = [], False, 0.0
    while not done:
        action = policy.act(obs, env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        policy.update(obs, action, reward)
        total += reward
        rows.append({"turn": info["step"], "user felt": info["prev_emotion"],
                     "detector saw": EMOTIONS[int(np.argmax(current_emotion_probs(obs)))],
                     "assistant": ACTIONS[action], "user now": info["true_emotion"],
                     "reward": round(reward, 2), "valence": info["valence"]})
        obs, done = next_obs, terminated or truncated
    return pd.DataFrame(rows), total, info["persona"]


def playground(agent):
    st.subheader("Same user, different policies")
    st.caption("Every policy meets the same simulated user and starting emotion (same seed). "
               "The user's type is hidden from all policies except the oracle.")
    c1, c2, c3 = st.columns(3)
    persona = c1.selectbox("Hidden user type", ["random", *PERSONA_NAMES, "blended (unseen)"])
    accuracy = c2.slider("Emotion detector accuracy", 0.4, 1.0, 0.65, 0.05)
    seed = c3.number_input("Session seed", 0, 10_000, 7)
    if persona in PERSONAS:
        st.info(PERSONAS[persona].description)
    choices = ["PPO (ours)", "Rule table + cooldown", "v1 rule table", "Random",
               "PPO + online personalisation", "Greedy oracle (cheats)"]
    if agent is None:
        st.warning(f"No trained model at {model_path()}; showing baselines only.")
        choices = [c for c in choices if not c.startswith("PPO")]
    selected = st.multiselect("Policies", choices, default=choices[:3], max_selections=4)

    if not selected:
        return
    valence, totals, tables = {}, {}, {}
    for name in selected:
        df, total, user = run_session(make_policy(name, agent), persona, accuracy, int(seed),
                                       history_len(agent))
        valence[name], totals[name], tables[name] = df.set_index("turn")["valence"], total, df
    st.caption(f"Simulated user: **{user}**")
    cols = st.columns(len(selected))
    for col, name in zip(cols, selected):
        spoke = (tables[name]["assistant"] != ACTIONS[LISTEN]).mean()
        col.metric(name, f"{totals[name]:+.2f}", f"spoke {spoke:.0%} of turns", delta_color="off")
    st.markdown("**User's valence over the session** (higher = feeling better)")
    st.line_chart(pd.DataFrame(valence), color=SERIES_COLORS[:len(selected)])
    for name in selected:
        with st.expander(f"Turn-by-turn: {name}"):
            st.dataframe(tables[name].drop(columns="valence"), hide_index=True)


def photo_tab(agent):
    st.subheader("Read an expression and choose a response")
    image = st.camera_input("Take a photo") or st.file_uploader("…or upload one",
                                                                type=["jpg", "jpeg", "png"])
    if image is None:
        return
    try:
        import cv2

        from emotion_rl.detector import EmotionDetector
    except ImportError:
        st.error('Install the vision extras: pip install -e ".[vision]"')
        return
    frame = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    detector = EmotionDetector(smoothing=1.0, hysteresis=0.0)
    with st.spinner("Reading expression..."):
        det = detector.detect(frame)
    if not det.face_found:
        st.warning("No face found. Try better lighting or move closer.")
        return
    st.bar_chart(pd.DataFrame({"probability": det.probs}, index=list(EMOTIONS)),
                 color=SERIES_COLORS[0])
    builder = ObservationBuilder(history_len=history_len(agent), fixed_time_frac=0.5)
    obs = builder.push(det.probs)
    action = agent.act(obs) if agent else RuleTablePolicy().act(obs)
    if action == LISTEN:
        st.success(f"Detected **{det.emotion}** ({det.confidence:.0f}%). "
                   "The agent chooses to **listen** and stay quiet this turn.")
        return
    from emotion_rl.assistant.llm import ResponseGenerator

    text, source = ResponseGenerator().generate(det.emotion, det.confidence, ACTIONS[action])
    st.success(f"Detected **{det.emotion}** ({det.confidence:.0f}%) → action "
               f"**{ACTIONS[action]}**\n\n> {text}")
    st.caption(f"Reply source: {source}")


def main():
    st.set_page_config(page_title="Emotion-aware RL assistant", layout="wide")
    st.title("Emotion-aware RL assistant")
    st.write("A reinforcement-learning agent decides **whether** and **how** to respond to "
             "someone's emotional state. It was trained on a simulator with hidden user types.")
    agent = load_agent(model_path())
    tab1, tab2 = st.tabs(["Simulator playground", "Try a photo"])
    with tab1:
        playground(agent)
    with tab2:
        photo_tab(agent)


main()
