"""
evaluate_metrics.py
═══════════════════════════════════════════════════════════════════════════════
Computes two sets of metrics for the Emotion-Aware RL Assistant project:

  PART 1 — PPO AGENT DECISION METRICS  (your own trained model)
           Treats the agent's action selection as a classification task:
           "Did the PPO agent pick the OPTIMAL action for each emotion?"
           Outputs: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

  PART 2 — DEEPFACE EMOTION DETECTION METRICS  (pre-trained model)
           If FER2013 test set exists in data/ → computes REAL accuracy.
           If not → reports DeepFace's published benchmark for reference.

Run:  python evaluate_metrics.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from stable_baselines3 import PPO

from emotion_detector import EMOTIONS_TO_INDEX, INDEX_TO_EMOTION, NUM_EMOTIONS
from rl_environment import ACTIONS, BEST_ACTION_FOR_EMOTION

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "model/ppo_agent/best_model"
FER_TEST_DIR     = "data/fer2013/test"     # optional
N_SAMPLES        = 1000                     # samples per emotion for PPO eval
OUTPUT_DIR       = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — PPO AGENT DECISION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_ppo_decisions():
    print("=" * 65)
    print("  PART 1 — PPO AGENT DECISION METRICS")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"[ERROR] PPO model not found at {MODEL_PATH}.zip")
        print("        Train first: python train_ppo.py --mode train")
        return None

    model = PPO.load(MODEL_PATH)
    print(f"[Loaded] PPO model from {MODEL_PATH}.zip\n")

    y_true = []   # optimal action (ground truth)
    y_pred = []   # action chosen by PPO agent

    # For each emotion, feed the one-hot state and record agent's action
    for emotion_idx in range(NUM_EMOTIONS):
        optimal_action = BEST_ACTION_FOR_EMOTION.get(emotion_idx, 2)

        # Build one-hot state vector for this emotion
        state              = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        state[emotion_idx] = 1.0

        # Query the agent N times (deterministic = same answer, but we
        # also test stochastic to capture policy distribution)
        for _ in range(N_SAMPLES):
            action, _ = model.predict(state, deterministic=False)
            y_true.append(optimal_action)
            y_pred.append(int(action))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Compute metrics ──────────────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("─" * 65)
    print(f"  PPO Agent Accuracy   : {acc * 100:.2f}%")
    print(f"  PPO Agent Precision  : {precision * 100:.2f}%")
    print(f"  PPO Agent Recall     : {recall * 100:.2f}%")
    print(f"  PPO Agent F1-Score   : {f1 * 100:.2f}%")
    print("─" * 65)

    # ── Per-emotion breakdown ────────────────────────────────────────────────
    print("\n  Per-Emotion Optimal-Action Selection Rate:")
    print("  " + "-" * 55)
    for emotion_idx in range(NUM_EMOTIONS):
        optimal = BEST_ACTION_FOR_EMOTION.get(emotion_idx, 2)
        mask    = (y_true == optimal)
        # accuracy for samples belonging to this emotion's optimal action
        emo_name = INDEX_TO_EMOTION[emotion_idx]
        state    = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        state[emotion_idx] = 1.0
        det_action, _ = model.predict(state, deterministic=True)
        det_action = int(det_action)
        correct = "✓" if det_action == optimal else "✗"
        print(f"  {emo_name:10s} → optimal: {ACTIONS[optimal]:14s} | "
              f"agent: {ACTIONS[det_action]:14s} {correct}")
    print()

    # ── Action-level classification report ───────────────────────────────────
    action_names = [ACTIONS[i] for i in range(len(ACTIONS))]
    print("  Action-Level Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(ACTIONS))),
        target_names=action_names,
        zero_division=0
    ))

    # ── Confusion matrix plot ────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ACTIONS))))
    plot_confusion(cm, action_names,
                   "PPO Agent: Optimal vs Selected Action",
                   os.path.join(OUTPUT_DIR, "ppo_confusion_matrix.png"))

    return {
        'accuracy':  acc,
        'precision': precision,
        'recall':    recall,
        'f1':        f1
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — DEEPFACE EMOTION DETECTION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_deepface(max_per_class=100):
    print("\n" + "=" * 65)
    print("  PART 2 — DEEPFACE EMOTION DETECTION METRICS")
    print("=" * 65)

    # Check if FER2013 test set is available
    if not os.path.isdir(FER_TEST_DIR):
        print(f"[INFO] FER2013 test set not found at '{FER_TEST_DIR}'")
        print("[INFO] Reporting DeepFace published benchmark instead.\n")
        print("─" * 65)
        print("  DeepFace FER2013 Published Benchmark (reference only):")
        print("  • Overall Accuracy : ~63-67%")
        print("  • Note: DeepFace is PRE-TRAINED — not trained in this project")
        print("  • Source: Serengil & Ozpinar (2020), LightFace framework")
        print("─" * 65)
        print("\n  To compute REAL metrics, download FER2013 to data/fer2013/")
        print("  https://www.kaggle.com/datasets/msambare/fer2013")
        return None

    # If dataset exists, run real evaluation
    from deepface import DeepFace
    import cv2

    print(f"[INFO] Found FER2013 test set. Evaluating (max {max_per_class}/class)...\n")

    y_true = []
    y_pred = []

    emotion_folders = [d for d in os.listdir(FER_TEST_DIR)
                       if os.path.isdir(os.path.join(FER_TEST_DIR, d))]

    for folder in emotion_folders:
        true_idx = EMOTIONS_TO_INDEX.get(folder.lower(), None)
        if true_idx is None:
            continue

        folder_path = os.path.join(FER_TEST_DIR, folder)
        images = os.listdir(folder_path)[:max_per_class]

        print(f"  Processing '{folder}' ({len(images)} images)...")

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                result = DeepFace.analyze(
                    img_path, actions=['emotion'],
                    enforce_detection=False, silent=True
                )
                pred_emotion = result[0]['dominant_emotion']
                pred_idx     = EMOTIONS_TO_INDEX.get(pred_emotion, 6)
                y_true.append(true_idx)
                y_pred.append(pred_idx)
            except Exception:
                continue

    if not y_true:
        print("[ERROR] No images could be processed.")
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Compute metrics ──────────────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "─" * 65)
    print(f"  DeepFace Accuracy   : {acc * 100:.2f}%")
    print(f"  DeepFace Precision  : {precision * 100:.2f}%")
    print(f"  DeepFace Recall     : {recall * 100:.2f}%")
    print(f"  DeepFace F1-Score   : {f1 * 100:.2f}%")
    print("─" * 65)
    print("  Note: DeepFace is PRE-TRAINED (credit: Serengil & Ozpinar 2020)\n")

    emotion_names = [INDEX_TO_EMOTION[i] for i in range(NUM_EMOTIONS)]
    print("  Per-Emotion Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(NUM_EMOTIONS)),
        target_names=emotion_names,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_EMOTIONS)))
    plot_confusion(cm, emotion_names,
                   "DeepFace Emotion Detection (FER2013 Test Set)",
                   os.path.join(OUTPUT_DIR, "deepface_confusion_matrix.png"))

    return {
        'accuracy':  acc,
        'precision': precision,
        'recall':    recall,
        'f1':        f1
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — Confusion Matrix Plot
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion(cm, labels, title, save_path):
    # Normalize to percentages per row
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True / Optimal')
    ax.set_title(title, fontweight='bold')

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val   = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f"{val*100:.0f}%",
                    ha='center', va='center', color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [Saved] Confusion matrix → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 63 + "╗")
    print("║      EMOTION-AWARE RL ASSISTANT — METRICS EVALUATION          ║")
    print("╚" + "═" * 63 + "╝")
    print()

    ppo_metrics      = evaluate_ppo_decisions()
    deepface_metrics = evaluate_deepface(max_per_class=100)

    # ── Final Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)

    if ppo_metrics:
        print(f"\n  [PPO Agent - Decision Quality]  (your trained model)")
        print(f"    Accuracy : {ppo_metrics['accuracy']*100:.2f}%")
        print(f"    Precision: {ppo_metrics['precision']*100:.2f}%")
        print(f"    Recall   : {ppo_metrics['recall']*100:.2f}%")
        print(f"    F1-Score : {ppo_metrics['f1']*100:.2f}%")

    if deepface_metrics:
        print(f"\n  [DeepFace - Emotion Detection]  (pre-trained, credited)")
        print(f"    Accuracy : {deepface_metrics['accuracy']*100:.2f}%")
        print(f"    Precision: {deepface_metrics['precision']*100:.2f}%")
        print(f"    Recall   : {deepface_metrics['recall']*100:.2f}%")
        print(f"    F1-Score : {deepface_metrics['f1']*100:.2f}%")
    else:
        print(f"\n  [DeepFace] Published benchmark: ~63-67% (FER2013)")

    print("\n" + "=" * 65)
    print("  [Done] Metrics evaluation complete!")
    print("=" * 65)
