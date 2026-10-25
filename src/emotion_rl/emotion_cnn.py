"""Train your own facial-expression CNN on FER2013 (so the vision part is your work too).

Dataset layout (Kaggle "msambare/fer2013"):
    data/fer2013/train/<emotion>/*.jpg
    data/fer2013/test/<emotion>/*.jpg

    python -m emotion_rl.emotion_cnn --data data/fer2013 --epochs 40
    python -m emotion_rl.detector_eval --custom-model models/emotion_cnn.pt  # vs DeepFace
    python -m emotion_rl.assistant.app --emotion-model models/emotion_cnn.pt  # use it live

A GPU makes training take minutes instead of hours; CPU works for small runs.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from emotion_rl.emotions import EMOTION_TO_INDEX, EMOTIONS, NUM_EMOTIONS

IMAGE_SIZE = 48


# ─── Data ──────────────────────────────────────────────────────────────────────
def load_folder(root, max_per_class=None):
    """Loads <root>/<emotion>/* into uint8 arrays (N, 48, 48) and labels in our index order."""
    from PIL import Image

    images, labels = [], []
    for folder in sorted(os.listdir(root)):
        label = EMOTION_TO_INDEX.get(folder.lower())
        path = os.path.join(root, folder)
        if label is None or not os.path.isdir(path):
            continue
        files = sorted(os.listdir(path))[:max_per_class]
        for name in files:
            with Image.open(os.path.join(path, name)) as img:
                img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
                images.append(np.asarray(img, dtype=np.uint8))
                labels.append(label)
    if not images:
        raise FileNotFoundError(f"No emotion folders with images found in {root}")
    return np.stack(images), np.array(labels, dtype=np.int64)


def augment(batch, rng):
    """Random horizontal flip and up-to-4-pixel shifts, on a (B, 1, 48, 48) tensor."""
    flip = torch.from_numpy(rng.random(batch.shape[0]) < 0.5)
    batch = batch.clone()
    batch[flip] = batch[flip].flip(-1)
    padded = nn.functional.pad(batch, (4, 4, 4, 4), mode="reflect")
    out = torch.empty_like(batch)
    for i, (dx, dy) in enumerate(rng.integers(0, 9, size=(batch.shape[0], 2))):
        out[i] = padded[i, :, dy:dy + IMAGE_SIZE, dx:dx + IMAGE_SIZE]
    return out


def to_tensor(images):
    return torch.from_numpy(images).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(1)


# ─── Model ─────────────────────────────────────────────────────────────────────
def _block(c_in, c_out, dropout):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout(dropout),
    )


class EmotionNet(nn.Module):
    """VGG-style CNN: 3 conv blocks (64-128-256) then a small classifier head."""

    def __init__(self, width=64, n_classes=NUM_EMOTIONS):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, width, 0.1), _block(width, width * 2, 0.2), _block(width * 2, width * 4, 0.3))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class EmotionCNN:
    """Inference wrapper usable as EmotionDetector(classifier=...)."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = EmotionNet(width=checkpoint.get("width", 64))
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, device)

    def predict_proba(self, face_bgr):
        import cv2

        gray = face_bgr if face_bgr.ndim == 2 else cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        return self.predict_batch(gray[None])[0]

    @torch.no_grad()
    def predict_batch(self, images_uint8):
        x = to_tensor(np.asarray(images_uint8, dtype=np.uint8)).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()


# ─── Training ──────────────────────────────────────────────────────────────────
def evaluate_model(model, images, labels, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = to_tensor(images[i:i + batch_size]).to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train(data_dir, out_path, epochs=40, batch_size=128, lr=1e-3, width=64, val_fraction=0.1,
          max_per_class=None, seed=0, device=None):
    from sklearn.metrics import f1_score

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    images, labels = load_folder(os.path.join(data_dir, "train"), max_per_class)
    order = rng.permutation(len(images))
    n_val = max(int(len(images) * val_fraction), 1)
    val_idx, train_idx = order[:n_val], order[n_val:]
    test_dir = os.path.join(data_dir, "test")
    test = load_folder(test_dir, max_per_class) if os.path.isdir(test_dir) else None
    print(f"[CNN] train={len(train_idx)} val={len(val_idx)} "
          f"test={0 if test is None else len(test[0])} device={device}")

    # Class weights counter FER2013's imbalance (disgust is ~1.5% of images)
    counts = np.bincount(labels[train_idx], minlength=NUM_EMOTIONS).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1) / NUM_EMOTIONS
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device),
                                    label_smoothing=0.1)

    model = EmotionNet(width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=steps_per_epoch)
    best_val, history = -1.0, []
    for epoch in range(epochs):
        model.train()
        start, total_loss = time.time(), 0.0
        shuffled = rng.permutation(train_idx)
        for i in range(0, len(shuffled), batch_size):
            idx = shuffled[i:i + batch_size]
            if len(idx) < 2:  # BatchNorm needs at least two samples
                continue
            x = augment(to_tensor(images[idx]), rng).to(device)
            y = torch.from_numpy(labels[idx]).to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        val_pred = evaluate_model(model, images[val_idx], labels[val_idx], device)
        val_acc = float((val_pred == labels[val_idx]).mean())
        history.append({"epoch": epoch + 1, "loss": total_loss / steps_per_epoch,
                        "val_acc": val_acc})
        print(f"[CNN] epoch {epoch + 1:3d}/{epochs} loss {history[-1]['loss']:.3f} "
              f"val_acc {val_acc:.3f} ({time.time() - start:.0f}s)")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "width": width,
                        "classes": list(EMOTIONS)}, out_path)

    report = {"best_val_acc": best_val, "history": history}
    if test is not None:
        best = EmotionCNN.load(out_path, device)
        pred = evaluate_model(best.model, test[0], test[1], device)
        report["test_acc"] = float((pred == test[1]).mean())
        report["test_macro_f1"] = float(f1_score(test[1], pred, average="macro"))
        print(f"[CNN] test accuracy {report['test_acc']:.3f}, "
              f"macro-F1 {report['test_macro_f1']:.3f}")
    with open(os.path.splitext(out_path)[0] + "_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a facial-expression CNN on FER2013")
    parser.add_argument("--data", default="data/fer2013")
    parser.add_argument("--out", default="models/emotion_cnn.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          width=args.width, max_per_class=args.max_per_class, seed=args.seed)


if __name__ == "__main__":
    main()
