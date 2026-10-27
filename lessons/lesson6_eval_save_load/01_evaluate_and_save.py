"""Lesson 6 — Evaluation & save/load (part 1 of 2).

Goal: train your CIFAR-10 CNN, properly evaluate it with a confusion matrix
(not just accuracy), and save it so it can be reused later without retraining.

This file is a scaffold. Data loading, the model, and the train/eval loop are
filled in for you (Lessons 2-4) so you can focus on what's new here. The
exercise is in three places, marked TODO (DONE once addressed):
  1. compute_confusion_matrix — tally predicted vs. true labels on the test set
  2. most_confused_pair — find which two classes get mixed up most often
  3. saving the trained model's state_dict to CHECKPOINT_PATH

Part 2 is lessons/lesson6_eval_save_load/classify_image.py — a SEPARATE script
that loads the checkpoint this file produces and classifies one image. Run
this file first.

Run with:  uv run python lessons/lesson6_eval_save_load/evaluate_and_save.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Known CIFAR-10 per-channel mean/std (computed in Lesson 2).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

BATCH_SIZE = 100
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")


# ---------------------------------------------------------------------------
# 1. Data (Lesson 2 — already done, nothing to fill in here)
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ---------------------------------------------------------------------------
# 2. Model (Lessons 3/4 — already done, nothing to fill in here)
# ---------------------------------------------------------------------------
# This must match EXACTLY what classify_image.py reconstructs — same class,
# same layer shapes — since a state_dict is just a name->tensor mapping, with
# no architecture info attached.
class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=16)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout_3 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm1(self.conv1(x))
        x = self.dropout_1(x)
        x = self.pool1(self.relu(x))
        x = self.batch_norm2(self.conv2(x))
        x = self.dropout_2(x)
        x = self.pool2(self.relu(x))
        x = self.flatten(x)
        x = self.dropout_3(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------------
# 3. Train/eval one epoch (Lesson 4 — already done, nothing to fill in here)
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# 4. Confusion matrix — DONE (this is the first exercise)
# ---------------------------------------------------------------------------
# A confusion matrix is a num_classes x num_classes grid where entry
# cm[true_label][predicted_label] counts how many test images with that true
# label got that prediction. A perfect model has all its mass on the
# diagonal (true == predicted); everything off-diagonal is a specific kind
# of mistake.
#
# Reminders:
#   - You already know how to get predictions from a batch: model.eval(),
#     torch.no_grad(), logits = model(images), predictions = logits.argmax(dim=1)
#     (see `evaluate` above).
#   - Build an all-zero (num_classes, num_classes) tensor or numpy array up
#     front, e.g. torch.zeros(num_classes, num_classes, dtype=torch.long).
#   - For each batch, for each (true_label, predicted_label) pair, increment
#     cm[true_label, predicted_label] by 1. You can do this with a plain
#     Python loop over the batch, or look up how to do it batch-at-once with
#     torch.bincount on a combined index (true * num_classes + predicted).
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    """Return a (num_classes, num_classes) tensor; cm[true, pred] = count."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            for true_label, predicted_label in zip(labels, predictions, strict=True):
                cm[true_label, predicted_label] += 1
    return cm


def plot_confusion_matrix(cm: torch.Tensor, classes: tuple[str, ...]) -> None:
    """Already done — just visualizes whatever `cm` you computed above."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm.cpu().numpy(), cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Most-confused class pair — TODO (second exercise)
# ---------------------------------------------------------------------------
# Reminders:
#   - You want the largest OFF-diagonal entry — the diagonal is always
#     correct predictions, not confusion, so exclude it (e.g. zero out the
#     diagonal on a copy before searching for the max).
#   - cm is not symmetric: cm[i, j] (true=i, predicted=j) and cm[j, i]
#     (true=j, predicted=i) are different counts. Decide whether you care
#     about direction or want to consider cm[i,j] + cm[j,i] together as "how
#     often these two classes get mixed up" — either is a defensible answer,
#     just be clear about which you're reporting.
def most_confused_pair(cm: torch.Tensor, classes: tuple[str, ...]) -> tuple[str, str, int]:
    """Return (true_class, predicted_class, count) for the top confusion."""
    cm_copy = cm.clone()
    cm_copy.fill_diagonal_(0)
    # we went for the overall largest off-diagonal entry because probably trying
    # to fix one direction of confusion does not necessarily fix the other
    # direction, so it's worth knowing which pair is the worst.
    max_val, max_idx = cm_copy.max(), cm_copy.argmax()
    true_class = classes[max_idx // len(classes)]
    predicted_class = classes[max_idx % len(classes)]
    return true_class, predicted_class, int(max_val)


# ---------------------------------------------------------------------------
# 6. Train, evaluate, save — TODO for the save step only
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"epoch {epoch:>3}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    cm = compute_confusion_matrix(model, test_loader, device, num_classes=len(CLASSES))

    true_cls, pred_cls, count = most_confused_pair(cm, CLASSES)
    print(f"Most confused pair: true={true_cls!r} predicted={pred_cls!r} ({count} times)")

    # DONE: save model.state_dict() to CHECKPOINT_PATH with torch.save.
    # Reminder: save model.state_dict(), not `model` itself — see the Lesson
    # 6 checkpoint question in COURSE.md about why.
    print(f"Saving checkpoint to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    plot_confusion_matrix(cm, CLASSES)
