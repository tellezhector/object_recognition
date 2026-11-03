"""Lesson 6 — Evaluation & save/load (extra: inspect a saved model).

Loads the checkpoint produced by 01_evaluate_and_save.py into a fresh model
(no training here) and displays its confusion matrix on the CIFAR-10 test
set, plus the most-confused class pair.

Run 01_evaluate_and_save.py first so checkpoint.pt exists, then:
  uv run python lessons/lesson6_eval_save_load/03_confusion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
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


# ---------------------------------------------------------------------------
# Model — must be an EXACT architecture match with 01_evaluate_and_save.py,
# the script that produced checkpoint.pt.
# ---------------------------------------------------------------------------
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


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = SimpleCNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    """Return a (num_classes, num_classes) tensor; cm[true, pred] = count."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        for true_label, predicted_label in zip(labels, predictions, strict=True):
            cm[true_label, predicted_label] += 1
    return cm


def most_confused_pair(cm: torch.Tensor, classes: tuple[str, ...]) -> tuple[str, str, int]:
    """Return (true_class, predicted_class, count) for the top confusion."""
    cm_copy = cm.clone()
    cm_copy.fill_diagonal_(0)
    max_val, max_idx = cm_copy.max(), cm_copy.argmax()
    true_class = classes[max_idx // len(classes)]
    predicted_class = classes[max_idx % len(classes)]
    return true_class, predicted_class, int(max_val)


def plot_confusion_matrix(cm: torch.Tensor, classes: tuple[str, ...]) -> None:
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = load_model(CHECKPOINT_PATH, device)

    cm = compute_confusion_matrix(model, test_loader, device, num_classes=len(CLASSES))
    true_cls, pred_cls, count = most_confused_pair(cm, CLASSES)
    print(f"Most confused pair: true={true_cls!r} predicted={pred_cls!r} ({count} times)")

    plot_confusion_matrix(cm, CLASSES)
