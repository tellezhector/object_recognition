"""Lesson 4 — Training loop.

Goal: write (not import) a full training loop for your Lesson 3 CNN —
forward pass, loss, backward pass, optimizer step, plus a no-grad validation
pass each epoch.

This file is a scaffold. Data loading, the model, and the epoch/plotting glue
are filled in for you (you already built those in Lessons 2-3) so you can
focus on the actual exercise: train_one_epoch and evaluate below, where
you'll find TODOs.
Run with:  uv run python lessons/lesson4_training_loop/training_loop.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Known CIFAR-10 per-channel mean/std (computed in Lesson 2).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

BATCH_SIZE = 100
NUM_EPOCHS = 50
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
# 2. Model (Lesson 3 — already done, nothing to fill in here)
# ---------------------------------------------------------------------------
# This is Lesson 3's SimpleCNN. If your own architecture differs, swap it in here.
class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm1(self.conv1(x))
        x = self.pool1(self.relu(x))
        x = self.batch_norm2(self.conv2(x))
        x = self.pool2(self.relu(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)


# ---------------------------------------------------------------------------
# 3. Loss and optimizer
# ---------------------------------------------------------------------------
# CrossEntropyLoss expects raw logits (no softmax applied by you) and
# integer class labels. Try SGD first, then compare against Adam once your
# loop works end to end.

# DONE: criterion = nn.CrossEntropyLoss()
# DONE: optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) or optim.Adam(...)
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------------------------
# 4. One epoch of training — THIS IS THE EXERCISE
# ---------------------------------------------------------------------------
# model.train() puts layers like Dropout/BatchNorm (if you have them) into
# their training-time behavior — see the Lesson 4 checkpoint in COURSE.md.


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch over `loader`. Returns (avg_loss, accuracy)."""
    model.train()
    # DONE:
    #  - keep running totals: total_loss, total_correct, total_samples
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    #  - for each (images, labels) batch in loader:
    #      - move images, labels to `device`
    #      - optimizer.zero_grad()
    #      - logits = model(images)
    #      - loss = criterion(logits, labels)
    #      - loss.backward()
    #      - optimizer.step()
    #      - accumulate: total_loss += loss.item() * images.size(0)
    #                    predictions = logits.argmax(dim=1)
    #                    total_correct += (predictions == labels).sum().item()
    #                    total_samples += images.size(0)
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
    #  - return total_loss / total_samples, total_correct / total_samples
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# 5. Validation pass — also the exercise
# ---------------------------------------------------------------------------
# No backward pass, no optimizer step — you're only measuring, not learning.
# model.eval() + torch.no_grad() together: no_grad skips building the graph
# (saves memory/compute since you won't call .backward()); eval() switches
# any train/eval-sensitive layers to their inference behavior.


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one no-grad pass over `loader`. Returns (avg_loss, accuracy)."""
    model.eval()
    # DONE: same running-totals/accumulation pattern as train_one_epoch, minus
    # zero_grad/backward/step — just forward pass, loss, and bookkeeping.
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
# 6. The epoch loop (glue — already done, nothing to fill in here)
# ---------------------------------------------------------------------------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

if __name__ == "__main__":
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch:>2}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # ---------------------------------------------------------------------
    # 7. Plot train vs val curves (already done, nothing to fill in here)
    # ---------------------------------------------------------------------
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(history["train_loss"], label="train")
    ax_loss.plot(history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(history["train_acc"], label="train")
    ax_acc.plot(history["val_acc"], label="val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.show()
