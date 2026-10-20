"""Lesson 5 — Regularization & overfitting.

Goal: diagnose overfitting, then apply regularization techniques and show
the train/val gap shrink.

This file is a scaffold. Data loading (minus augmentation) and the Lesson
3/4 model are filled in for you. The exercise is in four places, marked
TODO:
  1. an augmented train-time transform (RandomCrop + RandomHorizontalFlip)
  2. an optimizer with weight decay, plus an LR scheduler
  3. wiring the scheduler into the epoch loop
  4. picking a baseline vs. regularized config, running both, and plotting
     the comparison

Run with:  uv run python lessons/lesson5_regularization/regularization.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Known CIFAR-10 per-channel mean/std (computed in Lesson 2).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

BATCH_SIZE = 100
NUM_EPOCHS = 50  # bump this if your baseline hasn't visibly diverged yet
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")


# ---------------------------------------------------------------------------
# 1. Data — TODO: build an augmented train transform
# ---------------------------------------------------------------------------
# `test_transform` must stay un-augmented: you want to evaluate on the real
# distribution of images, not randomly perturbed ones (see Lesson 5
# checkpoint). `train_transform` is what you need to add augmentation to.
#
# Reminders:
#   - transforms.Compose runs its steps in order, and the geometric ones
#     (crop, flip) expect a PIL image, so they must come *before* ToTensor.
#   - transforms.RandomCrop(32, padding=4): pads the 32x32 image by 4px on
#     each side, then crops a random 32x32 window — the model never sees
#     exactly the same framing twice.
#   - transforms.RandomHorizontalFlip(): flips left-right with probability
#     0.5 by default.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
# TODO: train_transform = transforms.Compose([... augmentation ..., transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
# train_transform = test_transform


def make_loaders(train_transform: transforms.Compose) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 2. Model (Lesson 3/4 — already done, nothing to fill in here)
# ---------------------------------------------------------------------------
# Lesson 4's SimpleCNN, with batch_norm/dropout now toggleable so you can
# build a true unregularized baseline to diverge, and a regularized variant
# to compare it against.
class SimpleCNN(nn.Module):
    def __init__(self, use_batchnorm: bool = True, use_dropout: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=16) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
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


# ---------------------------------------------------------------------------
# 3. Optimizer + scheduler — TODO
# ---------------------------------------------------------------------------
# Reminders:
#   - weight_decay is a kwarg on most optimizers: optim.Adam(params, lr=...,
#     weight_decay=...). Try something small first, e.g. 1e-4 — too large
#     and you'll underfit instead.
#   - For a scheduler, torch.optim.lr_scheduler.StepLR(optimizer,
#     step_size=..., gamma=...) or CosineAnnealingLR(optimizer,
#     T_max=NUM_EPOCHS) are both reasonable choices here. A scheduler wraps
#     an already-created optimizer, so build the optimizer first.
def make_optimizer_and_scheduler(
    model: nn.Module, weight_decay: float, use_scheduler: bool
) -> tuple[optim.Optimizer, LRScheduler | None]:
    # DONE: optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    # DONE: scheduler = <StepLR or CosineAnnealingLR>(optimizer, ...) if use_scheduler else None
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# 4. Train/eval one epoch (Lesson 4 — already done, nothing to fill in here)
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
# 5. Run one full experiment — TODO
# ---------------------------------------------------------------------------
# Tie everything above together for one named configuration: build the data
# loaders (augmented or not), the model (with/without batchnorm & dropout),
# the optimizer/scheduler (with/without weight decay & LR schedule), then
# loop NUM_EPOCHS times calling train_one_epoch/evaluate, remembering to
# step the scheduler once per epoch (after the optimizer step, if present)
# and to record train/val loss+acc into a history dict like Lesson 4's.
def run_experiment(
    label: str,
    use_augmentation: bool,
    use_batchnorm: bool,
    use_dropout: bool,
    weight_decay: float,
    use_scheduler: bool,
) -> dict[str, list[float]]:
    # TODO:
    #  - train_loader, test_loader = make_loaders(train_transform if use_augmentation else test_transform)
    #  - model = SimpleCNN(use_batchnorm=use_batchnorm, use_dropout=use_dropout).to(device)
    #  - criterion = nn.CrossEntropyLoss()
    #  - optimizer, scheduler = make_optimizer_and_scheduler(model, weight_decay, use_scheduler)
    #  - history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    #  - for epoch in range(1, NUM_EPOCHS + 1):
    #      - train_one_epoch(...), evaluate(...), append to history
    #      - if scheduler is not None: scheduler.step()
    #      - print progress, labeled with `label` so interleaved runs are readable
    #  - return history
    train_loader, test_loader = make_loaders(train_transform if use_augmentation else test_transform)
    model = SimpleCNN(use_batchnorm=use_batchnorm, use_dropout=use_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = make_optimizer_and_scheduler(model, weight_decay, use_scheduler)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if scheduler is not None:
            scheduler.step()
        print(
            f"[{label}] epoch {epoch:>2}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
    return history


# ---------------------------------------------------------------------------
# 6. Compare a baseline vs. a regularized config — TODO
# ---------------------------------------------------------------------------
# Pick a baseline (everything off, so it actually overfits within
# NUM_EPOCHS) and a regularized config that turns on at least two of:
# batchnorm, dropout, augmentation, weight decay. Run both, then plot train
# vs. val curves for each on the same axes so the gap is directly
# comparable. (Hint: reuse Lesson 4's two-subplot loss/accuracy figure,
# plotting 4 lines per subplot instead of 2 — label each by config + split.)
if __name__ == "__main__":
    # DONE: baseline_history = run_experiment("baseline", use_augmentation=False, use_batchnorm=False, use_dropout=False, weight_decay=0.0, use_scheduler=False)
    baseline_history = run_experiment(
        "baseline",
        use_augmentation=False,
        use_batchnorm=False,
        use_dropout=False,
        weight_decay=0.0,
        use_scheduler=False,
    )
    # DONE: regularized_history = run_experiment("regularized", use_augmentation=True, use_batchnorm=True, use_dropout=True, weight_decay=1e-4, use_scheduler=True)
    regularized_history = run_experiment(
        "regularized",
        use_augmentation=True,
        use_batchnorm=True,
        use_dropout=True,
        weight_decay=1e-4,
        use_scheduler=True,
    )
    # DONE: plot both histories' train/val loss and accuracy curves together
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(baseline_history["train_loss"], label="baseline train")
    ax_loss.plot(baseline_history["val_loss"], label="baseline val")
    ax_loss.plot(regularized_history["train_loss"], label="regularized train")
    ax_loss.plot(regularized_history["val_loss"], label="regularized val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(baseline_history["train_acc"], label="baseline train")
    ax_acc.plot(baseline_history["val_acc"], label="baseline val")
    ax_acc.plot(regularized_history["train_acc"], label="regularized train")
    ax_acc.plot(regularized_history["val_acc"], label="regularized val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.show()
