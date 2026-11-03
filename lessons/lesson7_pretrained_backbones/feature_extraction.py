"""Lesson 7 — Pretrained backbones (feature extraction).

Goal: load a ResNet pretrained on ImageNet, freeze everything, replace the
final classification layer, and train only that new layer on Imagenette.

This file is a scaffold. Sections marked "already done" reuse things from
earlier lessons (data loading from Lesson 2, the train/eval loop from Lesson
4/6) so you can focus on what's new here. Sections marked TODO are this
lesson's actual exercise — fill in the function bodies (delete the
`raise NotImplementedError` once you do).

Run with:  uv run python lessons/lesson7_pretrained_backbones/feature_extraction.py
"""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Imagenette downloads here, same convention as Lesson 2's CIFAR-10.
# "320px" is the fastai-resized variant -- plenty of detail for a 224x224
# ResNet input, much smaller/faster to download and decode than "full".
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMAGENETTE_SIZE = "320px"

# A pretrained ResNet's early layers only ever saw inputs preprocessed this
# exact way during its ImageNet training -- a fixed 224x224 crop, pixel
# values normalized with these specific per-channel stats. Feed it something
# normalized differently (e.g. your CIFAR-10 stats from Lesson 2) and its
# learned filters are looking at inputs shaped unlike anything they trained
# on. These constants are a property of the pretrained weights, not
# something to (re)derive yourself.
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")


# ---------------------------------------------------------------------------
# 1. Dataset & transforms -- already done (same pattern as Lesson 2, just
#    pointed at Imagenette with ImageNet-pretrained-model preprocessing)
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = datasets.Imagenette(
    root=DATA_DIR, split="train", size=IMAGENETTE_SIZE, download=True, transform=transform
)
val_dataset = datasets.Imagenette(root=DATA_DIR, split="val", size=IMAGENETTE_SIZE, download=True, transform=transform)

CLASSES = train_dataset.classes  # tuple of synonym-tuples, one per class
NUM_CLASSES = len(CLASSES)
print(f"{NUM_CLASSES=}, classes={CLASSES}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ---------------------------------------------------------------------------
# 2. Load a pretrained backbone -- TODO
# ---------------------------------------------------------------------------
def build_pretrained_backbone() -> nn.Module:
    """Return a torchvision ResNet loaded with its ImageNet-pretrained weights.

    Hints:
      - `torchvision.models.resnet18` is the cheapest to iterate with on 8GB
        VRAM; `resnet34`/`resnet50` are there if you want to compare later.
      - The current API takes a `weights=` argument (an enum, e.g.
        `torchvision.models.ResNet18_Weights.DEFAULT`), not the old bare
        `pretrained=True`.
      - Afterwards, `print(model)` to see the full layer stack. Find the name
        of the final layer and read off its `in_features`/`out_features` --
        you'll need both in the next section.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Freeze every parameter -- TODO
# ---------------------------------------------------------------------------
def freeze_all_parameters(model: nn.Module) -> None:
    """Set requires_grad=False on every parameter in `model`, in place.

    Hint: `model.parameters()` yields every nn.Parameter in the whole module
    tree, including nested submodules -- a single for loop is all you need.
    This is the same `requires_grad` flag from Lesson 1, just flipped off
    instead of on.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Replace the classification head -- TODO
# ---------------------------------------------------------------------------
def replace_classifier_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap the final FC layer for a fresh nn.Linear with `num_classes` outputs.

    Hints:
      - For torchvision's ResNet family the final layer is the `fc`
        attribute, an `nn.Linear`. Its current `out_features` is 1000
        (ImageNet classes) -- wrong shape for Imagenette's 10.
      - Construct a new `nn.Linear(in_features, num_classes)` (reuse the old
        layer's `in_features`) and assign it over `model.fc`.
      - A freshly constructed `nn.Linear` has `requires_grad=True` by
        default and random weights -- it's the one piece of the network
        that both needs training and hasn't been frozen by step 3, as long
        as you replace it *after* freezing.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 5. Train/eval one epoch -- already done (identical to Lesson 4/6)
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
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
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
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# 6. Optimizer -- TODO
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    """Build an Adam optimizer that updates only the head's parameters.

    Why this matters: after steps 3-4, most of `model`'s parameters have
    `requires_grad=False`. Handing `model.parameters()` to the optimizer
    would still "work" (Adam just has nothing to step for the frozen ones),
    but it's clearer -- and cheaper, since Adam keeps two running-average
    buffers per parameter it tracks -- to only hand it what's actually
    trainable.

    Two ways to do this:
      - `model.fc.parameters()`, if you kept the attribute name from step 4.
      - `filter(lambda p: p.requires_grad, model.parameters())`, which finds
        the trainable ones regardless of attribute name -- handy if you
        change which layers are frozen later (Lesson 8).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 7. Put it together -- already done, calls everything above
# ---------------------------------------------------------------------------
model = build_pretrained_backbone()
freeze_all_parameters(model)
model = replace_classifier_head(model, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = build_optimizer(model, LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(
        f"epoch {epoch:>2}/{NUM_EPOCHS}  "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )
