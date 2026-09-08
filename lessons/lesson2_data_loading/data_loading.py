"""Lesson 2 — Data loading.

Goal: get CIFAR-10 flowing through a DataLoader, and prove it works by
inspecting one batch and eyeballing a few images.

This file is a scaffold. The section headers and comments lay out the steps;
the actual torchvision/torch calls are yours to write where you see TODO.
Run with:  uv run python lessons/lesson2_data_loading/data_loading.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 downloads here. `data/` already exists in the repo and is gitignored,
# so the dataset won't get committed.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# The 10 classes, in the label-index order torchvision uses. label 0 -> "airplane",
# label 1 -> "automobile", etc. Reference data, not something you compute.
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# 1. Normalization stats
# ---------------------------------------------------------------------------
# Normalizing means: for each colour channel, subtract the mean and divide by
# the std so the values are roughly centered on 0 with spread ~1. The network
# trains better when its inputs are on a consistent, small scale.
#
# The catch: to compute mean/std you first need the data loaded *without*
# Normalize in the transform (otherwise you're measuring already-normalized
# numbers). So this is a two-pass thing:
#   pass 1 -> load with just ToTensor, iterate the training set, accumulate stats
#   pass 2 -> rebuild the transform WITH Normalize(mean, std), reload
#
# For a first run you can skip pass 1 and use these well-known CIFAR-10 values,
# then come back and compute them yourself to check they match:
#   mean = (0.4914, 0.4822, 0.4465)
#   std  = (0.2470, 0.2435, 0.2616)

# TODO: compute these from the TRAINING set only (see checkpoint question), or
#       start with the known constants above.
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())

loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


sum_per_channel = torch.zeros(3)

# image_num = len(train_dataset)
# print(f"{image_num=}") # expect 50k
# we will redefine image_num temporarily to work with smaller samples
image_num = 0

n_batches = len(loader)  # ceil(len(train_dataset) / BATCH_SIZE)

print(f"pass 1/2: mean over {len(train_dataset)} images, {n_batches} batches")
for i, (images, _labels) in enumerate(loader, start=1):
    image_num += images.shape[0]
    sum_per_channel += images.sum(dim=(0, 2, 3))
    if i % 50 == 0 or i == n_batches:
        print(f"  batch {i:>4}/{n_batches}  ({image_num} images)")

n_pixels = image_num * 32 * 32  # 50k * 32 * 32
mean = sum_per_channel / n_pixels
print(f"{mean=}")  # expect (0.4914, 0.4822, 0.4465)


# sum of squared deviations from the mean, per channel
sq_dev_sum = torch.zeros(3)
print(f"pass 2/2: std over {len(train_dataset)} images, {n_batches} batches")
for i, (images, _labels) in enumerate(loader, start=1):
    # mean is (3,); view it as (3, 1, 1) so it broadcasts against each
    # (3, 32, 32) image: channel lines up, height/width are stretched.
    sq_dev_sum += ((images - mean.view(3, 1, 1)) ** 2).sum(dim=(0, 2, 3))
    if i % 50 == 0 or i == n_batches:
        print(f"  batch {i:>4}/{n_batches}")

# variance = mean of squared deviations; std = its square root
std = torch.sqrt(sq_dev_sum / n_pixels)
print(f"{std=}")  # expect (0.2470, 0.2435, 0.2616) on the full set


# ---------------------------------------------------------------------------
# 2. Transforms
# ---------------------------------------------------------------------------
# A transform is a function applied to each image as it's pulled from the
# dataset. transforms.Compose chains several into one. At minimum you need to
# turn the PIL image into a tensor; then normalize it.
#
# ToTensor() also rescales pixel values from 0-255 ints to 0.0-1.0 floats and
# reorders dimensions to (channels, height, width) — worth knowing for when you
# un-normalize later.

# TODO: build train_transform and test_transform with transforms.Compose([...]).
#       For this lesson they can be identical. (In Lesson 5 the train one grows
#       augmentation and they diverge.)
train_transform = ...
test_transform = ...


# ---------------------------------------------------------------------------
# 3. Datasets
# ---------------------------------------------------------------------------
# torchvision.datasets.CIFAR10 gives you a Dataset object: something you can
# index (dataset[i] -> (image, label)) and take len() of. `train=True` selects
# the 50k training split, `train=False` the 10k test split — CIFAR-10 ships
# with that split predefined, you don't make it yourself.

# TODO: create train_dataset and test_dataset.
#       datasets.CIFAR10(root=DATA_DIR, train=..., download=True, transform=...)
train_dataset = ...
test_dataset = ...


# ---------------------------------------------------------------------------
# 4. DataLoaders
# ---------------------------------------------------------------------------
# A Dataset gives you one example at a time. A DataLoader wraps it and hands you
# *batches* of stacked tensors, optionally shuffled, optionally loaded in
# parallel worker processes. The training loop in Lesson 4 iterates this.
#
# Convention: shuffle the training loader (so batch composition changes each
# epoch), don't shuffle the test loader (order doesn't matter for evaluation
# and un-shuffled is reproducible).

# TODO: create train_loader and test_loader.
#       DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=..., num_workers=2)
train_loader = ...
test_loader = ...


# ---------------------------------------------------------------------------
# 5. Inspect one batch
# ---------------------------------------------------------------------------
# Pull a single batch and check it's shaped the way you expect. `iter()` +
# `next()` gets you the first batch without a full loop.

# TODO: grab one batch of (images, labels) from train_loader.
# TODO: print images.shape  -> expect (BATCH_SIZE, 3, 32, 32)
# TODO: print images.dtype  -> expect torch.float32
# TODO: print labels.shape and a few label values


# ---------------------------------------------------------------------------
# 6. Show a few images
# ---------------------------------------------------------------------------
# matplotlib expects an image as (height, width, channels) with values in
# [0, 1] (floats) or [0, 255] (ints). Your batch tensors are (channels,
# height, width) and *normalized*, i.e. some values are negative. So to display:
#   - undo the normalize:  x * std + mean
#   - move channels last:  tensor.permute(1, 2, 0)
#   - clamp to [0, 1] to kill any tiny out-of-range values from rounding
#
# If you skip the un-normalize, the images come out with weird colours and
# blown-out contrast — that's the visual symptom to recognize.


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """(3, H, W) normalized tensor -> (H, W, 3) tensor in [0, 1] for matplotlib."""
    # TODO: implement using `mean` and `std` from section 1.
    raise NotImplementedError


# TODO: with matplotlib, draw a small grid (e.g. 8 images) from your batch,
#       titling each with CLASSES[label]. Save to a PNG next to this file or
#       plt.show() it.
