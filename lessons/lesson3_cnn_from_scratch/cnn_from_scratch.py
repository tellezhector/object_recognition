"""Lesson 3 — Build a CNN from scratch.

Goal: understand what convolution, pooling, and a classification head
actually do, by wiring one together yourself.

This file is a scaffold. The section headers and comments lay out the steps;
the actual nn.Module code is yours to write where you see TODO.
Run with:  uv run python lessons/lesson3_cnn_from_scratch/cnn_from_scratch.py
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 1. Work out the shapes on paper FIRST
# ---------------------------------------------------------------------------
# Before writing any layer, fill this table in by hand for a 32x32x3 input,
# for whatever architecture you choose in section 2. The formula for one
# spatial dimension through Conv2d (and MaxPool2d, treating it as a conv with
# no learned weights) is:
#   out = floor((in + 2*padding - kernel_size) / stride) + 1
# MaxPool2d defaults to stride = kernel_size, so kernel_size=2 just halves H
# and W.
#
# This whas filled by students:
# layer                          | config (k, stride, pad) | output (C, H, W)
# --------------------------------|--------------------------|------------------
# input                           | -                        | (3, 32, 32)
# conv1                           | k=3, stride=1, pad=1     | (16, 32, 32)
# pool1                           | k=2, stride=2, pad=0     | (16, 16, 16)
# conv2                           | k=3, stride=1, pad=1     | (32, 16, 16)
# pool2                           | k=2, stride=2, pad=0     | (32, 8, 8)
# flatten                         | -                        | (2048,)
# fc1                             | -                        | (16,)
# fc2 (output)                    | -                        | (10,)
#
# Section 3 gives you a way to check this table against what actually runs.


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------
# nn.Module subclasses declare layers in __init__ and wire them together in
# forward(). Aim for something like 2-3 conv blocks (Conv2d -> activation ->
# MaxPool2d) followed by 1-2 Linear layers. CIFAR-10 has 10 classes, so your
# last Linear layer's out_features must be 10 — and don't add a final
# softmax/activation after it, that gets folded into CrossEntropyLoss in
# Lesson 4.
#
# Reminders:
#   - Conv2d(in_channels, out_channels, kernel_size, padding=...): the first
#     conv's in_channels must match the image (3, for RGB). Every conv after
#     that takes the *previous* layer's out_channels as its in_channels.
#   - You need an activation function (e.g. nn.ReLU) between layers — stack
#     Linear/Conv layers with no nonlinearity between them and they collapse
#     into one big linear function, no matter how many you chain.
#   - Before the first Linear layer, flatten (C, H, W) down to one dimension
#     per image — nn.Flatten() as a layer, or x.view(x.size(0), -1) in
#     forward(). Either way, that first Linear's in_features must equal
#     C * H * W at that point (from your section 1 table).


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: define your conv blocks (nn.Conv2d, activation, nn.MaxPool2d)
        # and your FC head (flatten + nn.Linear layers) as attributes here.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: pass x through the layers you defined above, in order, and
        # return the final (batch, 10) tensor of class scores.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Verify shapes at runtime
# ---------------------------------------------------------------------------
# A dummy input lets you sanity-check shapes without touching real data or a
# GPU. Temporarily printing x.shape after each layer inside forward() is a
# normal way to check your section 1 table against what actually runs — pull
# those prints back out once you've confirmed it.

if __name__ == "__main__":
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 32, 32)  # (batch=1, C=3, H=32, W=32)
    output = model(dummy_input)
    print(f"{output.shape=}")  # expect (1, 10)
