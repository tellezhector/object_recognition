"""Lesson 6 — Evaluation & save/load (part 2 of 2).

Goal: prove the checkpoint from evaluate_and_save.py is actually reusable —
load it into a FRESH model instance, in a separate process, and classify one
new image end to end.

This is a SEPARATE script on purpose: it must not import anything from
evaluate_and_save.py (no shared `model` variable, no shared training state).
It reconstructs the architecture from scratch and loads only the saved
weights, the way you'd do it on a different machine or in a deployed service
days after training.

Run evaluate_and_save.py first so checkpoint.pt exists, then:
  uv run python lessons/lesson6_eval_save_load/classify_image.py path/to/image.jpg

The exercise is the four TODOs below: load_model, load_image, predict, and
wiring them together in __main__.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
import torch
from torch import nn
from torchvision import transforms

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

# Must match the normalization evaluate_and_save.py trained with.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Model (must be an EXACT architecture match with evaluate_and_save.py — a
# state_dict has no record of layer shapes, only learned tensor values keyed
# by name, so if this class doesn't match, load_state_dict will fail or
# silently load into the wrong shapes).
# ---------------------------------------------------------------------------
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


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ---------------------------------------------------------------------------
# DONE 1: reconstruct the architecture and load the saved weights into it.
# ---------------------------------------------------------------------------
# Reminders:
#   - Instantiate SimpleCNN() fresh (random weights) — this is "reconstructs
#     the architecture" from the exercise.
#   - torch.load(checkpoint_path, map_location=device) reads back whatever
#     was saved — here, a state_dict (an OrderedDict of tensor names ->
#     values). Pass weights_only=True if your torch version supports it.
#   - model.load_state_dict(the_loaded_dict) copies those values into the
#     matching named layers of your fresh instance.
#   - Put the model in eval mode before returning — you're doing inference,
#     not training (see the Lesson 4/5 checkpoints on what that toggles).
def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = SimpleCNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# DONE 2: load a single image file and turn it into the tensor shape the
# model expects.
# ---------------------------------------------------------------------------
# Reminders:
#   - PIL.Image.open(path).convert("RGB") gets you a PIL image regardless of
#     the source file's original mode (grayscale, RGBA, etc.) — the model
#     was trained on 3-channel images.
#   - `transform` above (already built) does resize->tensor->normalize in
#     one call: transform(pil_image) -> a (3, 32, 32) tensor.
#   - The model expects a BATCH dimension: (batch, 3, 32, 32), not (3, 32,
#     32). Look up `.unsqueeze(0)`.
def load_image(path: Path) -> torch.Tensor:
    pil_image = PIL.Image.open(path).convert("RGB")
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor


# ---------------------------------------------------------------------------
# DONE 3: run inference and turn the output into a human-readable label.
# ---------------------------------------------------------------------------
# Reminders:
#   - No gradients needed here — wrap the forward pass in torch.no_grad()
#     (or decorate this function with @torch.no_grad()).
#   - logits = model(image_tensor) -> shape (1, 10). argmax(dim=1) gives you
#     the predicted class index as a (1,)-shaped tensor; .item() unwraps it
#     to a plain Python int.
#   - Use that int to index into CLASSES to get the label string.
def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> str:
    with torch.inference_mode():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        predicted_index = logits.argmax(dim=1).item()
        label = CLASSES[predicted_index]
        return label


# ---------------------------------------------------------------------------
# TODO 4: wire it together.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: uv run python {Path(__file__).name} <path-to-image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO:
    #  - model = load_model(CHECKPOINT_PATH, device)
    #  - image_tensor = load_image(image_path)
    #  - label = predict(model, image_tensor, device)
    #  - print(f"Predicted class: {label}")
    model = load_model(CHECKPOINT_PATH, device)
    image_tensor = load_image(image_path)
    label = predict(model, image_tensor, device)
    print(f"Predicted class: {label}")

    # Show the exact tensor the model saw — undo the Normalize step so it
    # displays as a viewable image instead of near-zero-mean noise.
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    unnormalized = image_tensor.squeeze(0) * std + mean
    unnormalized = unnormalized.clamp(0, 1).permute(1, 2, 0).numpy()
    plt.title(f"Predicted: {label}")
    plt.imshow(unnormalized)
    plt.axis("off")
    plt.show()
