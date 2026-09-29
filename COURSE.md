# Object Recognition with PyTorch — A Learn-by-Doing Course

## How this works

Each lesson has a **goal**, **concepts** you need before you can do the exercise, and an
**exercise** — a task description, not code. I will not hand you working solutions up
front. When you get stuck, tell me *what you tried and what broke*, and I'll ask
questions or give you a nudge (a concept to re-read, a function to look up, a hint about
where your mental model is off) rather than the fix itself. If you explicitly want to see
a reference solution to compare against after you've got something working, ask — but the
default is: you write it, I review it.

Each lesson also has a **checkpoint**: 1-3 questions you should be able to answer in your
own words before moving on. If you can't, that's a signal to slow down, not a test.

Track your progress by checking off lessons below. Keep your code in this repo — a
`notebooks/` or `src/` per lesson is fine, your call.

## Your setup

- GPU: RTX 2080 Super, 8GB VRAM — plenty for everything in this course, including
  fine-tuning detection models, as long as you're sensible about batch size and image
  resolution.
- Background: comfortable with core ML concepts (loss functions, gradient descent),
  new to PyTorch specifically.
- Path: classification fundamentals → transfer learning → object detection.

## Datasets (and why)

I picked datasets deliberately so the *dataset* stops being a variable you have to think
about — the point of each phase is the modeling/training concept, not data wrangling.

| Phase | Dataset | Why this one |
|---|---|---|
| Fundamentals (1–6) | **CIFAR-10** | Ships built into `torchvision.datasets`, no manual download/parsing needed, small enough (60k 32×32 images) to iterate fast on your GPU, but non-trivial — a from-scratch CNN plateaus well below 100%, so you'll actually feel the effect of architecture/regularization choices. |
| Transfer learning (7–10) | **Imagenette** | A 10-class subset of ImageNet released by fastai specifically for fast prototyping. Real photos, `ImageFolder`-compatible directory layout (`train/<class>/*.jpg`) — you'll write your own `Dataset`/`ImageFolder` loading code against real files instead of a preprocessed tensor blob like CIFAR-10. Small enough to fine-tune in minutes on your GPU. |
| Detection (11–15) | **PASCAL VOC 2007** | The classic teaching dataset for detection — `torchvision.datasets.VOCDetection` gives it to you directly with XML annotations. ~5k trainval images, multiple objects per image, real bounding boxes. Small enough to train on 8GB VRAM; standard enough that every detection tutorial/paper concept (IoU, NMS, anchors, mAP) maps directly onto it. COCO is the "real" industry dataset but it's 10-100x the size and overkill for learning the concepts. |

Don't download everything now — each phase tells you what to grab when you get there.

---

## Lesson 0 — Environment

**Goal:** Working PyTorch install that sees your GPU, in an isolated, reproducible
environment.

**Status:** done for this repo, using `uv` + the `cookiecutter-uv` template. Steps below
are documented so you can replicate this on another machine, and so you understand what
each piece is doing rather than just having a working `venv` appear.

### Setup steps (already applied to this repo)

**1. Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# installs to ~/.local/bin — make sure that's on your PATH
```

**2. Check your GPU driver's max supported CUDA version**

```bash
nvidia-smi   # look at the "CUDA Version" field in the header
```

This tells you the *newest* CUDA toolkit your driver can run (drivers are
backwards-compatible with older CUDA toolkit versions, not forwards). E.g. a driver
reporting `CUDA Version: 13.1` can run wheels built for CUDA 13.0 or 13.1, but not 13.2+.
This machine's driver (590.48.01) reports CUDA 13.1.

**3. Scaffold the project with the uv cookiecutter template**

```bash
uvx cookiecutter https://github.com/fpgmaas/cookiecutter-uv.git --no-input \
  project_name="object-recognition" \
  author="Your Name" \
  email="you@example.com"
```

`uvx` runs `cookiecutter` in a throwaway environment without installing it globally. This
generates a full project (pyproject.toml, ruff/mypy/pytest config, pre-commit hooks,
GitHub Actions CI, MkDocs docs site, Dockerfile, devcontainer) into a new
`object-recognition/` directory — move its contents into your repo root (or generate it
directly there if you don't already have other files to preserve).

**4. Pin the Python version**

```bash
uv python pin 3.12
```

**5. Add non-GPU dependencies normally**

```bash
uv add numpy matplotlib pillow
```

**6. Add a CUDA-specific index for torch/torchvision**

PyPI only hosts CPU-only PyTorch wheels; GPU wheels live on PyTorch's own index, split by
CUDA version. Find the CUDA versions available and pick one at or below what your driver
supports (step 2):

```bash
curl -s https://download.pytorch.org/whl/cu130/torch/ | grep -oP 'torch-[\d.]+(?=%2Bcu130)'
```

Then add this to `pyproject.toml` (already done in this repo — see the live file for the
current version pins):

```toml
[project]
dependencies = [
    # ...
    "torch>=2.13.0",
    "torchvision>=0.25.0",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
torchvision = [{ index = "pytorch-cu130" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Swap `cu130` for whatever CUDA version fits your driver.

**7. Install everything**

```bash
uv sync
```

**8. Install pre-commit hooks**

```bash
make install   # runs `uv sync` + `uv run pre-commit install`
```

**9. Jupyter (optional, for notebooks)**

Some lessons are easier to explore interactively — plotting data, watching a training
loop's output evolve cell by cell — than as a flat script. `jupyterlab` and `ipykernel`
are dev dependencies already in `pyproject.toml`, so no separate install or kernel
registration is needed. To start it:

```bash
uv run jupyter lab
```

This opens JupyterLab in your browser, using the project's own `.venv` (same
torch/CUDA setup as your scripts) as the kernel. Navigate to a lesson's `.ipynb` file and
run cells with `Shift+Enter`. Plain `.py` scripts remain fine for lessons that don't need
visualization — notebooks are opt-in, not a replacement.

### Your exercise

The one part left for you: write a small script (`lessons/lesson0_environment/`) that
creates a tensor, moves it to `cuda`, and prints the device and
`torch.cuda.get_device_name(0)` — run it with `uv run python lessons/lesson0_environment/<file>.py`
to confirm your environment actually works end to end, not just that install commands
succeeded.

**Checkpoint:** What's the difference between a CUDA "driver version" and the CUDA
version PyTorch was built against? Why can they differ?

---

## Phase 1 — Fundamentals (CIFAR-10)

### Lesson 1 — Tensors & autograd

**Goal:** Understand what PyTorch is doing under the hood before you use the high-level
APIs that hide it.

**Concepts:** tensors vs numpy arrays, `requires_grad`, computational graphs, `.backward()`,
`.grad`, why we `zero_grad()`.

**What is autograd?**

Autograd is PyTorch's automatic differentiation engine — the machinery that computes
gradients for you, so you don't have to derive and hand-code calculus by hand every time
you change your model. Here's the mental model, step by step:

1. **Every tensor operation gets recorded.** When a tensor has `requires_grad=True`, any
   operation you do with it (`+`, `*`, `**`, matrix multiplies, anything differentiable)
   doesn't just compute a result — PyTorch also silently records *what operation produced
   it* and *what its inputs were*. Do this across several chained operations and you build
   up a **computational graph**: a record of every step from your inputs to your final
   output.

2. **The graph flows forward, gradients flow backward.** This recording happens during
   your normal "forward pass" — the part where you compute `prediction`, then `loss`, from
   your inputs. Once you have a single scalar number (like a loss value) at the end of that
   chain, calling `.backward()` walks the graph in reverse, applying the chain rule from
   calculus at each recorded step, to work out exactly how much each `requires_grad=True`
   tensor upstream contributed to that final number.

3. **The result lands in `.grad`.** For every leaf tensor involved (a tensor you created
   directly with `requires_grad=True`, as opposed to one PyTorch computed along the way),
   `.backward()` fills in `.grad` with "if I nudge this tensor's value up slightly, how much
   does the final output change?" — the partial derivative of the output with respect to
   that tensor.

4. **You do the rest.** Autograd only computes gradients — it doesn't decide what to do
   with them. That's why you write the update step yourself: subtract a small multiple of
   `.grad` from each parameter (that's gradient descent), inside `torch.no_grad()` so that
   update itself isn't recorded as *another* operation on the graph.

Why this matters: without autograd, you'd have to manually work out `d(loss)/d(w)` and
`d(loss)/d(b)` by hand for every architecture you ever write — fine for `y = wx + b`,
completely impractical for a 50-layer CNN. Autograd is what makes arbitrarily complex,
differentiable models trainable without hand-deriving calculus each time. High-level APIs
like `torch.nn` and PyTorch's optimizers are convenience layers *on top of* autograd — this
lesson has you skip them so you see the engine underneath before you rely on it invisibly.

**Exercise:** Without using `torch.nn`, implement linear regression (`y = wx + b`) by
hand: create `w`, `b` as tensors with `requires_grad=True`, write the forward pass, MSE
loss, call `.backward()`, and manually update `w`/`b` inside a `torch.no_grad()` block, in
a loop, on some synthetic data you generate yourself. Get the loss to converge.

**Syntax primer:** these snippets show the mechanics on a toy example that is *not* your
regression — don't copy them in, use them to see how the pieces behave.

Creating a tensor that tracks gradients, and reading its grad after `.backward()`:

```python
import torch

a = torch.tensor(3.0, requires_grad=True)
out = a ** 2          # some differentiable expression using a
out.backward()        # populates a.grad with d(out)/d(a)
print(a.grad)          # -> 6.0  (d/da of a^2 is 2a, at a=3)
```

Note `out` here is a scalar — `.backward()` needs a scalar to call it with no arguments
(that's what your loss will be). If you call it a second time without clearing grads
first, PyTorch *adds* the new gradient onto whatever's already in `.grad` rather than
replacing it — that's why a real training loop zeroes grads each iteration:

```python
a.grad.zero_()   # or: reset it however you update your params each step
```

Updating a tensor's value in place, without autograd trying to track that update as part
of the graph:

```python
with torch.no_grad():
    a -= 0.1 * a.grad   # a "step" in the direction that shrinks `out`
```

Generating synthetic data is just tensor creation — e.g. `torch.rand`, `torch.randn`, or
building an input tensor by hand and computing a target from some formula you pick plus
noise. Look up `torch.randn` and `torch.manual_seed` if you want reproducible noise.

**Checkpoint:** Why do you need `torch.no_grad()` when updating the weights? What would
go wrong if you forgot `zero_grad()` on the second iteration?

P.S. Watch: https://www.youtube.com/watch?v=VMj-3S1tku0

### Lesson 2 — Data loading

**Goal:** Get CIFAR-10 flowing through a `DataLoader`.

**Concepts:** `Dataset`, `DataLoader`, batching, `transforms.Compose`, normalization,
train/test split.

The CIFAR-10  dataset consists of 60000 32x32 colour images in 10 classes, with
6000 images per class. There are 50000 training images and 10000 test images.
https://cave.cs.toronto.edu/kriz/cifar.html

**Exercise:** Load CIFAR-10 via `torchvision.datasets.CIFAR10` (it'll download
automatically). Build train and test `DataLoader`s. Write a small script that pulls one
batch and prints its shape and dtype, and displays a few images with their labels using
matplotlib (remember to un-normalize before displaying).

**Checkpoint:** Why do we normalize inputs? Why compute normalization stats (mean/std)
from the *training* set only, not train+test combined?

### Lesson 3 — Build a CNN from scratch

**Goal:** Understand what convolution, pooling, and a classification head actually do,
by wiring one together yourself.

**Concepts:** `nn.Conv2d` (in/out channels, kernel size, stride, padding), `nn.MaxPool2d`,
`nn.Linear`, activation functions, how spatial dimensions shrink layer to layer, flattening
before the FC head.

**What does a convolution actually do?**

1. **A kernel is a small sliding filter.** `nn.Conv2d` holds a set of learnable weight
   grids (kernels), e.g. 3×3. Each kernel slides across the input and at every position
   computes a weighted sum of the pixels under it — one output value per position. Slide
   it across the whole image and you get a full output grid (a "feature map").

2. **`in_channels`/`out_channels` set how many kernels and what they see.** Each kernel
   spans all of the input's channels at once (a 3×3 kernel on an RGB image is really
   3×3×3 = 27 weights), so `in_channels` must match the previous layer's `out_channels`
   (3 for a raw RGB image). `out_channels` is how many independent kernels this layer
   learns in parallel — each one free to specialize in detecting a different local
   pattern (an edge, a color transition, later a texture).

3. **`stride` controls how far the kernel jumps between positions.** `stride=1` moves one
   pixel at a time (dense overlap). `stride=2` skips every other position, roughly
   halving the output size — stride does downsampling as part of the convolution itself.

4. **`padding` adds a border (usually zeros) before convolving.** A kernel can't be
   centered on the outermost pixels without something to overlap past the edge, so
   without padding, each conv shrinks the spatial size by `kernel_size - 1`. `padding=1`
   with a 3×3 kernel exactly cancels that shrink ("same" padding) — output H/W equals
   input H/W.

5. **Putting 2-4 together, the output size formula is:**
   `out = floor((in + 2*padding - kernel_size) / stride) + 1` — this is what you'll use
   to fill in the shape table in the Lesson 3 scaffold.

6. **`nn.MaxPool2d(kernel_size=k)` shrinks spatially without learned weights** — it just
   takes the max value in each k×k block, defaulting to `stride=k` so blocks don't
   overlap (e.g. `k=2` exactly halves H and W). This reduces compute and adds a little
   tolerance to small shifts in where a feature appears.

7. **Activation functions (e.g. `nn.ReLU`) between layers are what makes stacking
   layers meaningful.** Without a nonlinearity between them, any stack of Conv/Linear
   layers collapses mathematically into one big linear function — no matter how many
   you chain, you'd gain nothing over a single layer.

8. **Flattening bridges "where are features" to "what class is this."** After your conv
   blocks, the remaining tensor is (channels, H, W) per image — a spatial map, not a
   decision. `nn.Flatten()` (or `x.view(x.size(0), -1)`) collapses that into one vector
   per image so `nn.Linear` layers can combine everything detected into class scores.

**Exercise:** Subclass `nn.Module` and write a small CNN for CIFAR-10 (something like
2-3 conv blocks + 1-2 FC (Fully Connected) layers — architecture choices are yours).
Before running anything, work out on paper what the output shape is after each layer given a 32×32×3
input, and verify it matches at runtime.

**Checkpoint:** Given `Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)`
on a 32×32 input, what's the output shape, and why does `padding=1` matter here?

### Lesson 4 — Training loop

**Goal:** Write (not import) a full training loop.

**Concepts:** loss functions (`CrossEntropyLoss` — and why you don't need a softmax layer
before it), optimizers (`SGD` vs `Adam`), epochs vs steps, `model.train()`/`model.eval()`,
tracking loss/accuracy.

**What are `CrossEntropyLoss`, and `SGD` vs `Adam`, actually doing?**

1. **Your model's last layer outputs raw scores ("logits"), not probabilities.** A
   `Linear` layer's output can be any real number, positive or negative — nothing forces
   it to look like a probability distribution (values in [0,1] summing to 1).

2. **`CrossEntropyLoss` applies softmax internally, then penalizes the log-probability of
   the correct class.** Softmax converts logits into probabilities (exponentiate, then
   divide by the sum, so they're positive and sum to 1). Cross-entropy then measures
   `-log(probability assigned to the true class)` — low when the model is confidently
   correct, and it blows up (→ ∞) the more confidently wrong the model is. PyTorch's
   `CrossEntropyLoss` does both steps internally, combined for numerical stability, and
   expects raw logits as input — that's why adding your own softmax before it is wrong:
   you'd be softmaxing twice, distorting the gradient.

3. **Gradient descent means "step opposite the gradient, scaled by a learning rate."**
   `SGD` does close to exactly that, per batch: `param -= lr * param.grad` (optionally
   with momentum — a running average of recent gradients, so it doesn't fully reverse
   direction on every noisy batch).

4. **`Adam` adapts the step size per-parameter using running statistics of the gradient.**
   It tracks a running average of each parameter's gradient (like momentum) *and* a
   running average of the gradient's square, then divides the step by roughly the square
   root of that second average. Parameters with small/consistent gradients get bigger
   effective steps; parameters with large/noisy gradients get smaller ones. This usually
   converges faster and needs less learning-rate tuning than plain `SGD`, at the cost of
   more memory (it stores two extra running averages per parameter).

5. **`model.train()`/`model.eval()` toggle behavior in specific layer types** — most
   layers behave identically either way; it only matters for layers like `Dropout`
   (active only in train) and `BatchNorm` (uses batch statistics in train, running
   statistics in eval). This is exactly what Lesson 4's checkpoint is asking you to
   notice about your own architecture.

**Exercise:** Write the training loop for your Lesson 3 model: for each epoch, iterate
batches, forward pass, compute loss, backward, optimizer step, and log train loss +
accuracy. Add a validation pass each epoch (no grad, `model.eval()`). Plot train vs val
loss curves.

**Checkpoint:** What does `model.eval()` actually change in your specific architecture
(hint: does your model have any layers that behave differently in train vs eval)? If it
changes nothing yet, what layer would you need to add for it to matter?

### Lesson 5 — Regularization & overfitting

**Goal:** Diagnose and address overfitting.

**Concepts:** `BatchNorm2d`, `Dropout`, data augmentation (`RandomCrop`, `RandomHorizontalFlip`),
weight decay, learning rate scheduling.

**What do these regularization techniques actually do?**

1. **`BatchNorm2d` normalizes activations mid-network, per channel, using the current
   batch's statistics.** For each channel, it subtracts that batch's mean and divides by
   that batch's std (then applies a small learned scale/shift), so activations flowing
   into the next layer stay in a consistent, well-behaved range regardless of how the
   previous layer's outputs happen to be distributed. This mainly speeds up and
   stabilizes training, but as a side effect the batch-dependent noise it introduces also
   mildly regularizes.

2. **`Dropout` randomly zeroes a fraction of activations, only during training.** Each
   forward pass, every unit has some probability `p` of being set to 0. This prevents the
   network from becoming overly reliant on any single unit or specific combination of
   units — it's forced to learn redundant, more robust representations. At eval time
   dropout is disabled (all units active) — which is exactly why `model.eval()` matters
   once you add this layer.

3. **Data augmentation manufactures new training variety from existing images.**
   `RandomCrop`/`RandomHorizontalFlip` (and friends) apply a random transform *each time*
   an image is loaded — so the model never sees the exact same pixels twice across
   epochs, even though the underlying image repeats. This makes memorizing individual
   training images a less useful strategy than learning features that generalize.

4. **Weight decay penalizes large weights.** It adds a term proportional to the sum of
   squared weights onto the loss (equivalently, subtracts a small fraction of each
   weight's value every step, independent of the gradient). Smaller weights generally
   mean a smoother, less extreme function — one less likely to have contorted itself to
   fit training-set noise.

5. **Learning rate scheduling shrinks the step size as training progresses.** Early on,
   large steps make fast progress toward a good region of parameter space; late in
   training, those same large steps tend to overshoot and bounce around near a minimum
   instead of settling into it. Reducing the LR (via `StepLR`, `CosineAnnealingLR`, etc.)
   lets later epochs make smaller, more precise refinements.

All five are separate levers for the same underlying goal: keep the model from fitting
training-set-specific noise instead of the general pattern.

**Exercise:** Train your Lesson 4 setup long enough to see train/val accuracy diverge
(overfitting). Then apply at least two of: batch norm, dropout, augmentation, weight
decay — and show the gap shrink. Compare curves before/after.

**Checkpoint:** Why does data augmentation reduce overfitting? Where in the training
pipeline does it get applied — every epoch on the same images, or once up front?

### Lesson 6 — Evaluation & save/load

**Goal:** Properly evaluate a model and be able to reuse it later.

**Concepts:** confusion matrix, per-class accuracy, `torch.save`/`torch.load`,
`state_dict()` vs saving the whole model, inference mode.

**Exercise:** Compute a confusion matrix on the CIFAR-10 test set and identify your
model's most-confused class pair. Save the model's `state_dict`, then write a **separate**
script that reconstructs the architecture, loads the weights, and classifies a single new
image end-to-end (load → transform → predict → print label).

**Checkpoint:** Why is saving `state_dict()` generally preferred over saving the whole
model object with `torch.save(model, ...)`?

---

## Phase 2 — Transfer Learning (Imagenette)

### Lesson 7 — Pretrained backbones

**Goal:** Understand what a pretrained model actually gives you.

**Concepts:** ImageNet pretraining, `torchvision.models`, feature extraction vs
fine-tuning, freezing layers, replacing the classification head.

**Exercise:** Download Imagenette. Load a pretrained ResNet (your choice of size) from
`torchvision.models`, freeze all layers except the final FC layer, replace that layer to
output 10 classes, and train just that layer on Imagenette.

**Checkpoint:** Why does freezing most of the network still work well, when those layers
were trained on totally different classes than yours?

### Lesson 8 — Fine-tuning strategies

**Goal:** Go beyond linear-probing the head.

**Concepts:** differential/discriminative learning rates, gradual unfreezing, LR
schedulers (`StepLR`, `CosineAnnealingLR`), when fine-tuning beats feature extraction.

**Exercise:** Starting from Lesson 7, unfreeze more of the network progressively (e.g.
last block, then last two) and compare accuracy/training time against the frozen
baseline. Try using a smaller learning rate for the pretrained layers than the new head.

**Checkpoint:** Why would you want a *smaller* learning rate on the pretrained layers
than on your newly-initialized head?

### Lesson 9 — Custom datasets from image folders

**Goal:** Stop depending on `torchvision.datasets` built-ins; load arbitrary image data.

**Concepts:** `ImageFolder`, writing a custom `Dataset` subclass (`__len__`, `__getitem__`),
`transforms` for PIL images, when you'd need a custom `Dataset` vs when `ImageFolder`
suffices.

**Exercise:** Load Imagenette with `ImageFolder` first — should be nearly trivial given
its directory layout. Then, as an exercise, write your **own** `Dataset` subclass that
does the same thing without using `ImageFolder` (walk the directory yourself, build a
label map, open images with PIL in `__getitem__`). Confirm both give identical results.

**Checkpoint:** What does `__getitem__` need to return, and at what point in the pipeline
does the `transform` actually get applied to the raw image?

### Lesson 10 — Model comparison & error analysis

**Goal:** Practice the "which model do I actually ship" decision process.

**Concepts:** comparing architectures/hyperparameters rigorously, looking at *which*
examples a model gets wrong (not just aggregate accuracy), precision/recall per class.

**Exercise:** Train at least two variants (different backbone, or frozen vs fine-tuned,
or with/without augmentation) on Imagenette. Build a small report: accuracy table, and a
grid of images your best model got wrong with predicted vs true label. Look for a
pattern in the errors.

**Checkpoint:** What's one hypothesis your error grid suggests, and how would you test
it?

---

## Phase 3 — Object Detection (PASCAL VOC 2007)

### Lesson 11 — From classification to detection

**Goal:** Understand the problem shift: "what" → "what and where."

**Concepts:** bounding box formats (xyxy vs xywh vs normalized), what a detection
dataset's labels actually look like, `torchvision.datasets.VOCDetection`, parsing XML
annotations, drawing boxes on images.

**Exercise:** Load VOC 2007 trainval. Write code to parse one annotation and draw its
bounding box(es) + class labels on the image using matplotlib. Do this for a handful of
images including one with multiple objects of different classes.

**Checkpoint:** Why can't you just reuse your Lesson 3-6 classification pipeline directly
for detection — what fundamentally changes about the label shape and the loss?

### Lesson 12 — IoU, anchors, NMS

**Goal:** Understand the core primitives every anchor-based detector is built from,
before you use a library that does them for you.

**Concepts:** Intersection over Union, anchor boxes, how a detector proposes many
candidate boxes then filters, Non-Maximum Suppression.

**What are IoU, anchor boxes, and NMS?**

1. **Intersection over Union (IoU) measures how much two boxes overlap, as a single
   number from 0 to 1.** It's `area(box1 ∩ box2) / area(box1 ∪ box2)` — the area they
   share, divided by the total area either one covers. Two identical boxes give IoU=1;
   two boxes that don't touch give IoU=0. It's the standard way to ask "is this predicted
   box close enough to the true box to count as correct?"

2. **Anchor boxes are a fixed grid of reference boxes tiled across the image, at several
   scales and aspect ratios.** Instead of asking the network to predict raw box
   coordinates from nothing (a hard regression problem — where in the image, what size,
   what shape, all at once), each anchor gives it a starting guess, and the network only
   has to predict a small adjustment ("a bit wider than this anchor, shifted slightly
   left") plus a class/confidence score for that anchor. This is a much easier learning
   problem, and it's why detectors output *many* candidate boxes — one prediction per
   anchor, most anchors overlapping no real object at all.

3. **That flood of candidates is mostly redundant.** A real object near an anchor
   typically gets high-confidence predictions from several *neighboring* anchors too, all
   describing roughly the same box. Left alone, you'd report the same object 5-10 times.

4. **Non-Maximum Suppression (NMS) collapses duplicates down to one box per object.**
   Sort all candidate boxes by confidence score, descending. Take the highest-scoring
   box, keep it, and discard every remaining box whose IoU with it exceeds some threshold
   (e.g. 0.5) — those are almost certainly the same object. Repeat with the next
   highest-scoring box still remaining, until none are left. The result: one box per
   real object, the most confident one in each overlapping cluster.

**Exercise:** Implement `iou(box1, box2)` yourself (no library) and test it against
hand-computed cases. Then implement NMS yourself: given a list of boxes with scores,
return the kept boxes. Test it on a synthetic case with overlapping boxes you construct.

**Checkpoint:** Walk through your NMS implementation on paper with 3 overlapping boxes
of different scores — which get suppressed and why?

### Lesson 13 — Fine-tune a pretrained detector

**Goal:** Put it together: fine-tune a real detection model on VOC.

**Concepts:** `torchvision.models.detection` (e.g. Faster R-CNN with a ResNet-FPN
backbone), the different target dict format detection models expect
(`boxes`, `labels`), the detector's built-in loss during training vs its output format
during eval.

**Exercise:** Load a pretrained Faster R-CNN from `torchvision.models.detection`, adapt
its head for VOC's 20 classes (+background), and fine-tune on a subset of VOC 2007 first
(to iterate fast) before the full trainval set. Watch your GPU memory — tune batch size
accordingly.

**Checkpoint:** Why do detection models typically include an explicit "background" class
that classification models don't need?

### Lesson 14 — Evaluation with mAP

**Goal:** Properly evaluate a detector — accuracy alone doesn't apply here.

**Concepts:** precision/recall at an IoU threshold, precision-recall curves, mean Average
Precision, why mAP is reported at specific IoU thresholds (e.g. mAP@0.5).

**What is mAP actually measuring?**

1. **A detection only counts as a true positive if both the class and the box are
   right.** "Right" for the class is a plain match. "Right" for the box means IoU (from
   Lesson 12) against the matching ground-truth box exceeds a chosen threshold — e.g. at
   mAP@0.5, a correct-class prediction whose box only reaches IoU=0.3 with the true box
   still counts as a false positive. That's the split accuracy alone can't express:
   detection has to be right about *both* what and where.

2. **Precision and recall trade off as you change the confidence threshold you accept.**
   Precision = TP/(TP+FP) — of the boxes you kept, how many were correct. Recall =
   TP/(TP+FN) — of all real objects, how many did you find. Lowering your acceptance
   threshold finds more true positives (recall ↑) but also lets in more false ones
   (precision ↓). Sweeping the threshold from strict to loose traces a precision-recall
   curve for one class.

3. **Average Precision (AP) is the area under that curve, for one class.** A model that
   stays high-precision even as recall increases (rare) scores near 1.0; one that trades
   away precision fast for small recall gains scores lower.

4. **mAP is just the mean of AP across all classes** — one number summarizing overall
   detector quality, per class weighted equally regardless of how common that class is.

5. **The IoU threshold in "mAP@0.5" sets how strict "the box was right" needs to be.**
   Reporting mAP at multiple thresholds (0.5, 0.75, …) shows whether a model is merely
   finding the right general area (passes at 0.5, fails at 0.75) or genuinely localizing
   tightly (passes at both) — a looser threshold alone can hide sloppy box placement.

**Exercise:** Run your Lesson 13 model on VOC's test set. Implement (or carefully use a
library for, your choice — but understand it first) mAP@0.5 computation. Report per-class
AP and identify your weakest class.

**Checkpoint:** Explain in your own words why a detection with a "correct" class but a
poorly-placed box counts as a false positive at a given IoU threshold.

### Lesson 15 — Capstone

**Goal:** Put everything together end to end, your own choices, minimal hand-holding.

**Exercise:** Pick a small set of object classes you personally care about (could be a
VOC subset, or your own photos labeled with a tool like LabelImg/CVAT if you want to go
further). Fine-tune a detector, evaluate it with mAP, and build a small inference script
that takes an arbitrary image and outputs it with drawn boxes + labels + confidence
scores. Write up what worked, what didn't, and what you'd try next.

---

## Progress

- [x] Lesson 0 — Environment
- [x] Lesson 1 — Tensors & autograd
- [ ] Lesson 2 — Data loading
- [ ] Lesson 3 — Build a CNN from scratch
- [ ] Lesson 4 — Training loop
- [ ] Lesson 5 — Regularization & overfitting
- [ ] Lesson 6 — Evaluation & save/load
- [ ] Lesson 7 — Pretrained backbones
- [ ] Lesson 8 — Fine-tuning strategies
- [ ] Lesson 9 — Custom datasets from image folders
- [ ] Lesson 10 — Model comparison & error analysis
- [ ] Lesson 11 — From classification to detection
- [ ] Lesson 12 — IoU, anchors, NMS
- [ ] Lesson 13 — Fine-tune a pretrained detector
- [ ] Lesson 14 — Evaluation with mAP
- [ ] Lesson 15 — Capstone
