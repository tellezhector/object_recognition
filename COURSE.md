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

**Goal:** Working PyTorch install that sees your GPU.

**Exercise:**
- Install PyTorch with CUDA support (check pytorch.org for the right command for your
  CUDA version — `nvidia-smi` above shows driver 590.48.01).
- Install `torchvision`, `matplotlib`, `numpy`.
- Write a 5-line script that creates a tensor, moves it to `cuda`, and prints the device
  and `torch.cuda.get_device_name(0)`.

**Checkpoint:** What's the difference between a CUDA "driver version" and the CUDA
version PyTorch was built against? Why can they differ?

---

## Phase 1 — Fundamentals (CIFAR-10)

### Lesson 1 — Tensors & autograd

**Goal:** Understand what PyTorch is doing under the hood before you use the high-level
APIs that hide it.

**Concepts:** tensors vs numpy arrays, `requires_grad`, computational graphs, `.backward()`,
`.grad`, why we `zero_grad()`.

**Exercise:** Without using `torch.nn`, implement linear regression (`y = wx + b`) by
hand: create `w`, `b` as tensors with `requires_grad=True`, write the forward pass, MSE
loss, call `.backward()`, and manually update `w`/`b` inside a `torch.no_grad()` block, in
a loop, on some synthetic data you generate yourself. Get the loss to converge.

**Checkpoint:** Why do you need `torch.no_grad()` when updating the weights? What would
go wrong if you forgot `zero_grad()` on the second iteration?

### Lesson 2 — Data loading

**Goal:** Get CIFAR-10 flowing through a `DataLoader`.

**Concepts:** `Dataset`, `DataLoader`, batching, `transforms.Compose`, normalization,
train/test split.

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

**Exercise:** Subclass `nn.Module` and write a small CNN for CIFAR-10 (something like
2-3 conv blocks + 1-2 FC layers — architecture choices are yours). Before running
anything, work out on paper what the output shape is after each layer given a 32×32×3
input, and verify it matches at runtime.

**Checkpoint:** Given `Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)`
on a 32×32 input, what's the output shape, and why does `padding=1` matter here?

### Lesson 4 — Training loop

**Goal:** Write (not import) a full training loop.

**Concepts:** loss functions (`CrossEntropyLoss` — and why you don't need a softmax layer
before it), optimizers (`SGD` vs `Adam`), epochs vs steps, `model.train()`/`model.eval()`,
tracking loss/accuracy.

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

- [ ] Lesson 0 — Environment
- [ ] Lesson 1 — Tensors & autograd
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
