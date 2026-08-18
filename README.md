# object-recognition

A self-paced course on training neural networks for object recognition with PyTorch.
See [COURSE.md](COURSE.md) for the full lesson plan (fundamentals → transfer learning →
object detection).

## Environment

Managed with [uv](https://docs.astral.sh/uv/). Python is pinned via `.python-version`;
`torch`/`torchvision` are pulled from the PyTorch CUDA wheel index configured in
`pyproject.toml` (currently `cu130`, matched to a local RTX 2080 Super).

```bash
uv sync                 # install/update the environment from pyproject.toml + uv.lock
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Run anything in the environment with `uv run <command>` (e.g. `uv run python lessons/lesson1_tensors_autograd/main.py`),
or `uv run jupyter lab` if you prefer notebooks.

For the full from-scratch setup (installing uv, matching a CUDA wheel index to your GPU
driver, scaffolding via cookiecutter-uv), see **Lesson 0** in [COURSE.md](COURSE.md).

## Layout

- `lessons/lessonN_*/` — one directory per lesson in COURSE.md. Empty on purpose — this is
  where you write your own exercise code.
- `data/` — downloaded datasets land here (gitignored).
- `object_recognition/` — shared package code, if/when a lesson's code is worth reusing
  across later lessons (e.g. a shared training-loop helper). Not required to use.

## Dev tooling

This repo was scaffolded from [cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv),
so it also carries linting/testing/docs/CI scaffolding (`ruff`, `mypy`, `pytest`,
`pre-commit`, GitHub Actions, MkDocs). None of that is required to work through the
course — it's there if you want the practice of keeping a real project's tooling green
alongside the ML content.

```bash
make install             # uv sync + install pre-commit hooks
uv run pre-commit run -a # run lint/format checks
uv run pytest            # run tests
```
