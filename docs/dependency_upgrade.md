# Dependency upgrade

## Why the manifests were split into two tiers

Before this change, every dependency was exact-pinned in `setup.py`, most were
pinned _again_ in `requirements.txt`, and `install_env.sh` then ran an unpinned
`pip install torch torchvision` that overrode the pin it had just satisfied. The
result was a stack that looked reproducible and was not: the torch version you
got depended on the day you ran the installer.

Dependencies now fall into two tiers.

**Application layer** — compatible-version ranges. FastAPI, SQLAlchemy, requests,
urllib3, and friends have no coupling to the training stack. Exact-pinning them
bought nothing and froze known CVEs in place.

**ML runtime** — still exact-pinned. `torch`, `torchvision`, and `onnxruntime`
are ABI-sensitive and additionally constrained by vendored code (below). These
must move as one deliberate step, not as a side effect of an unrelated bump.

## What was removed

`unstructured`, `pypandoc`, `pandoc`, and `extract-msg` were declared runtime
dependencies but are imported nowhere in the codebase. `unstructured==0.6.6`
alone pulled a very large transitive tree and carried four advisories. `imageio`
was listed in `requirements.txt` and is likewise unused.

`tabulate` was kept — it is not imported by `anylearning/` directly, but the
vendored NanoDet COCO evaluator uses it.

## What was added

These were imported directly but never declared, arriving only transitively:
`pydantic`, `psutil`, `PyYAML`, `numpy`, `Pillow`, `opencv-python-headless`.
An undeclared direct dependency breaks the moment its accidental provider drops it.

## Security posture

The old pins, checked against [OSV](https://osv.dev):

| Package            | Old pin | Advisories      |
| ------------------ | ------- | --------------- |
| `python-multipart` | 0.0.6   | 18              |
| `urllib3`          | 2.0.2   | 16              |
| `requests`         | 2.31.0  | 6               |
| `unstructured`     | 0.6.6   | 4 (now removed) |
| `tqdm`             | 4.65.0  | 2               |
| `fastapi`          | 0.96.0  | 1               |

`python-multipart` and `urllib3` were the two on genuinely reachable paths —
multipart parses every file upload in `anylearning/routers/`, and urllib3 backs
the auto-labeling model downloads.

## The one migration this forces: Pydantic v1 → v2

`fastapi==0.96.0` constrains `pydantic<2`, so raising FastAPI moves the project
to Pydantic v2. The surface is small and fully enumerated:

- `class Config: orm_mode = True` → `model_config = ConfigDict(from_attributes=True)`
  — 4 sites (`routers/project.py`, `routers/model.py` ×2, `routers/dataset.py`)
- `.dict()` → `.model_dump()` — 3 sites
- `.from_orm()` → `.model_validate()` — 1 site (`routers/project.py:198`)

There are no `@validator`, `@root_validator`, or `parse_obj` usages. Pydantic v2
still accepts the v1 spellings with a deprecation warning, so the upgrade does
not hard-break on day one.

## Lightning 2 migration — done

`pytorch-lightning==1.9.0` (January 2023) used to be the anchor blocking
everything else, and it was worse than merely old: Lightning 1.9's
`lightning_fabric/__init__.py` calls `__import__("pkg_resources").declare_namespace()`
at import time. setuptools 81 removed `pkg_resources` and Python 3.12+ stopped
installing setuptools at all, so NanoDet could not be imported on a fresh
environment without a `setuptools<81` shim.

The migration turned out to be small, because `pl.Trainer(...)` in
`nanodet/tools/train.py` already used the 2.x-style `accelerator` / `devices` /
`strategy` / `precision` arguments. Only the removed epoch-end hooks needed work,
in `nanodet/trainer/task.py`:

| Lightning 1.x                         | Now                             |
| ------------------------------------- | ------------------------------- |
| `training_epoch_end(self, outputs)`   | `on_train_epoch_end(self)`      |
| `validation_epoch_end(self, outputs)` | `on_validation_epoch_end(self)` |
| `test_epoch_end(self, outputs)`       | `on_test_epoch_end(self)`       |

Lightning 2 no longer collects step return values, so the module accumulates them
itself in `self.validation_step_outputs` / `self.test_step_outputs`, reset in the
existing `on_validation_epoch_start` / `on_test_epoch_start` hooks. Forgetting the
reset would make results grow without bound and re-evaluate stale detections every
epoch, so `tests/nanodet/test_trainer/test_lightning_task.py` asserts it.

Note the hook names used here exist in **both** 1.9 and 2.x, so the change was
verified green under the old pin before the version moved.

Two bugs surfaced once the test actually exercised these hooks — previously it
passed an empty list, so the evaluation branch was never entered at all:

- the test had no evaluator, so `on_validation_epoch_end` was never really run;
- `on_test_epoch_end` leaked a file handle via `json.dump(res, open(path, "w"))`.

### Three more, found only by running a real training loop

The hook rename alone was not sufficient. `tests/e2e/test_all_flows_e2e.py`
actually calls `tools/train.py:main()`, and that turned up breakage the unit
tests could not see:

**`optimizer_step` had the wrong signature.** It still declared Lightning 1.x's
`optimizer_idx` parameter (plus `on_tpu` / `using_lbfgs`). Lightning 2 removed
those and calls the hook positionally, so the closure landed in `optimizer_idx`
and `optimizer_closure` stayed `None`. The optimizer then stepped without ever
running the closure, and Lightning aborted with _"The closure hasn't been
executed."_ This one is nasty because nothing about it looks version-specific.

**`strategy=None` / `devices=None` are rejected.** Lightning 1.9 accepted `None`;
Lightning 2 requires `"auto"`. Changing `devices` to `"auto"` then broke the
multi-GPU check `if devices and len(devices) > 1`, because `len("auto")` is 4 —
it now tests for a list explicitly.

**`lr` was a string in every shipped config.** PyYAML follows YAML 1.1, where
`1e-4` is _not_ valid float syntax — it parses as `str`. Production never noticed
because `prepare_config()` overwrites `lr` with the value from the UI, and
`min_lr` had already been wrapped in an explicit `float()` somewhere along the
way. The handpose flow, which uses the config value directly, died on
`'<=' not supported between instances of 'float' and 'str'`. All 15 occurrences
across the 7 config files are now written as `1.0e-4`.

Also fixed: handpose's `train()` never created `save_dir`, so it trained to
completion and then died on the first `torch.save()`. It only worked because
`BaseTrainer.prepare_folders()` happened to have made the directory first — the
same "not self-sufficient outside the app" pattern as the `logger=None` default.

**Every exported NanoDet ONNX model was in training mode.** `export_onnx.py`
never called `model.eval()` before `torch.onnx.export`, so the traced graph
captured BatchNorm updating its running statistics and Dropout active. torch
warns about precisely this, and nothing had ever exercised the export path to
surface it. This is a correctness bug in shipped models, not a version-upgrade
artefact — it predates the torch bump.

## segmentation-models-pytorch — done

`segmentation-models-pytorch==0.3.4` hard-pinned `timm==0.9.7` with an exact `==`,
so nothing modern in the timm ecosystem could coexist with it. Now on smp 0.5 with
timm 1.0, which also removed timm's `activations_jit.py` — the last in-tree user of
`@torch.jit.script`, which torch 2.11 deprecates.

While doing this, `semantic_segmentation/train.py` was found to hardcode
`pretrained = "imagenet"`, ignoring `config["model"]["pretrained"]` entirely. There
was therefore no way to build the model without fetching encoder weights, which
made any "offline" configuration a lie. The shipped config still specifies
imagenet, so production behaviour is unchanged.

## torch 2.5.1 → 2.11.0 — done

Verified green (185 tests) on torch 2.11.0+cu130 / torchvision 0.26.0 with
detectron2 rebuilt, Lightning 2.6.5, timm 1.0.28, smp 0.5.0, on a CUDA-capable
machine. Two things bit along the way; both are permanent lessons rather than
one-offs.

**`onnxscript` is now a required runtime dependency.** From torch 2.6 the
`torch.onnx.export` default moved to the dynamo-based exporter, which imports
`onnxscript` at module load. Without it _every_ trainer's ONNX export fails with
`ModuleNotFoundError: No module named 'onnxscript'` — and export is not an
optional feature here, it is the last step of every training job. It is now in
`setup.py` alongside torch.

**Upgrading across a CUDA major version needs a fresh environment.** torch 2.11
ships CUDA 13 wheels, which depend on differently-_named_ `nvidia-*-cu13`
distributions. An in-place upgrade therefore leaves the old `nvidia-*-cu12` set
installed but orphaned, and the stale copies shadow NVRTC. detectron2's
`pairwise_iou` then dies with:

```
nvrtc: error: failed to open libnvrtc-builtins.so.13.0
```

even though that file is present under `nvidia/cu13/lib/`. Do **not** try to fix
it by uninstalling the `-cu12` packages piecemeal — that also removes
`libcudnn.so.9` and torch stops importing entirely. A clean
`pip install --force-reinstall torch` (or a fresh env) resolves it, after which
the detectron2 training path passes. This is an environment artefact, not a
detectron2/torch incompatibility — worth knowing before concluding the latter.

Two `filterwarnings` ignores in `pyproject.toml` came out of this and are
deliberately narrow: fvcore's `@torch.jit.script` (commit-frozen via detectron2)
and a torch-internal pytree `FutureWarning` raised inside `torch.onnx`'s
decomposition pass. The second one aborts export as a `ConversionError` under
`-W error`; export was separately confirmed to produce a valid graph that loads
and runs in onnxruntime.

## NumPy 2

**The codebase itself is already NumPy 2 clean.** A scan of `anylearning/`
(including vendored NanoDet) for every NumPy 2 removal — `np.float_`,
`np.object`, `np.bool8`, `np.in1d`, `np.row_stack`, `np.trapz`, `np.product`,
`np.NaN` — and for `np.array(..., copy=False)`, which now raises, finds nothing.
The cap was never about our code. Four dependencies declared `numpy<2`:

| Package        | Pinned at | Declares     | First NumPy 2 release |
| -------------- | --------- | ------------ | --------------------- |
| `mediapipe`    | 0.10.18   | `numpy<2`    | **1.0.0**             |
| `onnxruntime`  | 1.18.1    | `numpy<2.0`  | **1.19**              |
| `torchmetrics` | 1.5.1     | `numpy<2.0`  | **1.6**               |
| `scipy`        | 1.12.0    | `numpy<1.29` | **1.13**              |

Two more are ABI rather than metadata constraints: `pycocotools` and `onnx` are C
extensions built against NumPy. Use `pycocotools>=2.0.10` and `onnx>=1.19`, which
ship wheels built for NumPy 2.

**mediapipe used to be the ceiling and no longer is.** 0.10.x shipped per-version
`cp39`–`cp312` wheels; 1.0.x ships a single `py3-none-manylinux_2_28_x86_64`
wheel, so it is ABI-agnostic and works on any Python 3.x.

Python version interacts with this. On 3.10 the newest usable versions are
`onnxruntime` 1.22 and `scipy` 1.15; `onnxruntime` 1.28 needs ≥3.11 and `scipy`
1.18 needs ≥3.12. Everything in the stack — torch 2.11, torchvision 0.26,
onnxruntime, scipy, pycocotools, onnx — publishes cp313 wheels, so Python 3.13 is
reachable, and that is where the project now sits: `setup.py` requires >=3.11
and classifies 3.11–3.13, and both CI workflows use 3.13.

The one thing metadata cannot answer is NEP 50 scalar promotion, which changes
numeric _results_ rather than raising — that needs the test suite run against
NumPy 2, not a grep.

## Remaining

1. **detectron2** has no PyPI release and is built against the installed torch, so
   any torch bump requires rebuilding it. `install_env.sh` pins it to commit
   `b4a4a3b` (2026-07-24 — still actively maintained) and passes
   `--no-build-isolation`, without which its `setup.py` cannot import torch and
   the build fails outright. It builds cleanly against torch 2.11.

2. **`onnxruntime`** is still at 1.18.1 (mid-2024). It is independent of the torch
   stack, so it can move on its own — and it must, to reach NumPy 2.

3. **Nuitka** is now pinned to the 4.x line. NumPy 2 support landed in 2.3.7, and
   newer releases matter for the packaged build: 2.8.10 aborted with a
   `NuitkaOptimizationError` while optimising `torch._dynamo.pgo` on torch 2.11.
   `build_app.sh` works around that by excluding `torch._dynamo` / `torch._inductor`
   — harmless, since nothing here calls `torch.compile` and `import torch` does not
   pull them in, but the exclusions may be droppable on 4.x.

## Verified stack

| Package                     | Was                    | Now              |
| --------------------------- | ---------------------- | ---------------- |
| torch / torchvision         | 2.5.1 / 0.20.1         | 2.11.0 / 0.26.0  |
| pytorch-lightning           | 1.9.0                  | 2.6.5            |
| segmentation-models-pytorch | 0.3.4                  | 0.5.0            |
| timm                        | 0.9.7 (pinned by smp)  | 1.0.28           |
| fastapi / pydantic          | 0.96.0 / v1            | 0.141.1 / 2.13.4 |
| onnxscript                  | absent (export broken) | 0.7.1            |
| detectron2                  | floating git HEAD      | pinned `b4a4a3b` |
