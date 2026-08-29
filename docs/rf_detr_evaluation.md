# RF-DETR: what it was measured to do, and what we decided

Evaluated on branch `rf-detr` in August 2026, against the models this app
already trains. The short version: **take it for object detection, refuse it
for instance segmentation, and do not make it the default.**

Measured on an RTX 3090 (shared, so accuracy and memory only) and an idle RTX
3080 (timings), 3 repeats per cell, with a control pair in every round.

## Object detection: it wins at every dataset size

Project 3, 315-image validation split, 20 epochs, batch 8, mAP@0.5:0.95:

| train images | NanoDet-Lightweight | RF-DETR-Nano |      |
| ------------ | ------------------- | ------------ | ---- |
| 100          | 0.188               | **0.530**    | 2.8x |
| 300          | 0.356               | **0.606**    | 1.7x |
| 945          | 0.460               | **0.626**    | 1.4x |

The 3080 agreed within 0.01 on every cell, so this is not a property of one
machine.

**There is no crossing point in this range.** RF-DETR wins from the smallest
size tested; the gap narrows with data without closing. The question is not
when RF-DETR starts winning but when NanoDet catches up, and it has not by 945
images.

And the comparison was unfair _to RF-DETR_. 1e-3 is NanoDet's shipped default;
RF-DETR's own templates ask for 1e-4. At each family's own rate on 100 images,
RF-DETR reaches **0.617 — more than NanoDet manages on 945**. At equal wall
clock rather than equal epochs (NanoDet 60 epochs / 679 s against RF-DETR 20
epochs / 624 s) it is 0.318 against 0.617.

## Instance segmentation: Mask R-CNN wins, and it is not close

Project 6, 49 train / 22 val, ~20 instances an image, 30 epochs, mask
mAP@0.5:0.95:

|                       | mask mAP  | peak alloc   |
| --------------------- | --------- | ------------ |
| **Mask R-CNN Medium** | **0.455** | **1,954 MB** |
| RF-DETR-Seg-Small     | 0.405     | 9,586 MB     |
| RF-DETR-Seg-Nano      | 0.377     | 9,513 MB     |

Mask R-CNN beats the best RF-DETR-Seg by 0.050 — eleven times the control
spread — using a fifth of the memory. Four RF-DETR-Seg runs died outright with
~5 GB free while Mask R-CNN kept going. A bigger model that needs a bigger GPU
and loses anyway.

## What it costs

|                       | NanoDet-Lightweight | RF-DETR-Nano       |
| --------------------- | ------------------- | ------------------ |
| peak GPU allocation   | 247 MB              | **4,009 MB** (16x) |
| registered checkpoint | 3.4 MB              | 121 MB             |
| exported ONNX         | 1.1 MB              | 108 MB             |
| ONNX on CPU, 1 thread | **16 ms**           | **463 ms** (29x)   |
| seconds per epoch     | 1x                  | 2.6x               |

Two of those decide the shape of the feature. Every RF-DETR model a user trains
costs about **230 MB of their data folder permanently**, against NanoDet's 4.5
MB. And the 29x CPU inference gap matters for an offline desktop product: a
GPU-less user can train one, slowly, and then cannot use it interactively. So
it is offered as "the accurate one, if you have a GPU", never as the default.

Packaging cost: `app.bin` 1.17 GB against the release build's 854 MB, `app.dist`
8.4 GB against 5.9, weights 2.0 GB / 37 files. Most of the compile is
`transformers`, not RF-DETR — and adding it invalidates every ccache entry, so
the first build after the merge is a cold one (1h49m measured).

## It does not work in a packaged build without two repairs

Both are one-line additions to `anylearning/frozen_compat.py`, which exists for
exactly this class of problem and is already called from `app.py` and
`training_job.py`.

1. **`import rfdetr` cannot complete in a compiled build.** It reaches
   `transformers/utils/doc.py`'s `get_docstring_indentation_level`, which calls
   `inspect.getsource`. A Nuitka module has no source, and the call is
   unconditional upstream:

       RuntimeError: ... Error: could not get source code

2. **Segmentation export fails**, which is worse than failing to train: the run
   _succeeds_, then `export_onnx` dies and registration is gated on export, so
   the finished model is discarded.

       ModuleNotFoundError: No module named 'torch.onnx.__init__'

   `rfdetr/models/heads/segmentation.py` defines a custom
   `torch.autograd.Function`; tracing one resolves `__module__`, and Nuitka
   names the package initialiser `torch.onnx.__init__`. Detection defines no
   custom Function, which is why only half of it looked broken.

With both applied: `feature_test.py` 19/20, `--self-test` 8/8 on the default
trainers, and RF-DETR detection and instance segmentation both training and
registering models in the packaged build.

**This is the finding to keep.** The branch passed 731 unit tests and trained
~100 times from source before anyone discovered it could not `import` in the
artefact. Nothing but a packaged build was ever going to say so.

## Multi-scale stays off, now measured rather than assumed

Confirmed from batch tensors: `multi_scale=False` pins every batch to 384x384;
with it on, shapes vary. The real range is **[224…544]** for Nano, not the
[128…704] a naive reading of `compute_multi_scale_scales` gives — the default
`num_windows=4` is not what the call passes, the variant's 2 is.

Detection differences are inside the noise floor at every dataset size.
Instance segmentation is the one real effect and it favours **off** (0.013,
three times its control spread, same sign in all three repeats). Off also costs
25% less memory on detection.

## Defects found in the branch, to fix before it ships

- **The dialog's learning rate overwrites the template's.** `prepare_config`
  takes `training_params.learning_rate`, whose dialog default is 0.001 — so a
  user accepting the defaults loses about 0.09 mAP. Needs a per-architecture
  default.
- **The number in the UI is not the model you get.** EMA is on by default and
  `BestModelCallback` registers the better of the regular and EMA checkpoints,
  but `rfdetr_logging.METRIC_LABELS` omits the EMA keys. One run recorded
  `best_total_source: ema` with an EMA metric of 0.613 while the UI showed
  0.602.
- **`LICENSES.md` names none of the 15 new runtime distributions.** All
  permissive, and `generate_licenses.py` fixes it — but it is a legal step, not
  a tidiness one.
- **The licence test guards on a substring.** Every config in the installed
  package reports `Apache-2.0`, including `RFDETRSegXLarge` and
  `RFDETRKeypointPreview`; the PML detection models are not in the package at
  all. The `"xlarge"` / `"2xl"` checks would reject correctly licensed models.
  The `license`-field check beside them is the reliable one.

## Sizes and tasks we are not taking

**Not Medium or Large.** Nano/Small/Medium/Large share one encoder and differ
in resolution and decoder depth. Small at 512 px already OOM'd on a 24 GB card
with another job on it, and needs more than the 8 GB Roboflow calls its
minimum. Nano already wins detection by 1.4-2.8x; a heavier variant buys
accuracy we do not need at a memory cost our users cannot pay.

**Not keypoints.** Upstream calls it preview, one size, COCO-person weights,
fine-tunable on any skeleton. The trainer would be cheap; the canvas is not.
`react-image-label`'s shape builder assumes one shape with one centre and
uniform handles, and per-joint dragging with stable identity, per-instance
grouping and visibility flags is new UI. `Project.labels` has nowhere to put a
joint-name list or an edge list. Handpose is not a precedent: its landmarks come
from mediapipe at upload time, one hand per image, stored as normalised
coordinates and never drawn or edited.

## Upstream advice the branch does not follow

Worth knowing before anyone tunes this. `grad_accum_steps: 4` is fixed in the
templates, so the dialog's batch of 8 is an effective batch of 32 against
Roboflow's recommended 16. Upstream asks for 100-200 epochs under 500 images;
the dialog default is far below that. And `skip_best_epochs` is 0, which
upstream warns can pin best-checkpoint selection to the pretrained weights.
