# Working list

Tracked here rather than in chat so it survives across sessions. Newest concerns
at the top of each section.

## In progress

- [ ] **Cross-platform packaging.** `.github/workflows/build.yml` builds Linux,
      Windows and macOS. macOS needed `imageio` restored for `.icns` conversion,
      and detectron2's `_C` had to be repointed at torch's dylibs (Nuitka
      resolves `@rpath` relative to the owning package, so it looked for
      `detectron2/libc10.dylib`). Windows and macOS have not yet produced a
      green run; Linux and Windows both got past the point macOS failed at.

      **Build first on Linux, locally.** It is free, and it is where the
          packaging problems are cheapest to find — the CI cache was pointed at the
          wrong directory and only a local run proved it. Spend CI minutes on macOS
          and Windows, which cannot be built here, one platform at a time.

### What a build costs

The repo is private, so Actions minutes are billed with GitHub's multipliers:
Linux 1x, Windows 2x, **macOS 10x**. Measured on the run cancelled on
2026-08-15:

| job           | wall    | billed                   |
| ------------- | ------- | ------------------------ |
| macOS (arm64) | 33 min  | 330                      |
| Windows       | 165 min | 332                      |
| Linux         | 165 min | 166                      |
|               |         | **828 min, no artefact** |

A cold all-platform build is ~2,340 billable minutes — more than the entire
2,000-minute Free monthly allowance, so a three-platform "loop until green" is
not affordable. Hence the `select` job: the branch name or the workflow input
picks one platform.

Two things make repeat builds cheaper, both in `build.yml`:

- Nuitka overrides `CCACHE_DIR` to `<its cache root>/ccache`, so caching
  ccache's default location stores nothing. Cache the whole Nuitka cache root.
- `module-cache` and `adapted_headers` in that root shorten the Python
  compilation phase that runs before any C is emitted — locally ~54 min cold
  versus ~13 min warm, a bigger saving than the C object cache alone.

GitHub allows 10 GB of cache per repository, so ccache is capped at 2.5 GB per
platform; its 5 GB default would make three platforms evict each other.

## Labelling canvas

- [ ] **Zoom rescales every shape.** `Director.zoom()` walks all elements and
      re-plots each one, and it mutates the stored coordinates rather than
      moving a viewport, so repeated zooming also accumulates float drift. The
      fix is a single transform (SVG `viewBox` or a root `<g>`), which is O(1)
      and leaves shape data untouched — but drawing reads `e.offsetX/offsetY`,
      so every builder would need to map through the CTM. Wheel events are now
      coalesced to one pass per frame, which caps the cost, but the underlying
      O(n) remains.

      Measured on `electron_microscopy_particle_segmentation` item 2 — 65
          polygons, 12,204 points: one zoom pass costs ~4.5 ms. That fits inside a
          16.7 ms frame, so zoom is smooth today; what coalescing removed was the
          several-passes-per-frame case a trackpad produces. It stops fitting at
          roughly 3-4x this shape count, which is when the viewport transform stops
          being optional.

## UI / UX review

Raised while trying the desktop app.

- [ ] **Component sizing is inconsistent.** Controls do not share a scale — button
      heights, input heights, icon sizes and paddings drift between views. Audit
      against a single size scale (shadcn's `sm` / `default` / `lg`) and make each
      component pick from it rather than setting one-off `h-*` / `px-*` values.
- [ ] **Colour usage needs a pass.** Check semantic tokens are used consistently
      (`primary`, `muted-foreground`, `destructive`) instead of literal greys like
      `text-gray-500`, which is what makes light/dark and future theming drift.
      `new-training-dialog.tsx` and the labelling bars are the obvious starting
      points.
- [ ] **Spacing rhythm.** Same idea for gaps and padding: pick a scale and hold to it.

Worth doing as one pass with a written convention, otherwise it re-drifts.

## Correctness / engineering

- [ ] **Test coverage.** First-party is ~53%. `routers/dataset.py` (431 uncovered)
      is the biggest single gap; `auto_labeling/` (372) needs the SAM ONNX models,
      which the suite deliberately does not download.
- [ ] **Trainer coverage.** `handpose_classification_trainer` 43%,
      `instseg_trainer` 33%, `nanodet_trainer` 47% — `prepare_data`/`train` paths
      need a project fixture in the database. These are the paths where both
      runtime bugs above hid: unit tests covered `prepare_config`'s output but
      nothing ran the trainers with a parameter set the UI can actually produce.
      Worth an integration test per project type that runs one epoch on a few
      images, since that is what found them.
- [ ] **`test_instseg_flow` stops at inference and never calls `export_onnx`.**
      That is the gap the Mask R-CNN export bug slipped through: export runs
      after training and gates model registration, so a flow test that ends at
      inference still passes while the user gets nothing. Adding the export
      step to that test hits a detectron2 tracing assert
      (`image_list.py:86`, `SystemError: __bool__ ... returned a result with an
    exception set`) that the real run does not, so the harness differs from
      the real path somewhere -- worth finding, since every other trainer's
      flow test does cover export. Covered for now by a unit test that pins
      `dynamo=False` (`test_instseg_export_uses_the_tracing_exporter`).

- [ ] **All ONNX export depends on a deprecated exporter, and cannot stop.**
      Every `torch.onnx.export` call passes `dynamo=False`. torch 2.9 made
      dynamo the default and deprecated the TorchScript path, so this is on a
      clock — but the packaged app has no way off it:

      - dynamo routes through `onnxscript`, whose `@script` decorator calls
            `inspect.getsource()` to parse a function's AST. A Nuitka-compiled
            binary has no source, so it raises "Decorator script does not work on
            dynamically compiled function". Observed in a real build.
          - detectron2's `TracingAdapter` additionally only works under tracing.

          When torch removes the TorchScript exporter, export breaks and takes model
          registration with it — a finished training run would be discarded. The way
          out is shipping `onnxscript` as real source alongside the binary rather
          than compiled, which needs `--include-data-dir` plus a path that makes it
          importable ahead of the compiled module table. Worth prototyping before it
          becomes urgent.

- [ ] **Training logs live in one growing text column.** Every line rewrites the
      whole `training_logs` value, so a run that logs per iteration gets slower
      as it goes. Measured, appending 16,000 lines (1.8 MB): 63s before the SQL
      append, 40s after -- better, but still superlinear, because SQLite rewrites
      the row either way. The fix is a `training_log_lines` table with one row
      per line, which also lets the UI tail the last N instead of refetching
      megabytes on every poll.

- [ ] **detectron2 CUDA extensions do not build against torch 2.11.** Its ATen
      headers are not nvcc-clean (`List_inl.h: need 'typename' before decltype`),
      with both gcc-13 and gcc-15. Currently built CPU-only, which is fine for
      Mask R-CNN (torchvision supplies the CUDA ops it actually uses) but means
      detectron2's own kernels are absent.
- [ ] **Nuitka `--nofollow-import-to=torch._dynamo`** works around a compiler crash
      on Nuitka 2.8.10. Retest whether 4.x still needs it.

## Done

- [x] Python 3.10 → 3.13, NumPy 1.26 → 2.4, torch 2.5.1 → 2.11, Lightning 1.9 → 2.6,
      smp 0.3.4 → 0.5, timm 0.9.7 → 1.0.28, FastAPI 0.96 → 0.141 (Pydantic v1 → v2)
- [x] Version single-sourced from `app_info.py`; bumped to 0.26.0
- [x] pywebview 6: update `webview.settings` in place instead of replacing it
- [x] NanoDet ONNX export ran in training mode (no `model.eval()`)
- [x] YOLO export silently dropped polygons; COCO polygon `area` was wrong
- [x] Unclosed SQLite engines in `migration_manager`
- [x] Circular import between the instance-seg factory and trainer
- [x] All five project types trained end to end on licence-cleared data, each
      registering a model (so ONNX export succeeded too):

      | Project type | Data | Result |
          |---|---|---|
          | Image Classification | chest X-ray (CC BY 4.0) | 78.4% val acc |
          | Object Detection (NanoDet) | helmet & jacket (Apache 2.0) | mAP 0.199, AP50 0.490 |
          | Image Segmentation (DeepLabv3) | EM particles (CC BY 4.0) | IoU 0.638 |
          | Instance Segmentation (Mask R-CNN) | EM particles (CC BY 4.0) | mAP@0.5 0.541 |
          | Handpose Classification | ASL letters (public domain) | 35.9% over 26 classes |

          Two runtime bugs only this surfaced: every trainer crashed on an unset
          pretrained-model selector, and Mask R-CNN's ONNX export picked the dynamo
          exporter. Both discarded a finished training run rather than failing early.
          `dental_segment` was deliberately *not* used for the instance-seg project:
          it has no recorded licence (see `anylearning-data/LICENSES.md`)

- [x] Real training on licence-cleared data: ResNet18 on `zhanglabdata_chest_xray`
      (CC BY 4.0). Batch 16 reached 78.4% validation accuracy vs 35.1% at batch 1,
      which is what confirmed the default batch-size fix
- [x] Labelling canvas rendered every image at 150px. The pane used `flex-1`
      inside a block container, so the class was inert and the pane took its
      height from its content — while the annotator sized the canvas from the
      pane. `absolute inset-0` gives it a definite size and breaks the cycle
- [x] Canvas refused to scale above 1:1, so a 640px image sat marooned in a
      2822×1445 pane. It now scales to fit in both directions
- [x] Wheel zoom needed Ctrl and moved in fixed 25% jumps; it now zooms on a
      plain wheel, scales with the delta, and applies once per animation frame
      instead of once per event
- [x] Panning is on middle-drag as well as Ctrl+drag, and `safe center` keeps a
      zoomed-in image reachable (plain centring put the overflow out of scroll
      range in both directions)
- [x] Shape fills were one constant colour, so classes were distinguishable only
      by a thin stroke; each class now fills with its own colour at 25% alpha
