# Testing

```bash
./run_tests.sh              # everything, with coverage
pytest tests/               # everything, faster
pytest tests/e2e/           # training / export smoke tests only
```

The default run is **fully offline and needs no dataset download**. That is
deliberate: AnyLearning is an offline product, and a test suite that quietly
depends on the network hides breakage that users will hit.

## Layout

| Path                                                                             | What it covers                                                                     |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `tests/nanodet/`, `tests/semantic_segmentation/`, `tests/instance_segmentation/` | unit tests for the vendored model code                                             |
| `tests/app/`, `tests/database/`, `tests/config/`, `tests/licenses/`              | API, persistence, config, and third-party model licensing                          |
| `tests/training/`                                                                | device placement and checkpoint/export helpers                                     |
| `tests/fixtures/datasets.py`                                                     | dataset builders and loaders (below)                                               |
| `tests/e2e/test_training_e2e.py`                                                 | classification and segmentation: data → training → checkpoint → ONNX → onnxruntime |
| `tests/e2e/test_all_flows_e2e.py`, `tests/e2e/test_rfdetr_e2e.py`                | the real training flows, including alternate RF-DETR architectures                 |
| `tests/training/test_rfdetr_keypoint_trainer.py`                                 | keypoint COCO schema, RF-DETR config, metrics and inference contract               |
| `tests/test_prepare_vertebral_keypoints.py`                                      | mixed JSON/YOLO-pose normalization for the licensed vertebral dataset              |
| `tests/e2e/test_real_datasets_e2e.py`                                            | the same flows on the licence-cleared real datasets                                |
| `smoke_test_build.sh`, `smoke_test_training.py`                                  | the _packaged_ binary (below) — not pytest                                         |
| `realworld_keypoint_test.py`                                                     | opt-in packaged RF-DETR train/export/inference on held-out real X-rays             |

## Every flow, every variant

The e2e modules drive the established project types through their real training
entry points. RF-DETR keypoint additionally runs through `--self-test`, because
its starting checkpoint is bundled with the application rather than the source
checkout:

| Project type            | Variants covered                                  |
| ----------------------- | ------------------------------------------------- |
| Image Classification    | ResNet18, ResNet34                                |
| Image Segmentation      | DeepLabV3+ over ResNet18 / 34 / 50                |
| Object Detection        | NanoDet lightweight / medium / large              |
| Instance Segmentation   | Mask R-CNN ResNet50 / ResNet101                   |
| Handpose Classification | MLP small / medium / large                        |
| Keypoint Detection      | RF-DETR Keypoint Preview (packaged `--self-test`) |

Each task also asserts its variant list against `anylearning/config.py`, so adding
a variant to the UI without adding it here fails the suite rather than shipping
an untested option.

Every variant runs the **whole job**, not just training:

```
data -> train_fn -> checkpoint -> ONNX export -> onnxruntime -> run_inference
```

Both ends matter. ONNX export is the last step of every real training job, and
`run_inference` is what the app calls when a user tests a model on an image —
neither was covered before, and both were broken in ways training alone could not
reveal (see `docs/dependency_upgrade.md`).

**Two globals need resetting between variants.** NanoDet's `cfg` is a module-level
CfgNode that `load_config` _merges_ into, and detectron2 registers datasets in a
process-global `DatasetCatalog` under fixed names. Training two variants in one
process therefore leaks state — the `pristine_nanodet_cfg` and
`clean_detectron2_catalog` fixtures undo it. Production is unaffected because
`routers/training.py` runs each job in its own `multiprocessing.Process`; this is
purely a consequence of testing several variants in a single process.

## Test data

Three sources, chosen so licence obligations never leak into the repository.
See `docs/model_license_policy.md`.

**Synthetic (default).** `build_classification_dataset`, `build_segmentation_dataset`,
and `build_detection_coco` generate coloured shapes in AnyLearning's native
on-disk formats. Generated data carries no licence, needs no network, and is
deterministic for a given seed — `test_fixture_data_is_deterministic` asserts
byte-identical output, because a fixture that drifts makes "reproducible" runs a
lie.

Each class gets a distinct shape _and_ colour so a small model can actually
separate them. A smoke test that cannot reduce its loss only proves the code did
not crash.

**Real datasets (optional).** Clone
[anylearning-data](https://huggingface.co/datasets/nrl-ai/anylearning-data) next to this
repository, or point `ANYLEARNING_DATA_DIR` at it:

```bash
hf download nrl-ai/anylearning-data --repo-type dataset --local-dir ../anylearning-data
pytest tests/e2e/test_real_datasets_e2e.py
```

Without it those tests skip rather than fail. Only licence-cleared datasets are
exposed through `REAL_DATASETS`; `neu_surface_defect` and `dental_segment` record
no licence and are excluded until that is resolved.

**Vertebral keypoints (opt-in download).**
`prepare_vertebral_keypoints.py --download` fetches the fixed CC BY 4.0 DOI
version, verifies its SHA-256, normalizes its mixed annotation formats, and
writes upload-ready COCO keypoint archives. `realworld_keypoint_test.py` drives
those archives through a packaged build. Exact provenance, commands, counts,
and the measured short-run result are in `docs/keypoint_realworld_validation.md`.

Archives are extracted once into `~/.cache/anylearning-test-data/` (override with
`ANYLEARNING_TEST_DATA`) — never inside either repository.

**Oxford-IIIT Pet (opt-in download).** `fetch_oxford_pet(download=True)`. CC BY-SA
4.0, so it is fetched rather than committed: ShareAlike propagates to derivatives.

## Testing a packaged build

None of the above runs against the thing users install. Nuitka failures do not
look like test failures: the suite stays green while the binary is broken.
Observed, in one afternoon — a binary that compiled, linked, and then died on
startup because `torchvision.ops` imported an excluded module; another that
segfaulted silently because a module built by `clone_module()` was dropped; and
a build that packaged a frontend from the previous day.

```shell
bash smoke_test_build.sh ./app.dist/app.bin       # starts it, checks API, routes, frontend
python smoke_test_training.py ./app.dist/app.bin  # every project type, GPU and CPU
```

`/api/health/imports` does the same probe from inside the running process,
importing torch, `torchvision.ops`, cv2, onnxruntime, Lightning and all five
trainers, and reporting per-module rather than dying on the first failure.
`tests/routers/test_health_router.py` covers it, including that it survives
reporting a failure rather than 500ing.

Two rules learned the hard way:

- **A smoke test that cannot fail is not a smoke test.** The original ran the
  binary with `|| echo "::warning::"`, so a startup crash produced a green run
  and a published artefact.
- **Check exit codes without a pipe.** `binary --help | tail` reports `tail`'s
  status; it briefly convinced me a segfaulting binary was fine.

## Conventions worth keeping

- **Don't index test data by position.** `dataset[1]` silently depended on
  `os.listdir` ordering, which varies by filesystem. Look items up by name.
- **Don't hand epoch-end hooks an empty list to make them pass.** Doing that in
  the NanoDet test meant the evaluation branch was never entered, hiding both a
  missing evaluator and a leaked file handle.
- **Keep `pretrained: None` in test configs** so runs stay offline.
