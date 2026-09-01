# Model & dependency license policy

AnyLearning is distributed under **Apache License 2.0** and may also ship
third-party code and model weights inside desktop installers. Every bundled
component must be redistributable and compatible with the repository's license
and contributor expectations.

Anything added to `anylearning/training/models/`, `anylearning/configs/auto_labeling/models.yaml`,
or the dependency manifests must clear this policy first.

## The two-license rule

Code and weights are licensed separately, and the weights are usually the stricter half.
Always record **both**. A repository under Apache 2.0 tells you nothing about the
checkpoint hosted on Hugging Face.

## Tier A — approved, no review needed

Permissive use and redistribution allowed, with no incompatible copyleft reach
into Apache-2.0 source.

| Component                         | Code       | Weights                      | Use for                                                         |
| --------------------------------- | ---------- | ---------------------------- | --------------------------------------------------------------- |
| RF-DETR (Nano/Small/Medium/Large) | Apache 2.0 | Apache 2.0                   | detection, instance seg, keypoint _(Nano/Small shipping today)_ |
| D-FINE COCO-only checkpoints      | Apache 2.0 | Apache 2.0                   | detection; Objects365-derived variants are excluded             |
| YOLOX                             | Apache 2.0 | Apache 2.0                   | detection (CPU-friendly)                                        |
| `timm`                            | Apache 2.0 | per-model, mostly Apache 2.0 | classification backbones                                        |
| `segmentation_models_pytorch`     | MIT        | per-encoder                  | semantic segmentation                                           |
| Anomalib                          | Apache 2.0 | n/a (trains from your data)  | anomaly detection                                               |
| RTMPose / MMPose, MMRotate        | Apache 2.0 | Apache 2.0                   | pose, oriented boxes                                            |
| ByteTrack                         | MIT        | n/a                          | tracking                                                        |
| Grounding DINO, OWLv2             | Apache 2.0 | Apache 2.0                   | zero-shot pre-labeling                                          |
| EfficientSAM-Ti / S               | Apache 2.0 | Apache 2.0                   | lightweight promptable segmentation                             |
| EfficientViT-SAM                  | Apache 2.0 | Apache 2.0                   | fast CPU promptable segmentation                                |
| RTMDet through MMDetection        | Apache 2.0 | Apache 2.0                   | detection and oriented boxes                                    |
| Florence-2                        | MIT        | MIT                          | captioning / multi-task                                         |
| MobileSAM, SAM 2                  | Apache 2.0 | Apache 2.0                   | promptable segmentation _(shipping today)_                      |
| MediaPipe                         | Apache 2.0 | Apache 2.0                   | hand/pose landmarks _(shipping today)_                          |
| detectron2                        | Apache 2.0 | Apache 2.0                   | instance seg _(shipping today)_                                 |

D-FINE decision (2026-08-31): an official repository collaborator
[confirmed](https://github.com/Peterande/D-FINE/issues/357#issuecomment-5344034723)
that `*_coco.pth` checkpoints are Apache-2.0. The same response states that
`*_obj365.pth` and `*_obj2coco.pth` checkpoints are not commercially cleared
by that grant, so only COCO-only artifacts qualify for Tier A.

## Tier B — allowed only with sign-off

Additional terms travel with distributions and bind downstream users. These
require a written maintainer decision recorded in `LICENSES.md` before merge.

| Component                        | License             | The catch                                                                                                                                                                                                                           |
| -------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SAM 3 / SAM 3.1**              | Meta SAM License    | The SAM License must travel with the weights and adds field-of-use restrictions that Apache-2.0 does not.                                                                                                                           |
| **DINOv3** backbones             | Meta DINOv3 License | Commercial use allowed, but downloads are gated behind a Meta approval form requiring personal data. Mirroring the weights inside our installer needs a licensing read before we rely on it.                                        |
| **RF-DETR XL / 2XL** (detection) | Roboflow PML 1.0    | Not Apache, and PML 1.0 requires each user to hold a Roboflow platform plan. The repository depends only on the Apache-2.0 `rfdetr` package and not `rfdetr_plus`.                                                                  |
| **RapidOCR PP-OCR weights**      | Unconfirmed         | RapidOCR's engineering code is Apache-2.0, but its README identifies Baidu as the model copyright holder. Official ONNX URLs and hashes are reusable as download metadata; do not mirror the weights until their grant is recorded. |

Maintainer decision for SAM3 (2026-08-31): the ONNX runtime adapter is approved,
and a model bundle may be offered as an explicit optional/server-first download.
It must not enter the default package or installer; every mirror must identify
the artifact as `Meta SAM License`, include the complete license beside the
weights, preserve immutable source/export revisions and SHA-256 values, and
surface the separate terms before use. This approval does not relicense the
weights as Apache-2.0.

## Tier C — rejected for bundling or runtime dependencies

Do not copy or vendor these implementations, depend on them at runtime, bundle
their configuration, or distribute their weights. Do not add them as optional
extras or "just for benchmarking". A license-neutral adapter for a documented
public ONNX tensor contract is allowed when users supply their own artifact and
accept responsibility for its license; this does not make the implementation or
weights eligible for distribution by AnyLearning.

| Component                                                                | License                     | Why rejected                                                                                                                                                                                                      |
| ------------------------------------------------------------------------ | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ultralytics YOLO26 / YOLO11 / YOLOv8                                     | AGPL-3.0 or paid enterprise | AGPL terms are incompatible with distributing the combined work as Apache-2.0.                                                                                                                                    |
| YOLOv12 official implementation                                          | AGPL-3.0                    | Copyleft implementation and weights are not bundled; only its user-supplied ONNX tensor contract is supported.                                                                                                    |
| YOLOv10 official implementation                                          | AGPL-3.0                    | Copyleft implementation and weights are not bundled; only its user-supplied ONNX tensor contract is supported.                                                                                                    |
| YOLOv9 official implementation                                           | GPL-3.0                     | Copyleft implementation and weights are not bundled; only its user-supplied ONNX tensor contract is supported.                                                                                                    |
| YOLO-World                                                               | GPL-3.0                     | Copyleft, same problem.                                                                                                                                                                                           |
| EdgeSAM                                                                  | S-Lab License 1.0           | **Non-commercial only.** "Redistribution and use for non-commercial purpose… are permitted". Verified directly against the repository's LICENSE file. Use EfficientViT-SAM instead.                               |
| D-FINE Objects365 / Objects365+COCO checkpoints                          | Objects365 lineage          | The D-FINE maintainers explicitly state that their Apache-2.0 grant does not commercially clear `_obj365` or `_obj2coco` weights. Only the separately confirmed COCO-only checkpoints may be mirrored or offered. |
| SegFormer (original NVIDIA weights)                                      | NVIDIA Source Code License  | Non-commercial. The _architecture_ is fine to reimplement; those checkpoints are not.                                                                                                                             |
| Anything with a "research only" / "non-commercial" / "evaluation" clause | —                           | No exceptions without legal sign-off.                                                                                                                                                                             |

## Known grey area: COCO-pretrained weights

Nearly every detection and segmentation checkpoint we would ship is pretrained on COCO.
COCO _annotations_ are CC BY 4.0, but the underlying images are Flickr-hosted under a mix of
terms. The industry ships these weights routinely and we do too, but be honest that this is
convention rather than a clean grant. It is not a reason to prefer one Tier A model over
another — they all share it.

## Checklist for adding a model

1. Record the **code** license and the **weights** license separately.
2. Confirm redistribution in source and desktop installers is permitted.
3. Confirm no copyleft (GPL / AGPL / S-Lab / CC-_-NC / CC-_-SA on weights).
4. Check for downstream use restrictions that bind our customers (Tier B).
5. Append the full license text to `LICENSES.md` and attribute in-app where required.
6. Add the component to the table above.

## Datasets

Test and example datasets follow the same rule. Three sources, in order of how
much the test run depends on them:

1. **Synthetic fixtures** (`tests/fixtures/datasets.py`) — generated procedurally,
   so no licence at all, no network, deterministic. These back the always-on
   end-to-end tests, which keeps the default run offline.
2. **The `anylearning-data` repository** — real datasets, audited in that repo's
   `LICENSES.md`. The public examples also include generated stick figures,
   attributed vertebral X-rays and Apache-2.0 Desert Locust keypoints. Only the
   core real-world smoke datasets are exposed through
   `REAL_DATASETS`: chest X-ray (CC BY 4.0), helmet/jacket (Apache 2.0), particle
   segmentation (CC BY 4.0), and ASL (Public Domain). `neu_surface_defect` and
   `dental_segment` record **no licence** and are deliberately excluded until that
   is resolved — `tests/e2e/test_real_datasets_e2e.py` asserts they stay out.
3. **Oxford-IIIT Pet** (CC BY-SA 4.0) — opt-in download, cached **outside** the
   repository so its ShareAlike term never reaches our source tree.

Note the asymmetry: CC BY only asks for attribution, but CC BY-SA propagates to
derivatives. That is why the ShareAlike dataset is fetched rather than committed.
