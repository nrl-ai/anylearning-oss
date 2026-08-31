# AnyLearning Platform Roadmap

Status: active implementation plan
Last updated: 2026-08-31
Primary repository: `nrl-ai/anylearning-oss`
Integration branch: `develop`

## Executive decision

AnyLearning becomes the primary product and platform for labeling, training,
evaluation, model export, and local or shared inference. AnyLabeling remains a
maintained GPL desktop client and gains compatibility with AnyLearning through a
versioned remote-inference protocol. New product breadth should not be built
twice in Qt and React.

The first shared boundary is named `anylearning.inference`, not
`anylearning-core`. It has a specific purpose and must not become a home for
unrelated desktop, project, or training code.

The implementation order is:

1. Protect current correctness, package integrity, and cross-platform behavior.
2. Establish lightweight inference contracts and strict dependency boundaries.
3. Move existing SAM inference behind the boundary without behavior changes.
4. Add a unified ONNX model registry and YOLO detection/segmentation adapters.
5. Add bounded batch inference and embedding precomputation.
6. Add a production `anylearning serve` profile with password authentication.
7. Connect AnyLabeling through the public protocol.
8. Expand annotation workflows only after their performance and persistence are
   proven.

No package, installer, model, container, or dataset is released merely because
its implementation PR merges. Releases require explicit approval.

## Product and repository ownership

| Concern                                               | Owner                     |
| ----------------------------------------------------- | ------------------------- |
| Labeling, training, evaluation, deployment UX         | AnyLearning               |
| Neutral inference contracts and model backends        | `anylearning.inference`   |
| Training implementations and export                   | `anylearning.training`    |
| Authenticated shared inference                        | `anylearning.server`      |
| Desktop-only FastAPI routes and webview integration   | AnyLearning desktop       |
| AnyLabeling critical maintenance and interoperability | AnyLabeling               |
| AnyLabeling remote inference integration              | AnyLabeling `RemoteModel` |

AnyLearning is Apache-2.0. Feature ideas and wire behavior may be independently
reimplemented, but GPL implementation code must not be copied into AnyLearning
unless the specific code has compatible provenance and licensing. Apache-2.0
AnyLearning components may be consumed by GPLv3 AnyLabeling.

## Highest priorities: performance, stability, and security

Performance, stability, and security are co-equal P0 requirements and outrank
model count and feature breadth. A feature is not merge-ready merely because it
works once. It must remain responsive, bound memory and concurrency, recover
from failure, release resources, preserve existing results, and maintain every
applicable security invariant.

Every relevant PR must provide a reproducible before/after measurement on the
same machine. Until stable baselines exist, report measurements without
inventing thresholds. Once a benchmark has a stable baseline, an unexplained
regression above 5% blocks merging.

Required measures, where applicable:

- CLI import and startup time.
- Time to an interactive desktop window.
- Image open and previous/next latency at p50 and p95.
- Canvas pan, zoom, and edit frame time at representative shape and point counts.
- Model cold load, first inference, warm inference, and unload time.
- Batch throughput, queue latency, and cancellation latency.
- Peak resident memory and memory growth during repeated work.
- GPU memory release, worker/thread cleanup, open file handles, and temporary
  artifact cleanup.
- Failure followed by a successful retry in the same process.
- Soak behavior for lifecycle, navigation, batch, cache, database, and server
  changes.

No percentage allowance applies to correctness failures, unbounded growth,
stale results, UI-thread inference, leaked workers, credential exposure,
authorization bypass, unsafe parsing, or silent data loss. Performance work may
not disable validation, authentication, isolation, redaction, or resource limits.

### Baseline program

1. Build a deterministic, redistributable benchmark corpus with provenance and
   checksums.
2. Publish versioned test data and model fixtures under the appropriate
   `nrl-ai` or `vietanhdev` Hugging Face namespace only after license review.
3. Add a headless benchmark command that emits versioned JSON.
4. Record Linux devbox, Apple Silicon macOS, and Windows baselines using fixed
   source, model, and dataset revisions.
5. Retain benchmark JSON as PR/CI artifacts and compare stable metrics.
6. Keep timing tests outside the deterministic unit suite.

### Security program

Treat images, annotations, archives, model files, checkpoints, configuration,
URLs, HTTP requests, database imports, and project metadata as attacker-controlled
at every trust boundary.

1. Expand `SECURITY.md` into a repository threat model covering the desktop
   loopback API, public inference server, website, project/model/data ingestion,
   training workers, build pipeline, release artifacts, and Hugging Face assets.
2. Define and test security invariants: authentication before protected work,
   project/request isolation, bounded parsing and execution, safe path handling,
   fail-closed authorization, secret redaction, and integrity-checked artifacts.
3. Add size, depth, count, decompression-ratio, time, memory, and concurrency
   limits before parsing or executing untrusted content.
4. Reject archive traversal, absolute paths, unsafe links, device files, and
   extraction outside a newly created destination. Test ZIP and TAR variants.
5. Do not deserialize untrusted pickle-based checkpoints or execute model-supplied
   Python. Prefer safe tensor/ONNX formats, integrity hashes, explicit provenance,
   and isolated conversion workflows.
6. Treat model inference and training as resource-exhaustion boundaries. Apply
   worker isolation, cancellation, deadlines, bounded output, and cleanup after
   crashes or forced termination.
7. Keep the desktop API loopback-only with a high-entropy header token. Remove
   credentials from URLs, narrow CORS, and never return filesystem paths or
   tracebacks outside an explicit local debug profile.
8. Review dependency and supply-chain alerts for realistic reachability. A
   reachable critical/high vulnerability blocks release; exceptions require a
   written owner decision, compensating controls, and an expiry date.
9. Generate an SBOM and provenance for installers, containers, Python packages,
   model weights, and benchmark data. Pin or hash sensitive build inputs and
   verify downloaded artifacts before use.
10. Keep automated secret scanning, dependency review, static checks, security
    regression tests, and targeted fuzz/property tests in CI.
11. Document vulnerability reporting, supported versions, coordinated disclosure,
    response ownership, and credential/model revocation procedures.
12. Perform a focused threat review whenever a PR adds a network endpoint,
    parser, archive format, model/backend, subprocess, file-write path, credential,
    or release/deployment mechanism.

## Architecture

```text
AnyLearning desktop ------------------+
                                      |
anylearning.server -------------------+--> anylearning.inference
                                      |      - contracts
optional AnyLabeling local adapter ---+      - model registry
                                             - lifecycle
AnyLabeling RemoteModel -- HTTP --> server   - backend adapters

anylearning.training --> artifact contracts and inference validation
```

### `anylearning.inference`

This is a headless library boundary inside the existing distribution first. It
contains:

- Protocol version and capability schemas.
- Neutral points, prompts, shapes, prediction results, and timing metadata.
- Model configuration validation.
- Model registry and backend interfaces.
- Session load, cache, inference, cancellation, and unload lifecycle.
- ONNX preprocessing and postprocessing shared by desktop and server.
- Deterministic contract fixtures and compatibility tests.

It must not contain:

- React, Qt, pywebview, or window-control code.
- FastAPI routes, authentication, or HTTP transport.
- Project databases or desktop settings.
- Training implementations, augmentations, optimizers, or checkpoints.
- Unconditional imports of PyTorch, ONNX Runtime, OpenCV, FastAPI, or desktop
  frameworks from the package root.

### `anylearning.training`

Training remains separate and may depend on lightweight inference/artifact
contracts. It owns trainers, data augmentation, subprocess/GPU orchestration,
metrics, checkpoints, evaluation, and ONNX export. Training must remain an
optional installation capability and must never become a server or inference
client requirement.

## Model support strategy

Add model families by reusable task contracts and decoders, not by copying a
large model zoo. Every adapter must pass correctness, performance, lifecycle,
security, packaging, and separate code/weight license gates.

### Mandatory ONNX inference boundary

Every new model integration must execute an ONNX artifact through
`anylearning.inference`. Framework-native checkpoints may exist in
`anylearning.training` for training and export, but they are not accepted by the
desktop or server inference path. Conversion is an explicit, isolated workflow;
inference never loads pickle checkpoints, imports a model's training framework,
or executes repository-supplied Python.

Each adapter must define a versioned ONNX input/output profile, supported opset,
static production shapes, preprocessing, postprocessing, and an exporter parity
test. Prefer an official ONNX exporter or official pre-exported artifact. A
third-party exporter is only a research lead: it must be independently
reimplemented or pinned, license-reviewed, and proven against the native model
before it becomes a supported source. See
[`docs/onnx_model_sources.md`](docs/onnx_model_sources.md) for the researched
source matrix and current gates.

ONNX artifacts larger than the single-protobuf limit may use external tensor
data only inside an integrity-addressed bundle manifest. The manifest enumerates
every relative file path, byte size, and SHA-256; downloads land in a private
staging directory and become visible atomically only after all files verify.
Absolute paths, parent traversal, links, undeclared files, and arbitrary external
references remain rejected. The shared zero-copy bundle loader and generic YOLO
wiring are implemented; their Linux, Windows, and macOS real-model gates remain
mandatory for every runtime/provider change.

### SAMExporter compatibility audit

The complete SAMExporter model matrix at commit
[`35133ce`](https://github.com/vietanhdev/samexporter/commit/35133ce8670e0d190ac10cc08efba9b9a443fb51)
was audited on 2026-08-30. AnyLearning should consume the documented ONNX graph
contracts through first-party `anylearning.inference` adapters; it must not add
SAMExporter as an inference dependency.

| SAMExporter family                   | AnyLearning decision                                          | Required inference work                                                                                                                                                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SAM ViT-B/L/H and quantized variants | Update the existing `segment_anything` backend                | Accept both embedded-preprocessing HWC encoders and normalized/padded NCHW encoders; use the official longest-side resize; select `low_res_masks` when exported and apply the aspect-ratio-aware resize/crop/resize path. Remove the fixed MobileSAM input geometry. |
| MobileSAM                            | Reuse the corrected SAM backend                               | Prove parity with its encoder/decoder pair; do not maintain a second preprocessing path.                                                                                                                                                                             |
| EfficientSAM-Ti/S                    | Add an `efficient_sam` backend                                | Implement its distinct dynamic NCHW RGB `/255` encoder and batched point/box decoder contract; validate names/ranks/dtypes and choose the highest-IoU mask candidate per query. This is not the same architecture or graph contract as EfficientViT-SAM.             |
| SAM2 Tiny/Small/Base+/Large          | Keep the existing SAM2 backend                                | Preserve AnyLearning's RGB input correction, add explicit provider selection and safe graph inspection/loading, and validate all four encoder/decoder sizes with real artifacts.                                                                                     |
| SAM2.1 Tiny/Small/Base+/Large        | Reuse the SAM2 backend                                        | Treat the revision and artifact pair as model identity; no separate preprocessing implementation is needed.                                                                                                                                                          |
| SAM3 ViT-H                           | Add a separate `sam3` backend after the smaller families pass | Add a bounded `TextPrompt` wire contract and a three-graph image/language/decoder session; validate its fixed geometric-prompt capacity, raw query outputs, confidence filtering, mask-IoU NMS, instance limits, scores, and multi-instance editable shapes.         |

Before any SAM family can be exposed by `anylearning.server`, every graph in its
pair or triplet must use the shared integrity-addressed external-data loader.
Model revision identity covers every graph and external tensor file. Session
creation must set an explicit provider order and enforce graph input/output
names, ranks, dtypes, static bounds, maximum prompts, maximum masks, and maximum
result points before expensive inference or postprocessing.

Prompt validation is shared across the family: point labels are exactly 0 or 1,
box prompts have positive area, coordinates are finite, and empty geometric
prompts are accepted only by a backend such as SAM3 that explicitly supports a
text-only request. SAM3 text is length-bounded before tokenization and its
configured ONNX token capacity is enforced.

The SAMExporter runtime code is MIT, while graph weights retain their upstream
license independently. SAM, MobileSAM, EfficientSAM, and SAM2/2.1 artifacts may
only be published after their Apache-2.0 provenance and checksums are recorded.
SAM3 weights retain the separate Meta SAM license, are optional/server-first,
and are never bundled into the default Apache-2.0 package. The license must ship
beside any published artifact bundle and be shown before download.

Implementation and real-model gates proceed in this order:

1. Extend safe multi-graph loading and model-revision hashing to SAM pairs and
   triplets without copying multi-gigabyte external tensors into Python memory.
2. Correct SAM/MobileSAM preprocessing and postprocessing, with point and box
   parity on landscape, portrait, and extreme-aspect images.
3. Add EfficientSAM-Ti, then EfficientSAM-S, with native/export parity and
   retained visual reports.
4. Validate the existing SAM2 backend against all SAM2 and SAM2.1 sizes, using
   Tiny as the cross-platform merge gate and the larger variants as scheduled
   resource-qualified jobs.
5. Add SAM3 text, point, box, and combined-prompt inference as a separately
   reviewed change because its contract, license, memory use, and multi-instance
   postprocessing differ materially.

Implementation status (2026-08-31):

- [x] Paired graphs use the bounded stable ONNX loader, independent graph and
      external-data manifests, explicit providers/thread limits, and pair-bound
      model revisions. Per-session CPU arenas default off after real lifecycle
      tests showed materially lower peak and retained RSS without changing
      deterministic outputs or warm prompt latency; throughput deployments may
      opt in after profiling.
- [x] SAM/MobileSAM now support HWC embedded preprocessing and raw NCHW
      preprocessing, official longest-side resize, low-resolution aspect-aware mask
      postprocessing, and highest-IoU candidate selection.
- [x] EfficientSAM-Ti/S has a distinct `efficient_sam` backend with dynamic
      native RGB input, allocation limits, named graph contracts, embedding cache,
      and authenticated-server support. The real merge gate uses Ti; S remains a
      scheduled artifact-size variant of the same graph contract.
- [x] Real EfficientSAM-Ti, MobileSAM, SAM 2 Tiny, and SAM 2.1 Tiny point/box
      runs pass in landscape and portrait orientations, both in-process and through
      authenticated HTTP. Reports, timings, RSS measurements, digests, and reviewed
      images are retained under `validation-results/` locally.
- [x] The hosted workflow defines Linux, Windows, and macOS real-model gates for
      all four smallest redistributable pairs, with immutable downloads and retained
      artifacts. The complete smallest-model matrix has passed on all three hosted
      operating systems.
- [x] Scheduled resource-qualified SAM ViT-B and EfficientSAM-S runs pass on
      Linux, Windows, and macOS with stable repeated lifecycle output, retained
      visual evidence, and bounded memory growth.
- [x] SAM2 Small/Base+/Large passed the exact-head hosted in-process and
      authenticated-server matrix on Linux, Windows, and macOS. All 36 annotated
      image pairs were pixel-identical between transports, and normalized model
      output matched across operating systems.
- [x] SAM2.1 Tiny/Small/Base+/Large passed the exact-head hosted in-process and
      authenticated-server matrix on Linux, Windows, and macOS. All 24 jobs and
      96 retained images passed with identical inference-semantic prediction
      sequences and decoded pixels across transports and operating systems.
- [x] SAM2 and SAM2.1 encoder bundles were deterministically normalized so their
      ONNX metadata no longer reports stale convolution output shapes or unused
      initializers. Exact-result parity was proven before the pinned artifacts
      were replaced. Tiny/Small/Base+/Large then passed direct and authenticated
      server inference on Linux, Windows, and macOS with cross-platform decoded
      pixel identity, visual review, and bounded post-warmup lifecycle growth.
- [x] SAM3 has a separate licensed three-graph backend with bounded text,
      point, box, and combined prompts; fixed-capacity contract validation;
      independent graph/external hashes; bounded mask-IoU NMS and editable
      multi-instance shapes; authenticated-server support; and explicit
      multi-gigabyte unload reclamation. Real local in-process and authenticated
      HTTP validation passes with retained visual reports. Its Linux, Windows,
      and macOS job is scheduled/manual rather than a per-PR download.
- [x] EfficientViT-SAM has a distinct `efficientvit_sam` ONNX backend with
      checksum-gated deterministic decoder preparation, native multimask policy,
      512/1024 encoder profiles, 1024-frame prompt scaling, point-only padding,
      resource bounds, embedding cache, and authenticated-server support. L0
      native-checkpoint parity and all five variants' local direct/server
      landscape/portrait visual validation pass. License-complete, checksum-
      enumerated bundles are published at immutable model revision
      `e5848b5d032cc9b5f3a3af199325005c45e24b50`. Exact head
      `489010c0ff58d557ceb660f9df33d25162c520f9` passed all 15 hosted variant/OS
      jobs plus aggregate transport and cross-platform decoded-pixel identity
      gates in runs `33347664285` and `33347671140`. All 120 retained renderings
      are pixel-identical where required, and every unique landscape/portrait
      point/box result passed visual review before PR #30 merged.
- [x] RF-DETR Nano detection and instance segmentation use a standalone
      `rfdetr_onnx` backend with strict static graph contracts, checksum-gated
      ONNX loading, sparse class slots, bounded top-k outputs, editable instance
      polygons, lifecycle reclamation, and authenticated-server support. Official
      `rfdetr==1.9.4` exports are published with license and provenance records at
      immutable model revision `dbe812f210253e50910eb26e465618e62b379111`.
      Exact head `36adee77d92a89b523f8803cc31209e1d43c8835` passed all six
      detection/segmentation jobs on Linux, Windows, and macOS plus aggregate
      transport and cross-platform checks in run `33359361230`. Prediction
      digests are exact across transports and operating systems; the only visual
      drift is 2-3 macOS renderer pixels at channel delta 1, inside the explicit
      16-pixel/delta-2 ceiling. All 24 retained outputs passed visual review,
      steady-state RSS growth stayed bounded, and PR #34 merged as
      `2fc72ad27348977ffcb88e4b5c267a7b960b888b`.

Every merge gate uses real ONNX graphs and real images, never mocked model
outputs. It records graph and image SHA-256 values, cold/warm latency, peak RSS,
output summaries, transport-sensitive consistency digests, inference-semantic
prediction digests, and annotated images under a retained validation-results
artifact. Prediction digests deliberately omit request IDs, source-file IDs, and
timings so equal model output can be compared directly across operating systems.
At least one landscape and one portrait result per family receive
visual inspection. Linux, Windows, and macOS run the smallest redistributable
artifact for each graph contract; multi-gigabyte variants run on resource-tagged
workers with explicit memory evidence.

| Order | Model capability                                                          | Distribution policy                                                      | Reason                                                            |
| ----- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| 1     | ONNX models trained/exported by AnyLearning                               | Built in                                                                 | Close the train-to-prelabel loop with artifacts we already create |
| 2     | User-supplied YOLOv5/v8/v9/v10/v11/v12/26 detection and segmentation ONNX | Generic decoder only; no external implementation code or weights bundled | Highest custom-model interoperability value                       |
| 3     | YOLOX ONNX detection                                                      | Neutral ONNX adapter implemented; benchmark before offering weights      | Apache-2.0, CPU-friendly detector                                 |
| 4     | SAM/MobileSAM compatibility, EfficientSAM, SAM2/2.1 validation, then SAM3 | Per-family policy in the audit above                                     | Close verified gaps in the maintained ONNX exporter we own        |
| 5     | EfficientViT-SAM encoder/decoder ONNX                                     | Permissive optional/bundled weights after review                         | Faster promptable segmentation on CPU                             |
| 6     | RF-DETR ONNX detection/instance segmentation                              | Reuse Apache-designated training artifacts                               | Accurate permissive family and existing AnyLearning investment    |
| 7     | D-FINE ONNX detection                                                     | COCO-only weights after redistribution confirmation                      | Official exporter; do not treat it as a segmentation model        |
| 8     | ONNX detector-to-SAM refinement                                           | Workflow over existing adapters                                          | Converts detector boxes into editable high-quality masks          |
| 9     | OWLv2 ONNX, then Grounding DINO ONNX                                      | Optional/server-first; Grounding DINO export remains gated               | Zero-shot prelabeling with a verified ONNX path first             |
| 10    | OBB through a permissive RTMDet ONNX backend                              | Only after OBB editing and export are complete                           | Geometry must be correct before inference                         |
| 11    | RapidOCR PP-OCR ONNX                                                      | Optional/server-first until workflow demand is validated                 | Permissive OCR without a new training framework                   |

Ultralytics YOLO implementations and distributed weights remain rejected from
the Apache-2.0 product under the current license policy. Supporting a documented
neutral tensor layout for a user-supplied ONNX file does not authorize bundling
Ultralytics code, configuration, or weights. The UI must identify custom artifacts
as user-provided and must not imply that AnyLearning grants rights to them.

Long-tail, proprietary, research-only, or framework-heavy families should
implement the public server protocol rather than expanding mandatory desktop
dependencies. Models with non-commercial or unclear terms are not accepted even
as optional examples or benchmark assets.

### Definition of done for a model family

- Code, weights, training data provenance, and redistribution terms are recorded
  separately.
- Inference accepts ONNX only; native checkpoint conversion stays outside the
  inference process and produces an integrity-addressed artifact.
- Official exporter output is compared against the native reference on a fixed
  corpus with recorded tolerances before its framework dependency is removed.
- Configuration and tensor layouts are validated with actionable diagnostics.
- Preprocessing and postprocessing use deterministic golden fixtures.
- Malformed models, extreme shapes, external-data paths, and oversized outputs
  fail within resource limits and without reading unintended files.
- Cold load, first/warm inference, unload, peak RSS, and repeated lifecycle
  measurements are recorded.
- Failure, retry, cancellation, and provider fallback are tested.
- Results preserve request, source, model, revision, confidence, and provenance.
- Shapes remain editable and survive save/reload plus relevant exports.
- CPU behavior works; accelerated behavior is verified or explicitly unsupported.
- Source and packaged-app behavior pass on supported operating systems.
- Server concurrency is tested if the backend is remotely exposed.

### Installation profiles

Keep the public `anylearning` distribution and import namespace. Move toward:

```text
pip install anylearning
pip install "anylearning[inference]"
pip install "anylearning[inference-gpu]"
pip install "anylearning[training]"
pip install "anylearning[server]"
pip install "anylearning[desktop]"
pip install "anylearning[all]"
```

The first packaging PRs must preserve the existing complete desktop install.
Dependencies should be separated only with import-boundary tests, clean-environment
installation tests, documented migration, and packaged-application verification.
Do not turn the current mandatory dependency list into extras in one unverified
rewrite.

Expected commands:

```text
anylearning infer
anylearning train
anylearning serve
anylearning desktop
```

## `anylearning.server`

The current desktop `--server` mode is not a production network service. It
contains desktop/project/window routes and assumptions that are inappropriate
for remote exposure. Build a separate application factory and CLI profile with
a narrow `/v1` inference API.

Initial endpoints:

```text
GET    /v1/health
POST   /v1/auth/token
GET    /v1/models
GET    /v1/models/{model_id}
POST   /v1/predictions
GET    /v1/predictions/{request_id}
DELETE /v1/predictions/{request_id}
```

Server requirements:

- Argon2id password-hash generation command.
- Password hash supplied by a secret manager/environment variable.
- Short-lived signed bearer tokens; passwords never enter URLs or persistent
  client configuration.
- Login rate limits and temporary backoff.
- TLS verification by default, normally via a documented reverse proxy.
- Narrow CORS policy; no wildcard production policy.
- Safe error bodies without paths, source files, tracebacks, secrets, or image
  content.
- Request/image/result size limits, timeouts, queue limits, and bounded
  concurrency.
- Shared model sessions with per-backend concurrency safety.
- Request IDs, structured logging, mandatory redaction, and graceful shutdown.
- A documented single-host container deployment.

Desktop webview tokens and production server tokens are separate mechanisms.
Desktop development-mode authentication bypasses must never affect the server
application.

## AnyLabeling integration

Integrate remotely before embedding Python code:

1. Publish protocol schemas and golden request/result fixtures.
2. Add a mock server contract suite.
3. Implement AnyLabeling `RemoteModel` capability discovery and prediction.
4. Add timeout, cancellation, TLS, error classification, and OS credential
   storage.
5. Run the same fixture through AnyLearning desktop, server, and AnyLabeling.
6. Consider optional local `anylearning[inference]` integration only after
   dependency, startup, memory, and packaging benchmarks demonstrate value.

The protocol exchanges image content or explicit supported object references,
not server-local project database IDs. Results always include request ID,
source identity, protocol version, model ID/revision, neutral shapes, warnings,
and timings so clients can reject stale results.

## Milestones and PR sequence

### P0: Foundation, reliability, and measurement

F1. Add this roadmap, branch policy, and no-release rule.
F2. Add lightweight versioned inference contracts with wire round-trip tests.
F3. Add import-boundary and import-time/RSS measurements.
F4. Record desktop and model lifecycle baselines on Linux, macOS, and Windows.
F5. Fix or explicitly disposition the existing cross-platform packaging gaps.
F6. Increase first-party coverage around routers, auto-labeling, lifecycle, and
failure recovery.
F7. Create benchmark/test artifacts and publish them with provenance.

### P0: Security foundation

SEC1. Expand the repository security policy and threat boundaries.
SEC2. Triage current dependency alerts by reachability and patch supported
frontend/website dependencies without hiding incompatible ML runtime changes.
SEC3. Harden the desktop loopback API: header-only high-entropy tokens, narrow
CORS, loopback enforcement, and production-safe error responses.
SEC4. Add adversarial archive, image, annotation, configuration, and path tests.
SEC5. Audit model/checkpoint loading for unsafe deserialization, external-data
paths, unbounded tensors, downloads, and missing integrity verification.
SEC6. Add SBOM, build provenance, dependency review, and artifact checksum gates.
SEC7. Threat-model and security-test the inference server before enabling any
non-loopback bind.

### P0: Inference platform

I1. Define backend, session, cancellation, capabilities, and registry interfaces.
I2. Correct and harden the existing SAM/MobileSAM adapter, preserve the verified
SAM2 RGB behavior, and load all graph pairs through the shared safe ONNX path.
I3. Add stable image identity and embedding cache keys including model revision.
I4. Add real SAM/MobileSAM/SAM2 visual parity and load/infer/unload soak tests,
then add EfficientSAM and SAM3 according to the audited family gates above.
I5. Implement a reusable YOLO decoder with tensor-layout diagnostics.
I6. Add YOLOv5/v8/v9/v10/v11/v12/26 detection and segmentation using raw and
end-to-end ONNX output profiles.
I7. Add confidence, IoU, class filters, dynamic shapes, and provider diagnostics.
I8. Connect AnyLearning-trained artifacts to auto-labeling through the same
contracts.
I9. Benchmark the ONNX-only YOLOX adapter, close the audited SAMExporter matrix,
then add EfficientViT-SAM, RF-DETR, and D-FINE backends in that order, enabling
only those that satisfy the model and source gates in
`docs/onnx_model_sources.md`.
I10. Add OWLv2, Grounding DINO, RTMDet OBB, and RapidOCR only from validated ONNX
artifacts; keep framework-native runtimes out of `anylearning.inference`.
I11. Verify custom exported models and packaged runtime behavior.
I12. Keep the shared external-data loader compatible with multi-gigabyte ONNX
bundles: exact SHA-256 coverage, bounded relative references, no links, zero-copy
read-only mappings, bundle-bound model revisions, and real-model validation on
Linux, Windows, and macOS. The loader and official YOLOX-S single-file/external
parity gate landed in PR #20; all three operating-system jobs and retained visual
reports passed.

### P0: Workflow correctness and performance

W1. Replace O(number-of-points) zoom mutation with a viewport transform while
preserving pointer-coordinate correctness.
W2. Add batch inference with progress, cancellation, bounded queues, and atomic
saves.
W3. Add resumable manifests, error reports, and stale-result rejection.
W4. Add embedding precomputation with exact image/model identity and size limits.
W5. Add YOLO-box-to-SAM refinement.
W6. Replace the growing training-log text field with appendable bounded storage
and paged/tailing reads.
W7. Complete import/export round trips for AnyLabeling, YOLO, COCO, and LabelMe.

### P1: Authenticated shared inference

S1. Add isolated server app factory and public threat assumptions.
S2. Add password hash CLI, token flow, redaction, and rate-limit tests.
S3. Add model discovery and deterministic synchronous prediction.
S4. Add bounded asynchronous queue, job state, cancellation, and timeouts.
S5. Add shared-session concurrency and cross-request isolation tests.
S6. Add Docker/reverse-proxy guidance and desktop-to-server end-to-end tests.
S7. Add AnyLabeling remote client and three-client contract compatibility tests.

### P1: Modular installation

P1. Classify every dependency by contracts, inference, training, server,
desktop, structured data, build, or development.
P2. Add clean virtual-environment install/import tests for proposed profiles.
P3. Introduce extras while preserving the existing complete install path.
P4. Make optional imports actionable rather than masking missing dependencies.
P5. Verify CLI help and base imports without heavyweight frameworks.
P6. Verify Nuitka builds and packaged self-tests after each dependency move.

### P2: Annotation breadth

A1. Add oriented-box persistence, rendering, editing, import/export, then model
decoding.
A2. Design mask-native storage, holes, disconnected regions, editing, and loss
reporting before Automask.
A3. Add classification tags and model results only after persistence/export
contracts exist.
A4. Consider OCR and video/tracking only after P0/P1 workflows meet stability
gates.

## Existing backlog incorporated

GitHub had no open issues at the 2026-08-30 audit. The following active items
come from `docs/TODO.md` and are promoted into this plan:

- Cross-platform packaged builds are not yet green on Windows and macOS.
- Canvas zoom mutates and replots every shape and will exceed a frame budget on
  larger annotations.
- First-party coverage is approximately 53%; dataset routing and auto-labeling
  have major gaps.
- Instance-segmentation end-to-end coverage does not yet exercise its real ONNX
  export path.
- ONNX export depends on the deprecated TorchScript exporter because packaged
  dynamo/onnxscript behavior is unresolved.
- Training logs use one growing database field and become superlinear.
- detectron2 CUDA extensions do not build against the current PyTorch version.
- UI sizing, semantic colors, and spacing need a consistent system.
- The default branch currently reports 60 open dependency alerts (including 23
  high severity, with duplicates between manifests and lockfiles). Website
  Next.js/PostCSS and frontend Effect alerts need immediate reachability review
  and supported upgrades; the PyTorch alert must be handled with the ML ABI and
  packaging matrix rather than an unverified isolated bump.

## Test and release gates

Every implementation PR must include deterministic written tests and the
relevant manual/visual checks. High-risk desktop, inference, training, server,
and packaging changes are checked on the physical Linux devbox, Apple Silicon
macOS machine, and Windows machine before release.

Every PR also states whether it changes a trust boundary. Security-sensitive
changes include abuse and failure tests, verify redaction, and demonstrate that
limits are enforced before expensive allocation or execution.

### Inference gate

- Contract serialization and compatibility fixtures pass.
- Preprocessing/postprocessing are deterministic.
- Results retain exact request/source/model identity.
- Load, failure, retry, cancellation, and unload pass in one process.
- CPU works; accelerated providers are verified or explicitly unsupported.
- Repeated inference does not grow memory or leak sessions/threads.

### Server gate

- Unauthenticated clients cannot enumerate models or operate jobs.
- Passwords and tokens do not appear in logs, errors, metrics, OpenAPI examples,
  command history generated by tests, or exported configuration.
- Token expiry/rotation, login rate limiting, TLS verification, size limits,
  cancellation, and graceful shutdown pass.
- Concurrent clients share sessions without cross-request leakage.
- The public server contains no desktop project/window routes.

### Desktop/package gate

- Written tests and pre-commit checks pass.
- Local source startup and packaged startup both pass.
- Label, save, reload, import, export, train, export-model, and inference smoke
  paths remain functional.
- Visual checks cover the affected workflow on Linux, macOS, and Windows.
- Install profiles are tested in clean environments with expected and forbidden
  imports recorded.

## Success measures

- Crash-free desktop and inference sessions.
- Zero stale/wrong-image result applications.
- Time from dataset open to first accepted prediction.
- p50/p95 navigation and canvas interaction latency.
- Model cold/warm latency and memory stability.
- Batch throughput, cancellation success, and failure recovery.
- Training completion through registered, loadable ONNX artifact.
- Server queue latency, bounded concurrency, and cross-client isolation.
- Clean install success by profile and packaged-build success by platform.
- Compatibility with AnyLabeling through the public protocol.
