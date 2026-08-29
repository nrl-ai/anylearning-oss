# Release testing

What to run before publishing an installer, on **every** platform you ship.

The point of this document is that **the test suite passing tells you nothing
about the packaged app.** Nuitka compiles the dependency graph to C; the things
that break are imports it could not follow statically, and they break only in
the frozen binary. In one afternoon of building locally, all of the following
were green in `pytest` and broken in the build:

| Symptom                           | Cause                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------ |
| Binary died on startup            | `torchvision.ops` imports `torch._dynamo`, which was excluded                  |
| Binary segfaulted, no message     | `array_api_compat.numpy.fft` built by `clone_module()`, dropped                |
| Every GPU run failed              | Nuitka's torch config ignores `triton`; its guard escalated torch's probe      |
| Training finished, model vanished | `onnxscript._framework_apis` missing, so ONNX export failed                    |
| App shipped yesterday's UI        | frontend export only rebuilt when absent                                       |
| Migrations never ran, silently    | `--include-data-dir` skips `.py`, so alembic's revisions were not in the build |

Three of those produce a _successful build_. Two produce a binary that passes
`--version`. Only running the real flows finds them.

---

## 0. Before you start

|          |                                                                                             |
| -------- | ------------------------------------------------------------------------------------------- |
| Weights  | `python fetch_weights.py` — **before** the build, see below                                 |
| Frontend | `bash build_frontend.sh` — also before the build                                            |
| Notices  | `python generate_licenses.py` — in _this_ environment, after any dependency change          |
| Build    | `bash build_app.sh` on the target OS (~1 h cold)                                            |
| Data     | a local copy of [anylearning-data](https://huggingface.co/datasets/nrl-ai/anylearning-data) |
| GPU box  | at least one NVIDIA machine, for the CUDA rows                                              |

The third fails in no way at all, which is why it is on the list: `LICENSES.md`
is generated from the installed distributions, and a dependency added since it
was last generated simply has no notice in the shipped file. Every permissive
licence we redistribute under asks for its notice to travel with the binary, so
this is a legal step rather than a tidiness one. Run it in the environment the
release is built from — a developer box with extra packages generates a file
that names things the installer does not contain.

The first two are easy to forget on a machine that has not built before, and
they fail in opposite ways. Without `weights/`, Nuitka refuses to start at all:

    FATAL: Error, malformed '--include-data-dir' value, must specify existing
    source data directory, not 'weights'

Without a fresh `build_frontend.sh`, it builds happily and ships whatever
`anylearning/frontend-dist` was left holding — which is untracked, so on a
fresh clone that is nothing, and on an old one it is the last release. The
export bakes the version in, so the giveaway is a sidebar reading a version you
are not building.

Test on a machine that has **never** had the development environment on it
where you can. A dev box has torch, CUDA libraries and a populated
`~/anylearning-data` on it, any of which can hide a packaging bug — the app
finds on your machine what it would not find on a user's.

---

## 1. Automated (run first — cheap, and catches most of it)

```shell
bash smoke_test_build.sh <binary>        # starts, serves, imports
python feature_test.py <binary>          # everything except training, over HTTP
<binary> --self-test                     # trains every type on data it draws itself
<binary> --self-test --architectures rfdetr,rfdetr-seg   # the second detector and
                                         # the second instance segmenter, which
                                         # the line above never touches
python smoke_test_training.py <binary>   # every type again, on your real projects
```

`<binary>` is `app.dist/app.bin` (Linux), `AnyLearning.app/Contents/MacOS/app`
(macOS), `AnyLearning.App\AnyLearning.exe` (Windows).

All four must exit 0. If the first fails, stop — there is no point testing a
binary that cannot start.

`feature_test.py` is the one that covers the product rather than the plumbing:
projects of every type, labels, uploads, import and export in four formats,
annotations, class balance, `copy_subset`, project archives, settings, the
legal documents, the licence endpoints, auto-labelling on the bundled weights,
and one training run end to end through inference and ONNX download. It starts
the binary on a temporary data root of its own and asks the running application
questions over HTTP — it imports nothing from the package, because everything
that has gone wrong in a release went wrong in packaging.

It found, in one afternoon: bundled auto-labelling models that could not load
at all, a project import that died on a null size, and both directions of the
training liveness check being wrong. Run it before spending an hour on the
manual flows.

`--only` and `--skip` take check names, so a fix can be re-verified in seconds.

The middle one is the reason a machine no longer needs a development
environment before it can be tested: `--self-test` generates its own labelled
project of each type into a temporary data root, trains it, and asserts a model
appears — which only happens after ONNX export succeeds. Nothing needs to be
installed and no real project is touched.

It cannot generate handpose data — that trainer reads hand landmarks with
mediapipe, and no drawn shape produces those — which is why
`smoke_test_training.py` still runs afterwards against real projects. Between
them every type is covered.

**Handpose is expected to be unavailable on macOS.** mediapipe aborts there, so
`GET /api/settings/capabilities` answers `{"handpose": false}` and the
project-creation form disables the type. On macOS, check that; on Windows and
Linux, check the opposite. An upload to a handpose project on a machine where
the model cannot run must _fail_ with that explanation — not complete over an
empty dataset, which is what it used to do.

Read the environment block it prints before reading anything else. It quickly
reveals cross-platform problems such as a torch build with no CUDA on a machine
with a GPU, or an Apple GPU the trainers will never select.

`GET /api/health/imports` gives the per-module picture when something is wrong;
it imports torch, `torchvision.ops`, cv2, onnxruntime, Lightning and all five
trainers inside the running process and reports each one.

---

## 2. Manual flows

Run every row on every OS. "Same data" means the same project on each platform,
so results are comparable.

### 2.1 Install and first run

- [ ] Installer completes without a security prompt you cannot dismiss
      (macOS: Gatekeeper; Windows: SmartScreen). Record what the user sees.
- [ ] App launches to a window, not a blank frame.
- [ ] Window chrome: title bar draggable, traffic lights / caption buttons
      present and not overlapped by page content. See
      `smoke_test_window_chrome.py`.
- [ ] Quit and relaunch. No "already running" error, no orphaned process
      holding the port. (The app falls back to a random port when 5678 is busy,
      which hides this — check the process list.)
- [ ] `~/anylearning-data/` is created on first run.

### 2.2 Projects and data

- [ ] Create one project of **each** type: Object Detection, Image
      Classification, Image Segmentation, Instance Segmentation, Handpose
      Classification, Keypoint Detection.
- [ ] Import a `.zip` for each; the item count matches the archive.
- [ ] Labels are auto-created from folder names when that box is ticked.
- [ ] Thumbnails render in the dataset grid.
- [ ] Delete a project; its folder under `~/anylearning-data/projects/` goes too.

### 2.4 Labelling

The canvas is the most platform-sensitive screen: it is SVG, pointer events and
a native scroll container.

- [ ] Image fills the pane, correct aspect ratio, no clipping.
- [ ] Draw a box / polygon; it lands where the cursor was.
- [ ] In a Keypoint Detection project, place landmarks, give two subjects
      different instance numbers, mark one point occluded, save and reopen; the
      instance and visibility labels still match.
- [ ] **Release the mouse outside the canvas**, then draw again — the second
      gesture must work. (This regressed once; the builder kept half-finished
      state and silently rejected everything after.)
- [ ] Wheel zooms; middle-drag and Ctrl+drag pan; a zoomed-in image can be
      panned to all four edges.
- [ ] Drag a vertex, move a shape, rotate, delete.
- [ ] Existing annotations load in the right place at the right scale.
- [ ] Auto-save reports "Annotation saved"; reopen the image and the shapes are
      still there.
- [ ] Open an image with many shapes (the EM-particle set has one with 65
      polygons / 12k points) and confirm zoom stays responsive.
- [ ] Auto-labelling (SAM) produces a shape, if the models are present.

### 2.5 Training — **once on GPU, once on CPU**

Force CPU with `CUDA_VISIBLE_DEVICES=` (empty) so both paths are covered; a GPU
box otherwise never exercises the CPU path a user without a GPU will hit.

For each project type:

- [ ] Training starts and the log streams into the UI.
- [ ] Metrics move in the right direction over a few epochs.
- [ ] **A model appears in the Models page when it finishes.** This is the real
      assertion: the model is registered only after ONNX export succeeds, so a
      finished run with no model means export failed.
- [ ] Terminating a run mid-training marks it terminated and frees the process.
- [ ] Kill the app mid-training, reopen: the run is reconciled to `error`, not
      left showing "training" for ever.

### 2.6 Inference

- [ ] "Try" a trained model on an image from the test split.
- [ ] The prediction is drawn/reported and is not obviously wrong.
- [ ] Works for a model trained on **this** machine and for one trained on
      **another** (copy the project folder across). Checkpoints store the whole
      `nn.Module`, so a GPU-trained model opened on a CPU-only machine is a real
      failure mode — see `device_utils.py`.

### 2.7 Export

- [ ] Export the **model**: the ONNX file is produced and is non-empty.
- [ ] The exported ONNX loads in onnxruntime and produces output.
- [ ] Export the **dataset** in each offered format; the archive downloads and
      contains labels, not just images. (YOLO once silently dropped polygons.)
- [ ] Export a project and re-import it into a clean install.

### 2.8 Performance modes

- [ ] Settings → Performance shows the machine's core count and the worker
      counts it resolves to.
- [ ] Training under **Power saving** visibly uses fewer cores than **Maximum**
      (watch a process monitor), and still finishes.
- [ ] The mode survives a restart.

### 2.9 Settings and the rest

- [ ] Theme toggle: light and dark, and no flash of the wrong theme on load.
- [ ] Preferences (page sizes) persist across restart.
- [ ] Update check does not offer a _downgrade_.
- [ ] Resize the window narrow and wide; nothing is unreachable.

---

## 3. Building the installers

`build_app.sh` produces something that runs; it does not produce something you
can put on the website. That is a separate step per platform, and it is the
step where a release stops being a local build.

### Windows — Inno Setup

```shell
bash build_app.sh                                     # makes AnyLearning.App\
"$LOCALAPPDATA/Programs/Inno Setup 6/ISCC.exe" AnyLearning-Windows-Setup.iss
```

`build_app.sh` writes `installer_version.iss` from `anylearning/app_info.py`,
which both `.iss` scripts `#include` — so the installer version follows the app
version and is never typed twice. Run the build first or ISCC stops on a
missing include.

Two scripts, and they are nearly the same file: `AnyLearning-GPU-Windows-Setup.iss`
differs only in the output name, `DiskSpanning`, and adding the install
directory to the system `PATH`.

**From 0.26.0 use the spanning one.** Inno refuses to build a single Setup.exe
larger than about 4.2 GB, and with the weights bundled the tree is 7.5 GB:

    Disk spanning must be enabled to create an installation larger than
    4200000000 bytes in size for a single Setup.exe

`DiskSpanning=yes` produces `...-Setup-<version>.exe` plus `-1.bin`, `-2.bin`
slices. Zip the whole set together and publish that -- **every slice has to be
in the zip**, because an installer missing one fails at install time rather
than at download time. It is what 24 shipped too:
`AnyLearning-GPU-Windows-Setup-0.24.13.zip`.

**Budget 20 minutes and 3.5 GB.** Measured on 0.26.0: 18 minutes of LZMA solid
compression for a 3,502,690,369-byte setup executable. The CUDA runtime is
nearly all of it — cuDNN, cuBLAS, cuFFT and cuSOLVER are several hundred MB
each and barely compress. Budget the disk too: the compile needs room for the
whole tree _and_ the output at once.

3.5 GB is a lot to ask someone to download, and 26 made it larger rather than
smaller: the pretrained weights now ship with the application, which is what
makes training work on a machine that has never been online. That is 1.6 GB of
`.pth`, `.onnx` and `.pkl` on disk, and they are already compressed, so the
installer gets almost nothing off them.

The two figures to quote on the website:

|                   | 24     | 26               |
| ----------------- | ------ | ---------------- |
| Windows installer | 3.5 GB | 3.5 GB + weights |
| macOS image       | 561 MB | 1.7 GB           |

The macOS one is the honest measure of the application without a CUDA runtime,
and a fair estimate of what a CPU-only Windows installer would be — that is
what the second `.iss` is for, and it is worth shipping if the download size
starts costing sales.

### macOS — disk image

```shell
bash build_app.sh        # makes AnyLearning.app
bash make_dmg.sh         # makes AnyLearning-macOS-<arch>-<version>.dmg
```

Compressed (UDZO), the bundle becomes the download: 1.6 GB became 561 MB in 24,
and in 26 a 2.9 GB bundle becomes 1.7 GB — the bundled weights are already
compressed, so almost none of that gigabyte comes off. The image is the
familiar drag-to-Applications layout.

Check the weights survived, before anything else: `find AnyLearning.app -path
'*/anylearning/weights/*' -type f | wc -l` must match `find weights -type f |
wc -l`. Nuitka silently shipped 23 of 41 files once — it dropped the Hugging
Face cache, which stores each file under `blobs/` and links to it from
`snapshots/` — and reported "Included 41 data files" while doing it. There are
no symlinks in `weights/` any more (`fetch_weights.py` flattens them) but count
them anyway: a build that quietly loses a file is exactly what this document
exists for.

**Do not publish this unsigned.** An unsigned, un-notarised bundle is refused by
Gatekeeper on every Mac except the one that built it, and the message the user
gets is _"AnyLearning is damaged and can't be opened"_ — which reads as a
corrupt download, not as a policy decision, so the user retries it and files a
bug. Shipping needs, in order:

1. A Developer ID Application certificate in the build machine's keychain.
2. `codesign --deep --force --options runtime --timestamp -s "Developer ID
Application: ..." AnyLearning.app` — the hardened runtime (`--options
runtime`) is required for notarisation.
3. `xcrun notarytool submit --wait` on the `.dmg`, then `xcrun stapler staple`
   so it validates without a network round trip.

Two things about this build in particular. Signing must come _after_
`patch_detectron2_macos.py` has repointed detectron2's load commands:
`install_name_tool` invalidates any signature on the dylibs it edits. And the
`.app` is around 1.6 GB of mostly Python and shared libraries, so `--deep`
signing is slow — minutes, not seconds. Neither is a reason to skip it.

Until a certificate exists, the honest instruction for users is
`xattr -dr com.apple.quarantine /Applications/AnyLearning.app`, which is in
`README.md`. Treat that as a stopgap: asking people to strip quarantine
attributes teaches them to do it for software that is not yours.

### Both

- Installers carry `app_icon.ico` / `app_icon.png` from the repository root.
  They are baked in at build time, so changing the icon means rebuilding, not
  just re-running the installer step.
- Nothing here is signed on Windows either. SmartScreen will warn on an
  unsigned setup until the download has enough reputation; an EV certificate
  skips that wait.

---

## 4. Platform-specific traps

**macOS**

- detectron2's `_C` links torch through `@rpath`; Nuitka resolves it relative to
  the owning package and fails unless the load commands are repointed. Editing
  them invalidates the ad-hoc signature, so it must be re-signed.
- `.icns` conversion needs `imageio` installed at build time.
- Gatekeeper quarantine on a downloaded build — see `README.md`.

**Windows**

- Built with MSVC, so ccache does not apply; Nuitka's own cache does.
- The binary is renamed to `AnyLearning.exe` and the tree to `AnyLearning.App`.
- Console is disabled in the build, so a startup crash is _silent_. Run the
  smoke test from a terminal to see anything at all.

**Linux**

- pywebview needs a GTK/WebKit backend; a missing one is a blank window, not an
  error.
- Nuitka needs `patchelf`.

**Data files that are Python**

`--include-data-dir` silently skips `.py` files -- it assumes anything
importable is being compiled instead. Alembic reads `env.py` and its revisions
from disk by path, so that assumption is wrong for them, and a build shipped a
migrations folder containing only a README. Nothing failed visibly:
`MigrationManager` logs its own errors, so every database went unstamped.

Use `--include-raw-dir` (or an explicit `*.py` pattern), which is what Nuitka's
author recommends in [issue 3270](https://github.com/Nuitka/Nuitka/issues/3270).
Embedding such files so they are not on disk at all needs Nuitka commercial;
that only matters if you want to _protect_ them, which we do not.

Shipping the file is not the same question as importing it at runtime. Both
were checked in a throwaway build before spending an hour on a real one: the
frozen binary found the revision, ran it, created the table and wrote the
stamp. `smoke_test_build.sh` now reads that stamp out of a database the build
creates, so this cannot regress quietly.

---

## 5. Sign-off

Record the result per platform. "Not tested" is a legitimate entry; "assumed to
work because another platform did" is not.

| Flow                | Linux | macOS | Windows |
| ------------------- | ----- | ----- | ------- |
| Automated smoke     |       |       |         |
| Install / first run |       |       |         |
| Licence             |       |       |         |
| Import data         |       |       |         |
| Labelling           |       |       |         |
| Train (GPU)         |       |       |         |
| Train (CPU)         |       |       |         |
| Inference           |       |       |         |
| Export model        |       |       |         |
| Export dataset      |       |       |         |

Note the build each row was run against — a rebuild between rows invalidates
the ones before it.
