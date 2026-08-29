# GPU Support

CUDA works on **Linux and Windows** with NVIDIA GPUs — the trainers pick the
device from `torch.cuda.is_available()`, so a CUDA-capable environment needs no
configuration. (Verified on Linux: NanoDet, DeepLabv3 and Mask R-CNN all train
on an RTX 3080.) Only the *packaged Windows installer* is a distinct,
manually-produced GPU artefact, which is what the rest of this file is about.

CI deliberately builds CPU-only — it has no GPU to exercise, and the CUDA
wheels add ~3 GB of download per job.

## Install PyTorch with CUDA Support

**On Windows this step is required, not optional.** PyPI's Windows wheels are
CPU-only, so `pip install -e .` there gives you `torch 2.11.0+cpu` and
`torch.cuda.is_available()` is False on a machine with a perfectly good GPU —
measured on an RTX 2070. Only Linux gets CUDA from the default index, where this
step is needed just to pin a *different* CUDA version.

Keep the torch and torchvision versions in step with `setup.py` — they are
exact-pinned there for a reason (see `docs/dependency_upgrade.md`).

```bash
# Uninstall existing torch and torchvision
pip uninstall -y torch torchvision

# Install PyTorch with CUDA Support (versions must match setup.py)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

> **Rebuild detectron2 afterwards.** It compiles against the installed torch, so
> changing torch leaves it linked against the old one:
>
> ```bash
> pip install --no-build-isolation --force-reinstall --no-deps \
>     "git+https://github.com/facebookresearch/detectron2.git@b4a4a3bd136852dae5fb1de37978dee412653e31"
> ```

> **Changing CUDA major version? Use a fresh environment.** The CUDA 12 and CUDA
> 13 wheels ship under *different* package names (`nvidia-*-cu12` vs
> `nvidia-*-cu13`), so an in-place upgrade leaves the old set installed and
> orphaned. The stale copies shadow NVRTC and detectron2 fails at runtime with
> `nvrtc: error: failed to open libnvrtc-builtins.so.13.0`, even though that file
> is present. Uninstalling the `-cu12` packages by hand makes it worse — it also
> removes `libcudnn.so.9` and torch stops importing.

## Build the App

```bash
./build_app.sh
```

## Package the App for Windows

Package the app for Windows using Inno Setup with setting file `AnyLearning-GPU-Windows-Setup.iss`.
