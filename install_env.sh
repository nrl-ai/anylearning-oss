#!/bin/bash

set -e

# Install system dependencies for pywebview on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing system dependencies for Linux..."

    # Check if running on Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        # Install GTK development libraries
        sudo apt-get install -y \
            python3-gi \
            python3-gi-cairo \
            gir1.2-gtk-3.0 \
            libcairo2-dev \
            pkg-config \
            python3-dev \
            libglib2.0-dev

        # PyGObject >= 3.52 builds against girepository-2.0, which ships in
        # libgirepository-2.0-dev (glib >= 2.80, i.e. Ubuntu 24.10 and newer).
        # libgirepository1.0-dev only provides girepository-1.0, so on a current
        # release the PyGObject wheel build fails with
        #   ERROR: Dependency 'girepository-2.0' is required but not found.
        # Prefer the 2.0 headers and fall back to 1.0 on older distributions.
        if apt-cache show libgirepository-2.0-dev &> /dev/null; then
            sudo apt-get install -y libgirepository-2.0-dev
        else
            sudo apt-get install -y libgirepository1.0-dev
        fi

        # Try to install WebKit2GTK (try different versions)
        if apt-cache show gir1.2-webkit2-4.1 &> /dev/null; then
            sudo apt-get install -y gir1.2-webkit2-4.1
        elif apt-cache show gir1.2-webkit2-4.0 &> /dev/null; then
            sudo apt-get install -y gir1.2-webkit2-4.0
        else
            echo "Warning: WebKit2GTK not found, trying alternative packages..."
            sudo apt-get install -y libwebkit2gtk-4.0-dev || sudo apt-get install -y libwebkit2gtk-4.1-dev || true
        fi

        # Alternative: Install QT dependencies (uncomment if you prefer QT over GTK)
        # sudo apt-get install -y python3-pyqt5 python3-pyqt5.qtwebengine

        # Install other required system packages
        sudo apt install -y patchelf libpango1.0-dev libgif-dev

    # Check if running on CentOS/RHEL/Fedora
    elif command -v yum &> /dev/null || command -v dnf &> /dev/null; then
        if command -v dnf &> /dev/null; then
            PKG_MANAGER="dnf"
        else
            PKG_MANAGER="yum"
        fi

        sudo $PKG_MANAGER install -y \
            python3-gobject \
            python3-gobject-devel \
            gtk3-devel \
            webkit2gtk3-devel \
            cairo-gobject-devel \
            pkg-config \
            python3-devel \
            glib2-devel
    fi
fi

# Install the dependencies
pip install -e .
pip install -r requirements.txt

# NanoDet
cd anylearning/training/models/nanodet && pip install -r requirements.txt && pip install -e . && cd -

# HandPose
cd anylearning/training/models/handpose && pip install -r requirements.txt && pip install -e . && cd -

# Semantic Segmentation
cd anylearning/training/models/semantic_segmentation && pip install -r requirements.txt && cd -

# Instance Segmentation (Detectron2)
#
# Do NOT reinstall torch/torchvision here. This step used to run an unpinned
# `pip install torch torchvision`, which silently upgraded past the torch pin
# that setup.py had just installed -- so the environment you ended up with
# depended on the day you ran this script. setup.py owns that pin and
# `pip install -e .` above has already applied it.
#
# Note also that a torch bump requires rebuilding detectron2, and that crossing a
# CUDA major version needs a *fresh* environment rather than an in-place upgrade
# (see docs/dependency_upgrade.md).
#
# detectron2 has no PyPI release, so it comes from git. Pin the commit: tracking
# the branch head means two developers running this script a week apart get
# different builds.
# --no-build-isolation is required: detectron2's setup.py imports torch to decide
# which extensions to compile, but PEP 517 builds run in an isolated environment
# where torch is absent, so the build fails with "No module named 'torch'".
# Disabling isolation lets it see the torch installed above.
DETECTRON2_COMMIT=b4a4a3bd136852dae5fb1de37978dee412653e31
# On macOS the extension has to be linked with room to grow in its Mach-O
# header. Packaging rewrites every dependency path -- @rpath/libc10.dylib
# becomes @executable_path/torch/lib/libc10.dylib, seventeen characters longer
# -- and a default link leaves no slack for that, so the build dies after the
# whole two-hour C phase with:
#
#   install_name_tool: changing install names or rpaths can't be redone ...
#   because larger updated load commands do not fit (the program must be
#   relinked, and you may need to use -headerpad_max_install_names)
#
# Measured: without this flag the rewrite does not fit even on a pristine
# wheel, so it is the link that has to change, not the patching.
if [[ "$OSTYPE" == "darwin"* ]]; then
    export LDFLAGS="${LDFLAGS:-} -Wl,-headerpad_max_install_names"
fi
python -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_COMMIT}"

# Setup pre-commit hooks
pre-commit install
