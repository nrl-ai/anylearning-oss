#!/bin/bash

set -e

# The previous --ignore flags pointed at anylearning/training/models/detectron2,
# a vendored copy that no longer exists (detectron2 is now installed from git by
# install_env.sh). pytest silently accepts --ignore paths that are absent, so the
# flags had quietly become no-ops.

coverage run -m pytest tests/
coverage report -m
