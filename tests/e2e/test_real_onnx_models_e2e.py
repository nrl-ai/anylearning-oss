"""Opt-in end-to-end validation against real, locally supplied ONNX models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from anylearning.inference.validation import run_real_model_validation

_MANIFEST = os.environ.get("ANYLEARNING_REAL_MODEL_MANIFEST")


@pytest.mark.skipif(
    not _MANIFEST,
    reason="set ANYLEARNING_REAL_MODEL_MANIFEST to a real-model JSON manifest",
)
def test_real_onnx_model_is_accurate_repeatable_and_visually_logged():
    output_root = Path(
        os.environ.get("ANYLEARNING_REAL_MODEL_OUTPUT", "validation-results")
    )
    output_dir = run_real_model_validation(Path(_MANIFEST), output_root=output_root)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["passed"], summary["images"]
    assert summary["model"]["sha256"]
    assert summary["provenance"]["artifact_license"]
    assert (output_dir / "results.json").is_file()
    assert (output_dir / "index.html").is_file()
    for image in summary["images"]:
        assert image["consistent_runs"]
        assert (output_dir / image["annotated_image"]).is_file()
