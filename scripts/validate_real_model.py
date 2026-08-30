#!/usr/bin/env python3
"""Run a manifest-defined real ONNX model validation and print its report path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the checkout importable when invoked as `python scripts/...` without an
# editable install. Installed entry points do not need this path adjustment.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anylearning.inference.validation import run_real_model_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("validation-results"))
    arguments = parser.parse_args()
    output = run_real_model_validation(
        arguments.manifest,
        output_root=arguments.output_root,
    )
    print(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
