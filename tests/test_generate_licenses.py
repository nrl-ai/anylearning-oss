from __future__ import annotations

import pytest

from generate_licenses import extract_vendored_licences


def test_extract_vendored_licences_ignores_previous_generated_headers() -> None:
    notice = """# Third-party licences

## Vendored components

```
# Third-party licences

## Summary

## Vendored components

These are copied into the application.

```
NanoDet - Apache License 2.0
nanodet terms

DeepLabV3+ - MIT License
deeplab terms
```
```

## Full licence texts
"""

    assert extract_vendored_licences(notice) == (
        "NanoDet - Apache License 2.0\n"
        "nanodet terms\n\n"
        "DeepLabV3+ - MIT License\n"
        "deeplab terms"
    )


def test_extract_vendored_licences_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="vendored licence payload"):
        extract_vendored_licences("# Third-party licences\n")
