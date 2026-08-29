"""Serving the third-party licence notices to the UI.

Shipping the notice file is only half of the obligation -- the other half is
that someone can actually read it, which is what this is for. It reads the file
on request rather than at import: it is a few tens of kilobytes that most
sessions never open.
"""

from fastapi import APIRouter, HTTPException

from anylearning import legal

router = APIRouter(prefix="/api/legal", tags=["Legal"])


@router.get("/notices")
async def get_notices():
    text = legal.read_notices()
    if text is None:
        # A build that did not ship the notices is a packaging defect, and
        # saying so is more useful than an empty panel that looks deliberate.
        raise HTTPException(
            status_code=404,
            detail="This build did not ship LICENSES.md.",
        )
    # Structured, not raw markdown: the UI renders a list of components, and
    # markdown syntax shown literally in a <pre> reads as a rendering bug.
    return {"components": legal.parse_notices(text)}


@router.get("/license")
async def get_license():
    """Serve the project's Apache-2.0 license as plain text."""
    text = legal.read_license()
    if text is None:
        raise HTTPException(
            status_code=404,
            detail="This build did not ship LICENSE.",
        )
    return {"text": text}


@router.get("/model-policy")
async def get_model_policy():
    """Which model weights may ship, and under what licences.

    It must be readable by application users, not only repository visitors.
    """
    text = legal.read_model_policy()
    if text is None:
        raise HTTPException(
            status_code=404,
            detail="This build did not ship the model licence policy.",
        )
    return {"text": text}
