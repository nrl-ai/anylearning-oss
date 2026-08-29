"""Window controls over HTTP.

The title bar itself does not come through here -- it calls
`window.pywebview.api.window_*` directly, which is one hop instead of three and
needs no token. These routes are the same actions for anything that only has
the HTTP API, and they go through `anylearning.window_chrome` so the two paths
cannot drift into disagreeing about what "maximise" means.
"""

from fastapi import APIRouter
from loguru import logger

from anylearning import window_chrome

router = APIRouter(prefix="/window", tags=["Window Control"])


@router.post("/close")
def close():
    if window_chrome.close_window():
        return {"message": "Window closed"}
    logger.error("No window to close.")
    return {"message": "No window to close."}


@router.post("/maximize")
def maximize():
    if window_chrome.maximize_window():
        logger.info("Window maximized")
        return {"message": "Window maximized"}
    logger.error("No window to maximize")
    return {"message": "No window to maximize"}


@router.post("/restore")
def restore():
    if window_chrome.restore_window():
        logger.info("Window restored")
        return {"message": "Window restored"}
    logger.error("No window to restore")
    return {"message": "No window to restore"}


@router.post("/minimize")
def minimize():
    if window_chrome.minimize_window():
        logger.info("Window minimized")
        return {"message": "Window minimized"}
    logger.error("No window to minimize")
    return {"message": "No window to minimize"}
