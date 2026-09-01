import sys

# Answer --version before importing anything heavy. The imports below reach
# torch, which writes its own diagnostics to stdout, so `AnyLearning --version`
# used to answer with torch's "counting will not work for triton kernels" glued
# to the front of the version string -- ten seconds after being asked. It is
# the first thing support asks a user for, so it should be one clean line, and
# immediate.
#
# (The version itself is deliberately not written out here: a literal would be
# a second place to update, which tests/app/test_version_is_single_sourced.py
# forbids.)
if "--version" in sys.argv[1:]:
    from anylearning.app_info import __appname__, __version__

    print(f"{__appname__} v{__version__}")
    raise SystemExit(0)

import argparse
import base64
import logging
import multiprocessing
import os
import random
import secrets
import socket
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import requests
import uvicorn
import webview
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

# Ask torch to answer "is there a CUDA device?" through NVML rather than by
# initialising CUDA. Training uses a spawned child, but keeping the API process
# itself free of an unnecessary CUDA context avoids reserving GPU memory merely
# to report which hardware exists. Set before torch is imported anywhere; it is
# read at import time.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


from anylearning import frozen_compat, weights, window_chrome
from anylearning.app_info import __appname__, __description__, __version__
from anylearning.config import DATA_ROOT
from anylearning.migration_manager import MigrationManager
from anylearning.routers.dataset import router as dataset_router
from anylearning.routers.health import router as health_router
from anylearning.routers.labeling import router as labeling_router
from anylearning.routers.legal import router as legal_router
from anylearning.routers.model import router as model_router
from anylearning.routers.project import router as project_router
from anylearning.routers.settings import router as settings_router
from anylearning.routers.structured import router as structured_router
from anylearning.routers.training import router as training_router
from anylearning.routers.window import router as window_router
from anylearning.utils import extract_frontend_dist

app = FastAPI(
    title=__appname__,
    description=__description__,
)

security = HTTPBearer(auto_error=False)


def resolve_frontend_file(static_folder: str, path: str) -> Path | None:
    """Resolve an exported Next.js route without letting it escape the bundle.

    Next's static export writes ``/projects/dataset`` as
    ``projects/dataset.html``. Browser navigation uses the extensionless URL,
    so returning only ``index.html`` for that request silently booted the root
    page and dropped the selected project. Assets keep their literal paths,
    while extensionless routes also try the two layouts produced by static
    site generators.
    """
    root = Path(static_folder).resolve()
    requested = path.strip("/")
    literal = root / requested
    candidates = [literal]
    if not literal.suffix:
        candidates.extend((Path(f"{literal}.html"), literal / "index.html"))

    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


async def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if os.environ.get("ANYLEARNING_DEVELOPMENT"):
        return
    token = None
    if credentials:
        token = credentials.credentials
    else:
        token = request.query_params.get("token")
    if not token or not hasattr(webview, "token") or token != webview.token:
        logger.warning("Token verification failed")
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token


def create_app():
    frozen_compat.apply()
    logging.info("Extracting frontend distribution...")
    static_folder = os.path.abspath(os.path.join(DATA_ROOT, "frontend-dist"))
    extract_frontend_dist(static_folder)

    # Bundled auto-labelling models go into the data root the first time, so
    # the labelling screen finds them instead of downloading them.
    seeded = weights.seed_auto_labeling_models()
    if seeded:
        logging.info(f"Installed bundled auto-labelling models: {', '.join(seeded)}")
        # The manager read its configs when routers/labeling.py was imported,
        # which is before this runs, so what it is holding says these models
        # are not installed. Nothing else re-reads them.
        from anylearning.routers.labeling import model_manager

        model_manager.load_model_configs()

    logging.info("Running migrations...")
    migration_manager = MigrationManager()
    migration_manager.run_all_migrations()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def debug_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": str(type(exc).__name__),
                "file": exc.__traceback__.tb_frame.f_code.co_filename,
                "line": exc.__traceback__.tb_lineno,
            },
        )

    @app.middleware("http")
    async def catch_exceptions_middleware(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(e),
                    "trace": traceback.format_exc(),
                    "path": request.url.path,
                    "method": request.method,
                },
            )

    app.include_router(project_router, dependencies=[Depends(verify_token)])
    app.include_router(dataset_router, dependencies=[Depends(verify_token)])
    app.include_router(window_router, dependencies=[Depends(verify_token)])
    app.include_router(training_router, dependencies=[Depends(verify_token)])
    app.include_router(model_router, dependencies=[Depends(verify_token)])
    app.include_router(labeling_router, dependencies=[Depends(verify_token)])
    app.include_router(settings_router, dependencies=[Depends(verify_token)])
    app.include_router(structured_router, dependencies=[Depends(verify_token)])
    # No token: this is the endpoint you reach for when the app is not working,
    # and requiring the per-window token would make it useless from a script or
    # a packaging smoke test. It reports module availability, nothing private.
    app.include_router(health_router)
    # No token either: the licence notices are text we are obliged to publish,
    # and gating them behind a per-window token would mean the one document a
    # user is entitled to read is the one they cannot fetch from a browser.
    app.include_router(legal_router)

    @app.get("/api/is_anylearning")
    async def is_anylearning():
        return {"is_anylearning": True}

    @app.get("/{path:path}")
    async def serve_nextjs(path: str):
        file_path = resolve_frontend_file(static_folder, path)
        if file_path is not None:
            return FileResponse(file_path)
        index_path = os.path.join(static_folder, "index.html")
        if os.path.exists(index_path) and os.path.isfile(index_path):
            return FileResponse(index_path)
        return {"error": "File not found"}

    return app


def run_server(host, port, development=False, ssl_keyfile=None, ssl_certfile=None):
    if development:
        uvicorn.run(
            "anylearning.app:create_app",
            host=host,
            port=port,
            reload=development,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    else:
        uvicorn.run(
            create_app(),
            host=host,
            port=port,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )


def generate_self_signed_cert(keyfile, certfile):
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Generate certificate
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .sign(private_key, hashes.SHA256())
    )

    # Write private key
    with open(keyfile, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write certificate
    with open(certfile, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def run_desktop_app(host, port, development=False):
    print("Running desktop app...")
    # A compositor-owned frame is the only path that moves, resizes, snaps and
    # exposes the system menu consistently across both X11 and Wayland.
    native_frame = sys.platform.startswith("linux")
    ssl_keyfile = None
    ssl_certfile = None

    # Set environment variables to disable sandbox mode in development
    if development:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            "--no-sandbox --disable-web-security --disable-features=VizDisplayCompositor"
        )
        logger.info("Disabled webview sandbox mode for development")

    def start_server():
        run_server(
            host,
            port,
            development,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    url = f"http://{host.replace('0.0.0.0', '127.0.0.1').replace('localhost', '127.0.0.1')}:{port}"

    # Wait for server to start
    max_retries = 10
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
        except Exception:
            logger.info(
                f"Waiting for server to start (attempt {retry_count + 1}/{max_retries})..."
            )
            time.sleep(1)
            retry_count += 1
            if retry_count == max_retries:
                logger.error("Could not connect to server, restarting...")
                server_thread.join(timeout=1)
                return run_desktop_app(host, port, development)

    def download_file(url, suggested_name):
        try:
            result = window.create_file_dialog(
                webview.SAVE_DIALOG, save_filename=suggested_name
            )

            if result:
                if url.startswith("data:"):
                    # Handle base64 data URL
                    try:
                        # Extract the base64 data after the comma
                        base64_data = url.split(",")[1]

                        # Decode base64 and write to file
                        with open(result, "wb") as f:
                            f.write(base64.b64decode(base64_data))
                        return True
                    except Exception as e:
                        logger.error(f"Error decoding base64 data: {str(e)}")
                        return False
                else:
                    # Regular URL download
                    urllib.request.urlretrieve(url, result)
                    return True
            return False
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False

    def import_onnx_auto_labeling_model(options):
        """Install a model chosen through the native dialog without exposing paths."""
        try:
            result = window.create_file_dialog(
                webview.FileDialog.OPEN,
                allow_multiple=False,
                file_types=("ONNX model (*.onnx)",),
            )
            if not result:
                return {"ok": False, "cancelled": True}
            source = result if isinstance(result, str) else result[0]
            from anylearning.auto_labeling.custom_onnx import (
                install_custom_yolo_onnx,
            )
            from anylearning.routers.labeling import model_manager

            installed = install_custom_yolo_onnx(source, options)
            registered = model_manager.register_custom_model(installed)
            return {
                "ok": True,
                "model_name": registered["name"],
                "display_name": registered["display_name"],
            }
        except (OSError, ValueError) as error:
            logger.warning("Could not import ONNX auto-labeling model: {}", error)
            return {"ok": False, "error": str(error)}
        except Exception:
            logger.exception("Could not import ONNX auto-labeling model")
            return {
                "ok": False,
                "error": "The selected ONNX model could not be imported.",
            }

    # Update in place rather than replacing the dict. pywebview ships defaults
    # for every setting and reads them unconditionally -- replacing the whole
    # dict dropped the other keys, and pywebview 6 then died creating the window
    # with KeyError: 'IGNORE_SSL_ERRORS'. Assigning only what we override also
    # means new settings in future pywebview releases keep their defaults.
    webview.settings.update(
        {
            "ALLOW_DOWNLOADS": True,
            "ALLOW_FILE_URLS": True,
            "OPEN_EXTERNAL_LINKS_IN_BROWSER": True,
            # The app draws its own title bar, so a press drags the window only
            # when it lands on a drag surface itself. Left off, pywebview walks
            # up from whatever was clicked, and every control *inside* the bar
            # drags the window too.
            "DRAG_REGION_DIRECT_TARGET_ONLY": True,
            "DRAG_REGION_SELECTOR": (
                ".pywebview-native-titlebar"
                if native_frame
                else ".pywebview-drag-region"
            ),
        }
    )
    window = webview.create_window(
        __appname__,
        url,
        width=1200,
        height=800,
        resizable=True,
        # Windows and macOS integrate the workbench bar with a custom frame;
        # Linux keeps the compositor-owned frame for reliable move, resize,
        # snap and system-menu behavior on both X11 and Wayland.
        frameless=not native_frame,
        # Drag surfaces are declared in the DOM; easy_drag would make the whole
        # window one, canvas included.
        easy_drag=False,
        # An alpha channel, only where the corners have to be rounded by the
        # page rather than by the system.
        transparent=window_chrome.needs_transparency() and not native_frame,
    )

    # Native, user-mediated file operations. The browser/server build never
    # receives these methods or a local path.
    window.expose(download_file, import_onnx_auto_labeling_model)
    window_chrome.attach(window, native_frame=native_frame)

    # Start the webview event loop
    webview.start(debug=development, gui="qt" if native_frame else None)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_random_port():
    while True:
        port = random.randint(1024, 65535)
        if not is_port_in_use(port):
            return port


def is_anylearning_running(host, port):
    try:
        response = requests.get(f"http://{host}:{port}/api/is_anylearning")
        return response.status_code == 200 and response.json().get(
            "is_anylearning", False
        )
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__description__,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the server on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5678,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as a server without the desktop application",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Run server in development mode with auto-reload",
    )
    parser.add_argument(
        "--self-test",
        dest="self_test",
        action="store_true",
        help="Train every project type on generated data and report; see anylearning/selftest",
    )
    args, extra = parser.parse_known_args()

    if args.version:
        print(f"{__appname__} v{__version__}")
        return

    if args.self_test:
        # Imported here: it launches this same binary as a server, and nothing
        # about a normal start should pay for loading it.
        from anylearning import selftest

        raise SystemExit(selftest.main(extra))

    logging.info(f"Starting {__appname__}...")
    logging.info(f"Version: {__version__}")

    if args.development:
        logging.info("Running in development mode")
        args.server = True
        os.environ["ANYLEARNING_DEVELOPMENT"] = "TRUE"

    if is_port_in_use(args.port):
        if is_anylearning_running(args.host, args.port):
            logger.error(
                f"AnyLearning is already running on port {args.port}. Please close it first."
            )
            error_html = f"""
                <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                           padding: 2.5rem;
                           text-align: center;
                           background: #ffffff;">
                    <div style="color: #1f2937;
                              margin-bottom: 1.5rem;
                              font-size: 1.5rem;
                              font-weight: 600;">
                        ⚠️ Already Running
                    </div>
                    <p style="color: #4b5563;
                             font-size: 1.1rem;
                             line-height: 1.6;
                             margin: 0;">
                        AnyLearning is currently active on port {args.port}.<br>
                        Please close the existing window before launching a new instance.
                    </p>
                </div>
            """
            webview.create_window(
                "AnyLearning",
                html=error_html,
                width=520,
                height=280,
                background_color="#f3f4f6",
            )
            webview.start()
            return
        new_port = get_random_port()
        logger.info(f"Port {args.port} is in use, using random port {new_port} instead")
        args.port = new_port

    if args.server:
        # verify_token compares against webview.token, and nothing printed it,
        # so every request from a script was rejected and the mode was unusable.
        #
        # It is not minted here. pywebview assigns `token` at *import* time, so
        # the previous `if not hasattr(webview, "token")` was never true and the
        # value below never reached anyone -- the mode stayed as broken as
        # before, quietly. Take the token that verify_token will actually check.
        if not getattr(webview, "token", None):
            webview.token = secrets.token_urlsafe(32)
        logger.info(f"API token for this server: {webview.token}")
        logger.info("Pass it as 'Authorization: Bearer <token>' or '?token=<token>'.")
        run_server(args.host, args.port, args.development)
    else:
        logger.info("Running desktop app")
        run_desktop_app(args.host, args.port, args.development)


if __name__ == "__main__":
    # Initialize multiprocessing support for frozen executables
    if hasattr(multiprocessing, "freeze_support"):
        multiprocessing.freeze_support()

    main()
