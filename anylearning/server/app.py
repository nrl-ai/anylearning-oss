"""A narrow FastAPI application containing no desktop or project routes."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import BoundedSemaphore
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from anylearning.inference import CURRENT_PROTOCOL_VERSION, ModelCapabilities

from .auth import InvalidTokenError, PasswordAuthenticator, TokenClaims, TokenSigner
from .config import ServerSettings
from .limits import RequestBodyLimitMiddleware
from .rate_limit import LoginRateLimiter

logger = logging.getLogger(__name__)

_MAX_MODELS = 256
_PASSWORD_EXECUTOR_PREFIX = "anylearning-password"


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HealthResponse(_ResponseModel):
    status: str = "ok"
    protocol_version: str = CURRENT_PROTOCOL_VERSION


class LoginRequest(_ResponseModel):
    password: str = Field(min_length=1, max_length=1_024)


class TokenResponse(_ResponseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ModelListResponse(_ResponseModel):
    models: tuple[ModelCapabilities, ...]


def create_server_app(
    settings: ServerSettings,
    *,
    models: Iterable[ModelCapabilities] = (),
) -> FastAPI:
    """Build the authenticated public app without importing desktop routes."""
    catalog = _model_catalog(models)
    authenticator = PasswordAuthenticator(settings.password_hash)
    signer = TokenSigner(
        settings.token_secret,
        ttl_seconds=settings.token_ttl_seconds,
    )
    limiter = LoginRateLimiter(
        attempts_per_client=settings.login_attempts_per_client,
        global_attempts=settings.global_login_attempts,
        window_seconds=settings.login_window_seconds,
    )
    password_slots = BoundedSemaphore(settings.max_concurrent_password_checks)
    password_executor = ThreadPoolExecutor(
        max_workers=settings.max_concurrent_password_checks,
        thread_name_prefix=_PASSWORD_EXECUTOR_PREFIX,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        password_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="AnyLearning Inference Server",
        version=CURRENT_PROTOCOL_VERSION,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=settings.max_request_body_bytes,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=["X-Request-ID"],
            max_age=600,
        )

    bearer = HTTPBearer(auto_error=False, bearerFormat="AnyLearning-v1")

    async def require_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> TokenClaims:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise _unauthorized()
        try:
            return signer.verify(credentials.credentials)
        except InvalidTokenError as error:
            raise _unauthorized() from error

    @app.middleware("http")
    async def safe_response_headers(request: Request, call_next: Any):
        request.state.request_id = secrets.token_hex(16)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _error: RequestValidationError):
        # FastAPI's normal validation body includes the rejected input. That is
        # inappropriate for the endpoint whose rejected input is a password.
        return JSONResponse(status_code=422, content={"detail": "Invalid request"})

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, error: Exception):
        logger.error(
            "Unhandled inference server request failure",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "error_type": type(error).__name__,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/auth/token", response_model=TokenResponse)
    async def login(request: Request, body: LoginRequest) -> TokenResponse:
        client_key = _client_key(request)
        retry_after = limiter.admit(client_key)
        if retry_after is not None:
            raise _rate_limited(retry_after)
        if not password_slots.acquire(blocking=False):
            raise _rate_limited(1)
        try:
            loop = asyncio.get_running_loop()
            valid = await loop.run_in_executor(
                password_executor,
                authenticator.verify,
                body.password,
            )
        finally:
            password_slots.release()
        if not valid:
            raise _unauthorized(detail="Invalid credentials")
        limiter.authentication_succeeded(client_key)
        return TokenResponse(
            access_token=signer.issue(),
            expires_in=signer.ttl_seconds,
        )

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models(
        _claims: TokenClaims = Depends(require_token),
    ) -> ModelListResponse:
        return ModelListResponse(models=tuple(catalog.values()))

    @app.get("/v1/models/{model_id}", response_model=ModelCapabilities)
    async def get_model(
        model_id: str,
        _claims: TokenClaims = Depends(require_token),
    ) -> ModelCapabilities:
        model = catalog.get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return model

    return app


def _model_catalog(
    models: Iterable[ModelCapabilities],
) -> dict[str, ModelCapabilities]:
    catalog: dict[str, ModelCapabilities] = {}
    for model in models:
        if not isinstance(model, ModelCapabilities):
            raise TypeError("server models must be ModelCapabilities")
        if model.model_id in catalog:
            raise ValueError("server model identifiers must be unique")
        catalog[model.model_id] = model
        if len(catalog) > _MAX_MODELS:
            raise ValueError(f"server supports at most {_MAX_MODELS} models")
    return dict(sorted(catalog.items()))


def _client_key(request: Request) -> str:
    # Forwarded headers are deliberately ignored. Deployments that terminate
    # TLS at a proxy should also rate-limit there; otherwise every proxy client
    # safely shares this stricter server-side bucket.
    return request.client.host if request.client is not None else "unknown"


def _unauthorized(detail: str = "Not authenticated") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _rate_limited(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many authentication attempts",
        headers={"Retry-After": str(retry_after)},
    )
