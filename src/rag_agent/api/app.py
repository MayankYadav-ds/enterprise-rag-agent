"""Application factory with lazy services and sanitized structured request logs."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from rag_agent.api.document_routes import router as document_router
from rag_agent.api.documents import DocumentError
from rag_agent.api.routes import router
from rag_agent.api.runtime import DependencyUnavailable, Runtime
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import ensure_auth_schema
from rag_agent.auth.password import dummy_password_hash
from rag_agent.auth.routes import router as auth_router
from rag_agent.generation import GenerationError, InvalidResponseError, ProviderError

logger = logging.getLogger("rag_agent.api")


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Build the API without contacting Qdrant, Ollama, or model registries."""
    runtime = Runtime(settings or APISettings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(runtime.settings.log_level)
        try:
            # Keep the first unknown-user login from paying an extra hash-generation cost.
            dummy_password_hash()
            ensure_auth_schema(runtime)
            yield
        finally:
            runtime.close()
            logger.removeHandler(handler)
            handler.close()
            logger.setLevel(old_level)

    app = FastAPI(
        title="Enterprise Multi-Document RAG Agent",
        version="0.1.0",
        description="Evidence-constrained answers and retrieval with source/page citations.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.include_router(auth_router)
    app.include_router(router)
    app.include_router(document_router)

    def error(request: Request, status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={"code": code, "message": message, "request_id": request.state.request_id},
        )

    @app.middleware("http")
    async def request_logging(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            response = error(request, 500, "internal_error", "Internal server error")
        response.headers["X-Request-ID"] = request.state.request_id
        route = request.scope.get("route")
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "route": getattr(route, "path", "unmatched"),
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            )
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Default Pydantic errors include input values; never echo request contents.
        return error(request, 422, "invalid_request", "Request does not match the API schema")

    @app.exception_handler(DocumentError)
    async def document_error(request: Request, exc: DocumentError) -> JSONResponse:
        return error(request, exc.status, exc.code, str(exc))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        response = error(request, exc.status_code, "http_error", "Request cannot be processed")
        if exc.headers and "Retry-After" in exc.headers:
            response.headers["Retry-After"] = exc.headers["Retry-After"]
        return response

    @app.exception_handler(DependencyUnavailable)
    @app.exception_handler(ProviderError)
    async def unavailable(request: Request, exc: Exception) -> JSONResponse:
        return error(
            request, 503, "dependency_unavailable", "Required service or model unavailable"
        )

    @app.exception_handler(GenerationError)
    @app.exception_handler(InvalidResponseError)
    async def invalid_generation(request: Request, exc: Exception) -> JSONResponse:
        return error(request, 502, "generation_failed", "Generation did not produce a valid answer")

    return app
