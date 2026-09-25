from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.errors import AppError
from app.api.routes_forecast import router as forecast_router
from app.api.routes_measurements import quality_router as data_quality_router
from app.api.routes_measurements import router as measurements_router
from app.api.routes_reports import router as reports_router
from app.api.routes_sites import router as sites_router
from app.api.routes_sizing import router as sizing_router
from app.api.routes_sizing import runs_router as sizing_runs_router
from app.api.routes_storage import router as storage_router
from app.api.routes_storage import runs_router as dispatch_runs_router
from app.api.routes_tariffs import router as tariffs_router
from app.api.routes_weather import router as weather_router
from app.infrastructure.db import Database


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:8]}"


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


def create_app(database_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="光储一体 · 储能配置优化系统",
        version="0.1.0",
        description="工商业光伏储能配置、调度和收益分析后端。",
    )
    app.state.db = Database(database_path or os.getenv("STORAGE_DB_PATH", "data/storage.db"))

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = f"req_{uuid.uuid4().hex[:8]}"
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in error["loc"][1:]),
                "reason": error["msg"],
            }
            for error in exc.errors()
        ]
        return _error_response(request, 422, "VALIDATION_ERROR", "request validation failed", details)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(request, 500, "INTERNAL_ERROR", "internal server error")

    @app.get("/api/v1/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": app.version}

    app.include_router(sites_router, prefix="/api/v1")
    app.include_router(measurements_router, prefix="/api/v1")
    app.include_router(data_quality_router, prefix="/api/v1")
    app.include_router(tariffs_router, prefix="/api/v1")
    app.include_router(storage_router, prefix="/api/v1")
    app.include_router(dispatch_runs_router, prefix="/api/v1")
    app.include_router(forecast_router, prefix="/api/v1")
    app.include_router(sizing_router, prefix="/api/v1")
    app.include_router(sizing_runs_router, prefix="/api/v1")
    app.include_router(reports_router, prefix="/api/v1")
    app.include_router(weather_router, prefix="/api/v1")
    return app


app = create_app()
