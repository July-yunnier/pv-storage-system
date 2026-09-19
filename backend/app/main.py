from __future__ import annotations

import os

from fastapi import FastAPI

from app.api.routes_measurements import router as measurements_router
from app.api.routes_sites import router as sites_router
from app.api.routes_storage import router as storage_router
from app.api.routes_tariffs import router as tariffs_router
from app.infrastructure.db import Database


def create_app(database_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="光储一体 · 储能配置优化系统",
        version="0.1.0",
        description="工商业光伏储能配置、调度和收益分析后端。",
    )
    app.state.db = Database(database_path or os.getenv("STORAGE_DB_PATH", "data/storage.db"))

    @app.get("/api/v1/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": app.version}

    app.include_router(sites_router, prefix="/api/v1")
    app.include_router(measurements_router, prefix="/api/v1")
    app.include_router(tariffs_router, prefix="/api/v1")
    app.include_router(storage_router, prefix="/api/v1")
    return app


app = create_app()
