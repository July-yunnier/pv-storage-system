from __future__ import annotations

from typing import Any


class AppError(Exception):
    """统一业务异常，由全局异常处理器转换为契约错误结构。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details


def not_found(message: str, details: list[dict[str, Any]] | None = None) -> AppError:
    return AppError(message, status_code=404, code="NOT_FOUND", details=details)


def validation_error(message: str, details: list[dict[str, Any]] | None = None) -> AppError:
    return AppError(message, status_code=422, code="VALIDATION_ERROR", details=details)


def data_quality_error(message: str, details: list[dict[str, Any]] | None = None) -> AppError:
    return AppError(message, status_code=422, code="DATA_QUALITY_ERROR", details=details)


def conflict(message: str, details: list[dict[str, Any]] | None = None) -> AppError:
    return AppError(message, status_code=409, code="CONFLICT", details=details)


def provider_error(message: str, details: list[dict[str, Any]] | None = None) -> AppError:
    return AppError(message, status_code=502, code="PROVIDER_ERROR", details=details)
