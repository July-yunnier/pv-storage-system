from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import BaseModel, Field

from app.api import errors
from app.api.dependencies import get_db
from app.domain.models import TariffRead
from app.domain.policy_tariff import (
    PolicyConfirmRequest,
    PolicyParseResponse,
    candidate_to_tariff,
    validate_policy_candidate,
)
from app.infrastructure.db import Database
from app.infrastructure.policy_llm import extract_text_from_pdf, parse_policy_text_with_llm

router = APIRouter(prefix="/policy-tariffs", tags=["tariffs"])

MAX_PDF_BYTES = 15 * 1024 * 1024


class PolicyTextParseRequest(BaseModel):
    policy_name: str = Field(default="待复核分时电价政策", min_length=1, max_length=200)
    policy_text: str = Field(min_length=1)


def _parse(policy_text: str, policy_name: str) -> PolicyParseResponse:
    try:
        candidate = parse_policy_text_with_llm(policy_text, policy_name)
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    except RuntimeError as exc:
        raise errors.provider_error(str(exc)) from exc
    return PolicyParseResponse(
        candidate=candidate,
        validation_issues=validate_policy_candidate(candidate),
        requires_human_confirmation=True,
    )


@router.post("/parse-text", response_model=PolicyParseResponse)
def parse_policy_text(payload: PolicyTextParseRequest) -> PolicyParseResponse:
    """解析政策文本为候选规则；此接口不会写数据库。"""
    return _parse(payload.policy_text, payload.policy_name)


@router.post("/parse-pdf", response_model=PolicyParseResponse)
async def parse_policy_pdf(
    file: UploadFile = File(...),
    policy_name: str = Form(default="待复核分时电价政策"),
) -> PolicyParseResponse:
    """提取文本型 PDF 并解析；扫描件应由外部 OCR 接口先转成文本。"""
    raw = await file.read(MAX_PDF_BYTES + 1)
    if not raw:
        raise errors.validation_error("pdf file is empty")
    if len(raw) > MAX_PDF_BYTES:
        raise errors.validation_error("pdf file exceeds 15 MB")
    filename = file.filename or ""
    if file.content_type not in (None, "application/pdf") and not filename.lower().endswith(".pdf"):
        raise errors.validation_error("file must be a PDF")
    try:
        text = extract_text_from_pdf(raw)
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    except RuntimeError as exc:
        raise errors.provider_error(str(exc)) from exc
    return _parse(text, policy_name)


@router.post(
    "/confirm",
    response_model=TariffRead,
    status_code=status.HTTP_201_CREATED,
)
def confirm_policy_tariff(
    payload: PolicyConfirmRequest,
    db: Database = Depends(get_db),
) -> TariffRead:
    """人工确认后才把严格校验通过的候选写入系统价目。"""
    try:
        tariff = candidate_to_tariff(
            payload.candidate,
            tariff_id=payload.tariff_id,
            target_date=payload.target_date,
        )
        return db.create_tariff(tariff)
    except ValueError as exc:
        message = str(exc)
        if "already exists" in message:
            raise errors.conflict(message) from exc
        raise errors.validation_error(message) from exc
