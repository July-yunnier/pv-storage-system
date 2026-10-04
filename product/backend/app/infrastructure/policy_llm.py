"""政策 PDF 文本提取与 LLM 候选解析。

所有云端参数均来自环境变量，不在代码中写入密钥或固定供应商地址：
``LLM_API_KEY``、``LLM_BASE_URL``、``LLM_MODEL``。
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Any

from app.domain.policy_tariff import PolicyTariffCandidate


def _extract_with_pypdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # 可选能力不阻塞基础后端启动
        raise RuntimeError("缺少 pypdf，请安装 requirements-ai.txt") from exc

    reader = PdfReader(io.BytesIO(raw))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n--- 第 {number} 页 ---\n{text.strip()}")
    return "".join(pages).strip()


def _ocr_text_item(item: Any) -> str | None:
    """兼容 PaddleOCR 2.x/3.x 常见返回结构，提取一条文字。"""
    if isinstance(item, dict):
        text = item.get("rec_text") or item.get("text")
        return str(text).strip() if text else None
    if isinstance(item, (list, tuple)):
        if len(item) >= 2 and isinstance(item[1], (list, tuple)) and item[1]:
            text = item[1][0]
            if isinstance(text, str):
                return text.strip()
        if item and all(isinstance(part, str) for part in item):
            return " ".join(item).strip()
    return None


def _extract_with_paddleocr(raw: bytes) -> str:
    try:
        import fitz
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "扫描版 PDF 需要 PaddleOCR，请安装 requirements-ocr.txt"
        ) from exc

    lang = os.getenv("PADDLEOCR_LANG", "ch")
    dpi = int(os.getenv("PADDLEOCR_DPI", "200"))
    if not 72 <= dpi <= 400:
        raise ValueError("PADDLEOCR_DPI must be between 72 and 400")
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    document = fitz.open(stream=raw, filetype="pdf")
    pages: list[str] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            suffix = ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix) as image_file:
                pixmap.save(image_file.name)
                result = ocr.ocr(image_file.name, cls=True)
            lines: list[str] = []
            for group in result or []:
                if isinstance(group, dict):
                    texts = group.get("rec_texts") or []
                    lines.extend(str(text).strip() for text in texts if str(text).strip())
                    continue
                for item in group or []:
                    text = _ocr_text_item(item)
                    if text:
                        lines.append(text)
            if lines:
                pages.append(f"\n--- 第 {number} 页 OCR ---\n" + "\n".join(lines))
    finally:
        document.close()
    return "".join(pages).strip()


def extract_text_from_pdf(raw: bytes) -> str:
    """按配置提取 PDF：默认 pypdf 优先，文本不足则本地 PaddleOCR。"""
    extractor = os.getenv("POLICY_PDF_EXTRACTOR", "auto").strip().lower()
    if extractor not in {"auto", "pypdf", "paddleocr"}:
        raise ValueError(
            "POLICY_PDF_EXTRACTOR must be one of auto, pypdf, paddleocr"
        )
    min_chars = int(os.getenv("POLICY_PDF_MIN_TEXT_CHARS", "80"))
    if min_chars < 0:
        raise ValueError("POLICY_PDF_MIN_TEXT_CHARS must be non-negative")

    if extractor == "paddleocr":
        text = _extract_with_paddleocr(raw)
    else:
        text = _extract_with_pypdf(raw)
        if extractor == "auto" and len(text) < min_chars:
            text = _extract_with_paddleocr(raw)
    if not text:
        raise ValueError("PDF 未提取到文字，请检查文件质量或 OCR 配置")
    return text


def _prompt(policy_text: str, policy_name: str) -> str:
    schema = {
        "policy_name": policy_name,
        "policy_document_no": None,
        "currency": "CNY",
        "base_price": None,
        "transmission_and_fund": 0.0,
        "published_date": None,
        "effective_from": None,
        "effective_to": None,
        "applicable_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "day_types": ["all"],
        "voltage_level": None,
        "customer_category": None,
        "capacity_charge_yuan_per_kva_month": None,
        "demand_charge_yuan_per_kw_month": None,
        "rules": [
            {
                "start_time": "00:00",
                "end_time": "08:00",
                "type": "valley",
                "multiplier": 0.5,
                "energy_addition": 0.0,
            }
        ],
        "issues": [],
        "evidence": [{"page": 1, "quote": "原文短句", "field": "base_price"}],
    }
    return f"""
你是电力政策结构化助手。请从政策原文提取候选配置，只输出一个合法 JSON 对象。

硬性要求：
1. 不得猜测、补写或沿用示例数字；原文无法确认的关键字段必须填 null，并写入 issues。
2. type 只能是 sharp、peak、flat、valley（原文“尖峰”写 sharp）；时间必须为 15 分钟边界，日末可写 24:00。
3. rules 应覆盖 00:00 至 24:00 且不重叠；跨午夜时段优先拆成两段。
4. energy_addition 仅填写按 kWh 收取的附加项。容量费、需量费必须放在各自字段，不能加到度电价格。
5. 每个关键结论尽量提供 evidence，quote 必须是政策原文，page 来自文本中的页码标记。
6. 日期使用 YYYY-MM-DD；若只知道月份或无法确定，填 null 并说明。

输出字段结构如下。下面仅表示字段和数据类型，不代表政策事实：
{json.dumps(schema, ensure_ascii=False, indent=2)}

政策名称：{policy_name}
政策原文：
{policy_text}
""".strip()


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"未配置环境变量 {name}")
    return value


def parse_policy_text_with_llm(
    policy_text: str, policy_name: str = "待复核分时电价政策"
) -> PolicyTariffCandidate:
    if not policy_text.strip():
        raise ValueError("policy_text must not be empty")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai，请安装 requirements-ai.txt") from exc

    client = OpenAI(
        api_key=_required_env("LLM_API_KEY"),
        base_url=_required_env("LLM_BASE_URL"),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )
    response = client.chat.completions.create(
        model=_required_env("LLM_MODEL"),
        messages=[{"role": "user", "content": _prompt(policy_text, policy_name)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM 返回了空内容")
    try:
        payload: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM 未返回合法 JSON") from exc
    return PolicyTariffCandidate.model_validate(payload)
