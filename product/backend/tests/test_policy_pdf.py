import os

import pytest

from app.infrastructure import policy_llm


def test_auto_pdf_uses_pypdf_when_text_is_sufficient(monkeypatch) -> None:
    monkeypatch.setenv("POLICY_PDF_EXTRACTOR", "auto")
    monkeypatch.setenv("POLICY_PDF_MIN_TEXT_CHARS", "10")
    monkeypatch.setattr(policy_llm, "_extract_with_pypdf", lambda raw: "足够长的政策文本内容")
    monkeypatch.setattr(
        policy_llm,
        "_extract_with_paddleocr",
        lambda raw: pytest.fail("不应调用 OCR"),
    )
    assert policy_llm.extract_text_from_pdf(b"pdf") == "足够长的政策文本内容"


def test_auto_pdf_falls_back_to_paddleocr(monkeypatch) -> None:
    monkeypatch.setenv("POLICY_PDF_EXTRACTOR", "auto")
    monkeypatch.setenv("POLICY_PDF_MIN_TEXT_CHARS", "80")
    monkeypatch.setattr(policy_llm, "_extract_with_pypdf", lambda raw: "")
    monkeypatch.setattr(policy_llm, "_extract_with_paddleocr", lambda raw: "OCR政策文本")
    assert policy_llm.extract_text_from_pdf(b"pdf") == "OCR政策文本"


def test_invalid_pdf_extractor_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("POLICY_PDF_EXTRACTOR", "unknown")
    with pytest.raises(ValueError, match="POLICY_PDF_EXTRACTOR"):
        policy_llm.extract_text_from_pdf(b"pdf")


def test_local_ollama_example_is_complete() -> None:
    path = policy_llm.__file__
    backend = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    content = open(os.path.join(backend, ".env.ai.example"), encoding="utf-8").read()
    assert "LLM_BASE_URL=http://127.0.0.1:11434/v1" in content
    assert "LLM_MODEL=qwen2.5:7b" in content
    assert "POLICY_PDF_EXTRACTOR=auto" in content
