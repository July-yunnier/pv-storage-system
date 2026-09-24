"""单站 Sizing 报告 PDF 构建（reportlab，中文用内置 CID 字体 STSong-Light）。"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.domain.models import SiteRead, TariffRead
from app.domain.sizing.result import SizingRunDetail

FONT = "STSong-Light"
_font_registered = False


def _ensure_font() -> None:
    global _font_registered
    if not _font_registered:
        pdfmetrics.registerFont(UnicodeCIDFont(FONT))
        _font_registered = True


def _style(size: int = 10, leading: int | None = None, bold: bool = False) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"s{size}{'b' if bold else ''}",
        fontName=FONT,
        fontSize=size,
        leading=leading or size * 1.5,
        spaceAfter=4,
    )


def _table(rows: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF4")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def build_sizing_pdf(
    site: SiteRead, tariff: TariffRead | None, run: SizingRunDetail, path: Path
) -> None:
    _ensure_font()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"sizing report {run.run_id}",
    )
    story = []

    story.append(Paragraph("储能容量规划报告（Sizing Report v1）", _style(18, bold=True)))
    story.append(Spacer(1, 4 * mm))
    story.append(
        _table(
            [
                ["站点", f"{site.name}（{site.site_id}）"],
                ["报告日期", (run.created_at.date().isoformat() if run.created_at else "-")],
                ["Sizing Run", run.run_id],
                ["Input Digest", run.input_digest],
            ],
            col_widths=[35 * mm, 130 * mm],
        )
    )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("一、输入摘要", _style(13, bold=True)))
    storage = site.storage
    tariff_desc = (
        f"{tariff.name}（{tariff.tariff_id}），输配电及基金 {tariff.transmission_and_fund} 元/kWh"
        if tariff
        else f"{site.tariff_id}（未能读取电价详情）"
    )
    story.append(
        _table(
            [
                ["光伏容量", f"{site.pv.capacity_kwp} kWp，系统损耗 {site.pv.system_loss}"],
                ["现有储能", f"{storage.pcs_kw} kW / {storage.capacity_kwh} kWh，RTE {storage.round_trip_eff}"],
                ["电价", tariff_desc],
                ["代表日", "、".join(d.isoformat() for d in run.representative_dates)],
            ],
            col_widths=[35 * mm, 130 * mm],
        )
    )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("二、推荐方案", _style(13, bold=True)))
    rec = run.recommended
    if rec is None:
        story.append(Paragraph(f"无推荐方案。{run.notes}", _style()))
    else:
        story.append(
            _table(
                [
                    ["PCS 功率", f"{rec.pcs_kw} kW"],
                    ["电池容量", f"{rec.capacity_kwh} kWh（{rec.duration_h}h）"],
                    ["CAPEX", f"{_fmt(rec.capex_yuan)} 元"],
                    ["年收益 / 年 OPEX", f"{_fmt(rec.annual_benefit_yuan)} / {_fmt(rec.annual_opex_yuan)} 元"],
                    ["NPV / IRR", f"{_fmt(rec.npv_yuan)} 元 / {_fmt(rec.irr * 100 if rec.irr is not None else None)}%"],
                    ["简单回本年限", f"{_fmt(rec.payback_years)} 年"],
                ],
                col_widths=[45 * mm, 120 * mm],
            )
        )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("三、候选对比", _style(13, bold=True)))
    rows = [["PCS(kW)", "容量(kWh)", "CAPEX(元)", "年收益(元)", "NPV(元)", "IRR", "回本(年)", "循环/日"]]
    for c in run.candidates:
        rows.append(
            [
                f"{c.pcs_kw:g}",
                f"{c.capacity_kwh:g}",
                _fmt(c.capex_yuan, 0),
                _fmt(c.annual_benefit_yuan, 0),
                _fmt(c.npv_yuan, 0),
                "-" if c.irr is None else f"{c.irr:.2%}",
                _fmt(c.payback_years),
                f"{c.equivalent_full_cycles_per_day:.2f}",
            ]
        )
    story.append(_table(rows))
    story.append(Spacer(1, 6 * mm))

    if run.sensitivity.price_spread or run.sensitivity.battery_price:
        story.append(Paragraph("四、敏感性分析（基于推荐方案）", _style(13, bold=True)))
        for title, entries in (
            ("峰谷价差缩放", run.sensitivity.price_spread),
            ("电池单价缩放", run.sensitivity.battery_price),
        ):
            if not entries:
                continue
            story.append(Paragraph(title, _style(10, bold=True)))
            rows = [["变动", "年收益(元)", "NPV(元)", "回本(年)"]]
            for entry in entries:
                rows.append(
                    [
                        f"{entry.delta:+.0%}",
                        _fmt(entry.annual_benefit_yuan, 0),
                        _fmt(entry.npv_yuan, 0),
                        _fmt(entry.payback_years),
                    ]
                )
            story.append(_table(rows))
        story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("五、完整假设清单", _style(13, bold=True)))
    rows = [["参数", "取值"]]
    for key, value in run.assumptions.items():
        rows.append([key, str(value)])
    story.append(_table(rows, col_widths=[70 * mm, 95 * mm]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("六、限制说明", _style(13, bold=True)))
    for line in (
        "年化口径：典型日算术平均×365，未考虑季节差异与极端天气。",
        "调度口径：规则式策略 rule_v1，非全局最优；实际运行受 EMS 执行精度影响。",
        "电价口径：按当前 TOU 政策电价静态展开，未考虑电价政策调整风险。",
        "收益未计需量电费变化、补贴、税费与融资成本；NPV 优先于简单回本年限解读。",
    ):
        story.append(Paragraph(f"· {line}", _style()))

    doc.build(story)
