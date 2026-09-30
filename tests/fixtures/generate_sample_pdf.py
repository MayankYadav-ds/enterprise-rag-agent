"""Generate the small PDF fixture used by ingestion tests.

Run with ``python tests/fixtures/generate_sample_pdf.py`` after changing this
fixture. The generated binary is intentionally committed because it exercises
the real parser without requiring ReportLab during a test run.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Table, TableStyle

OUTPUT_PATH = Path(__file__).with_name("sample_report.pdf")


def _draw_header_and_footer(canvas: Canvas, page_number: int) -> None:
    """Draw the repeated edge content that the parser should remove."""
    width, height = letter
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(width / 2, height - 0.45 * inch, "ENTERPRISE RAG TEST REPORT")
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(width / 2, 0.4 * inch, f"Page {page_number} of 2")


def _draw_table(canvas: Canvas) -> None:
    """Draw a ruled table that pdfplumber can detect reliably."""
    rows = [
        ["Metric", "FY 2025", "FY 2024"],
        ["Revenue", "$125", "$100"],
        ["Operating margin", "24%", "21%"],
    ]
    table = Table(rows, colWidths=[2.4 * inch, 1.3 * inch, 1.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    table.wrapOn(canvas, 0, 0)
    table.drawOn(canvas, 0.9 * inch, 4.75 * inch)


def create_fixture(output_path: Path = OUTPUT_PATH) -> None:
    """Create a deterministic two-page report with text, headings, and a table."""
    canvas = Canvas(str(output_path), pagesize=letter, invariant=1)
    canvas.setTitle("Sample Ingestion Report")
    width, height = letter

    _draw_header_and_footer(canvas, page_number=1)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(0.9 * inch, height - 1.15 * inch, "Financial Overview")
    canvas.setFont("Helvetica", 11)
    canvas.drawString(0.9 * inch, height - 1.55 * inch, "This hyphen-")
    canvas.drawString(
        0.9 * inch,
        height - 1.75 * inch,
        "ated phrase should be repaired during text cleaning.",
    )
    canvas.drawString(
        0.9 * inch,
        height - 2.05 * inch,
        "The table below is a compact fixture for readable Markdown extraction.",
    )
    _draw_table(canvas)
    canvas.showPage()

    _draw_header_and_footer(canvas, page_number=2)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(0.9 * inch, height - 1.15 * inch, "Operational Notes")
    canvas.setFont("Helvetica", 11)
    canvas.drawString(
        0.9 * inch,
        height - 1.55 * inch,
        "Second-page body text confirms original PDF page numbering is retained.",
    )
    canvas.save()


if __name__ == "__main__":
    create_fixture()
