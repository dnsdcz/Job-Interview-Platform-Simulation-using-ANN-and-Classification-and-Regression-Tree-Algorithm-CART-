# services/pdf_utils.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fpdf import FPDF


class InterviewPDF(FPDF):
    def header(self):
        # Logo (adjust path & size as needed)
        # This is a filesystem path, e.g. job-interview-platform/static/img/logo.png
        try:
            self.image("static/img/logo.png", x=10, y=8, w=25)
        except Exception:
            # If logo is missing, just skip it
            pass

        # Move cursor down a bit and center the title
        self.set_xy(0, 10)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Summary Report", ln=True, align="C")
        self.ln(5)


    def footer(self):
        # Page number at the bottom
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def generate_interview_pdf(
    filename: str,
    candidate_name: str,
    position: str,
    qualification_status: str,
    result_message: str,
    match_score_pct: Optional[float] = None,
    average_score: Optional[float] = None,
    remarks: Optional[list[str]] = None,
    output_dir: str = "summary_reports",
) -> str:
    """
    Generate a nicely formatted interview PDF and return the file path.

    Args:
        filename: base filename (without path).
        candidate_name: candidate's full name.
        position: applied position.
        qualification_status: 'Qualified', 'Partially Qualified', etc.
        result_message: main text you show in the summary.
        match_score_pct: e.g. 83.0
        average_score: e.g. 0.78
        remarks: optional list of bullet-point remarks.
        output_dir: directory where PDFs will be saved.

    Returns:
        Full path to the generated PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    pdf = InterviewPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Basic info
    pdf.set_font("Helvetica", "", 12)

    today_str = datetime.now().strftime("%B %d, %Y")

    pdf.cell(0, 8, f"Candidate Name: {candidate_name}", ln=True)
    pdf.cell(0, 8, f"Position: {position}", ln=True)
    pdf.cell(0, 8, f"Date: {today_str}", ln=True)
    pdf.ln(4)

    # Result summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Overall Result:", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Qualification Status: {qualification_status}", ln=True)

    if match_score_pct is not None:
        pdf.cell(0, 8, f"Match Score: {match_score_pct:.0f}%", ln=True)
    if average_score is not None:
        pdf.cell(0, 8, f"Average Score: {average_score:.2f}", ln=True)

    pdf.ln(4)

    # Result message (your “Congratulations…” text)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Result Message:", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 6, result_message)
    pdf.ln(4)

    # Top skills / remarks
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top Skills / Remarks:", ln=True)
    pdf.set_font("Helvetica", "", 12)

    if remarks:
        for r in remarks:
            pdf.multi_cell(0, 6, f"- {r}")
    else:
        pdf.multi_cell(0, 6, "- No additional remarks provided.")

    pdf.output(filepath)
    return filepath
