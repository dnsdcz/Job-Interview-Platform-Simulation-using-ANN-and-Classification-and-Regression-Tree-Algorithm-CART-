# services/pdf_utils.py
import os
from datetime import datetime

from flask import current_app
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf_summary(user_name: str, position: str, qualifications: str, status: str) -> str:
    """
    Creates a simple PDF summary and returns its file path.
    """
    summary_dir = current_app.config["SUMMARY_REPORT_DIR"]
    os.makedirs(summary_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_name}_{position}_{timestamp}.pdf"
    filepath = os.path.join(summary_dir, filename)

    qualifications = qualifications or "No qualifications provided"

    c = canvas.Canvas(filepath, pagesize=letter)
    text = c.beginText(50, 750)
    text.setFont("Helvetica", 12)
    text.textLine(f"Candidate Name: {user_name}")
    text.textLine(f"Position: {position}")
    text.textLine(f"Qualification Status: {status}")
    text.textLine("Top Skills / Remarks:")
    for line in qualifications.split(","):
        text.textLine(f"- {line.strip()}")
    c.drawText(text)
    c.showPage()
    c.save()

    return filepath
