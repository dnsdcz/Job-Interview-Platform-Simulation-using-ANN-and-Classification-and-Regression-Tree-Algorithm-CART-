# blueprints/summary.py
from datetime import datetime
import pdfkit
import os
import json
from flask import (
    Blueprint,
    current_app,
    render_template,
    session,
    request,
    jsonify,
    redirect,
    url_for,
    send_file,
)
from extensions import mysql, logger, limiter
from services.scoring import score_many
from services.pdf_utils import generate_pdf_summary

summary_bp = Blueprint("summary", __name__)


@summary_bp.route("/summary")
def summary_page():
    try:
        return render_template("summary.html")
    except Exception as e:
        logger.error(f"Error rendering summary.html: {e}")
        return f"<h1>Error rendering summary:</h1><pre>{e}</pre>", 500


@summary_bp.route("/summary_report")
def summary_report():
    user_id = session.get("user_id")
    if not user_id:
        return "User session not found.", 400

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT * FROM chatbot WHERE user_id = %s ORDER BY id DESC LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        return "No summary found for this user.", 404

    keys = [desc[0] for desc in cur.description]
    data = dict(zip(keys, row))
    cur.close()

    assessment_data = json.loads(data.get("assessment_data") or "[]")
    advice_list = json.loads(data.get("advice") or "[]")

    return render_template(
        "summary.html",
        name=data.get("user_name"),
        position=data.get("position"),
        skills=data.get("skills"),
        qualification_status=data.get("qualification_status"),
        confidence=data.get("confidence"),
        assessment_data=assessment_data,
        advice_list=advice_list,
    )


@summary_bp.route("/save_summary_report", methods=["POST"])
@limiter.limit("10/minute")
def save_summary_report():
    try:
        data = request.get_json(force=True)
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not logged in or session expired."}), 403

        user_name = data.get("user_name")
        position = data.get("position")
        experience = data.get("experience", "")
        skills = data.get("skills", [])
        qualification_status = data.get("qualification_status", "")
        advice = data.get("advice", [])
        assessment_data = data.get("assessment_data", [])
        confidence = float(data.get("confidence", 0))
        average_score = float(data.get("average_score", 0))

        if not user_name or not position:
            return jsonify({"error": "Missing user_name or position"}), 400

        cur = mysql.connection.cursor()
        cur.execute(
            """
            INSERT INTO chatbot
            (user_id, user_name, position, experience, skills,
             qualification_status, advice, assessment_data,
             confidence, average_score, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
            """,
            (
                user_id,
                user_name,
                position,
                experience,
                json.dumps(skills),
                qualification_status,
                json.dumps(advice),
                json.dumps(assessment_data),
                confidence,
                average_score,
            ),
        )
        mysql.connection.commit()
        cur.close()

        return jsonify({"message": "Summary report saved.", "redirect": url_for("summary.summary_page")}), 201
    except Exception as e:
        logger.error(f"Error in save_summary_report: {e}")
        return jsonify({"error": "Failed to save summary.", "details": str(e)}), 500


@summary_bp.route("/download_summary")
def download_summary():
    user_name = session.get("name", "Candidate")
    position = session.get("position", "Unknown")
    skills = session.get("skills", [])
    status = session.get("qualification_status", "Not Qualified")

    if isinstance(skills, list):
        skills_str = ", ".join(skills)
    else:
        skills_str = str(skills)

    path = generate_pdf_summary(user_name, position, skills_str, status)
    return send_file(path, as_attachment=True)


@summary_bp.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.get_json()

    user_name = data.get("user_name", "Candidate")
    role = data.get("role", "Unknown")
    skills = data.get("skills", "")
    qualification_status = data.get("qualification_status", "Not Qualified")
    advice_list = data.get("advice_list", [])
    assessment_data = data.get("assessment_data", [])

    rendered = render_template(
        "summary.html",
        user_name=user_name,
        role=role,
        skills_str=skills,
        qualification_status=qualification_status,
        advice_list=advice_list,
        assessment_data=assessment_data,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_reports/{user_name}_summary_{timestamp}.pdf"
    os.makedirs("summary_reports", exist_ok=True)

    config = pdfkit.configuration(wkhtmltopdf="/usr/local/bin/wkhtmltopdf")
    pdfkit.from_string(rendered, filename, configuration=config)

    with open(filename, "rb") as f:
        pdf_data = f.read()

    resp = current_app.response_class(pdf_data, mimetype="application/pdf")
    resp.headers[
        "Content-Disposition"] = f"attachment; filename={os.path.basename(filename)}"
    return resp
