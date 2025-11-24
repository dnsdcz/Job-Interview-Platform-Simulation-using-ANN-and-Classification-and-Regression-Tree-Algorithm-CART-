# blueprints/hr.py
from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from extensions import mysql

hr_bp = Blueprint("hr", __name__)


@hr_bp.route("/hr")
def hr_dashboard():
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    cur.execute("SELECT email, username FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM chatbot")
    chatbot_count = cur.fetchone()[0]
    chatbot_progress = 100 if chatbot_count > 0 else 0

    cur.execute("SELECT COUNT(*) FROM applicants")
    applicant_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM applicants WHERE status = 'completed'")
    completed_applications = cur.fetchone()[0]
    applicant_progress = (
        (completed_applications / applicant_count) *
        100 if applicant_count > 0 else 0
    )

    cur.execute("SELECT position, max_allowed FROM position_limits")
    limits = {pos: max_allowed for pos, max_allowed in cur.fetchall()}

    cur.execute(
        "SELECT position, COUNT(*) as current_count FROM applicants GROUP BY position"
    )
    counts = {pos: count for pos, count in cur.fetchall()}

    positions = ["Business Analyst", "Project Analyst", "Java Developer"]
    progress_data = []
    for pos in positions:
        current = counts.get(pos, 0)
        max_allowed = limits.get(pos, 10)
        percentage = (current / max_allowed) * 100 if max_allowed > 0 else 0
        percentage = min(percentage, 100)
        progress_data.append(
            {"position": pos, "percentage": round(
                percentage, 2), "current": current, "max": max_allowed}
        )

    cur.close()
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))

    email, username = user
    return render_template(
        "Hrpage.html",
        email=email,
        username=username,
        chatbot_count=chatbot_count,
        applicant_count=applicant_count,
        chatbot_progress=chatbot_progress,
        applicant_progress=applicant_progress,
        progress_data=progress_data,
    )


@hr_bp.route("/set-username", methods=["POST"])
def set_username():
    if "user_id" not in session:
        flash("You must be logged in to set a username.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    username = request.form["username"]

    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET username = %s WHERE id = %s",
                (username, user_id))
    mysql.connection.commit()
    cur.close()

    flash("Username updated successfully!", "success")
    return redirect(url_for("hr.hr_dashboard"))
