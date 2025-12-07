# blueprints/hr.py
from flask import (
    Blueprint,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash,
)
from extensions import mysql
import datetime
import smtplib
from email.message import EmailMessage

hr_bp = Blueprint("hr", __name__)

# === SIMPLE SMTP CONFIG – CHANGE THESE TO YOUR REAL VALUES ===
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "your_email@example.com"      # TODO: change
SMTP_PASS = "your_app_password"           # TODO: change
FROM_NAME = "AceView HR"


def send_email(to_address, subject, body):
    """
    Simple SMTP email sender.
    Make sure SMTP_* constants above are configured correctly.
    """
    if not to_address:
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to_address
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception as e:
        # Log the error; don’t crash the request
        print(f"[EMAIL ERROR] Could not send to {to_address}: {e}")


@hr_bp.route("/hr")
def hr_dashboard():
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    # --- User info ---
    cur.execute("SELECT email, username FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))
    email, username = user

  # --- General stats: applicants + chatbot ---

    # Total applicants in applicants table (for your top cards)
    cur.execute("SELECT COUNT(*) FROM applicants")
    applicant_count = cur.fetchone()[0] or 0

    # Total records processed by chatbot
    cur.execute("SELECT COUNT(*) FROM chatbot")
    chatbot_total = cur.fetchone()[0] or 0

    # Only those marked as qualified in chatbot
    cur.execute("""
            SELECT COUNT(*)
            FROM chatbot
            WHERE LOWER(qualification_status) = 'qualified'
        """)
    qualified_total = cur.fetchone()[0] or 0

    # Example: chatbot_progress = "has chatbot run for anyone?"
    chatbot_progress = 100 if chatbot_total > 0 else 0

    # Applicant progress = % of chatbot entries that are qualified
    applicant_progress = (
        (qualified_total / chatbot_total) * 100
        if chatbot_total > 0
        else 0
    )

    # --- Existing per-position progress_data (for your progress bars) ---
    cur.execute("SELECT position, max_allowed FROM position_limits")
    limits = {pos: max_allowed for pos, max_allowed in cur.fetchall()}

    cur.execute(
        "SELECT position, COUNT(*) AS current_count FROM applicants GROUP BY position"
    )
    counts = {pos: count for pos, count in cur.fetchall()}

    positions = ["Business Analyst", "Project Analyst", "Java Developer"]
    progress_data = []
    for pos in positions:
        current = counts.get(pos, 0)
        max_allowed = limits.get(pos, 10)
        percentage = (current / max_allowed) * \
            100 if max_allowed > 0 else 0
        percentage = min(percentage, 100)
        progress_data.append(
            {
                "position": pos,
                "percentage": round(percentage, 2),
                "current": current,
                "max": max_allowed,
            }
        )

     # --- Jobs overview data for the dashboard table ---
    cur.execute(
        """
        SELECT 
            pl.position,
            pl.opening_date,
            pl.deadline_date,
            pl.max_allowed,
            pl.form_access,
            -- count only qualified applicants from chatbot
            COUNT(
                CASE 
                    WHEN LOWER(c.qualification_status) = 'qualified' 
                    THEN 1 
                    ELSE NULL 
                END
            ) AS applicant_count
        FROM position_limits pl
        LEFT JOIN applicants a 
            ON a.position = pl.position
        LEFT JOIN chatbot c
            ON c.user_id = a.user_id
        GROUP BY 
            pl.position, pl.opening_date, pl.deadline_date, 
            pl.max_allowed, pl.form_access
        ORDER BY 
            CASE WHEN pl.opening_date IS NULL THEN 1 ELSE 0 END,
            pl.opening_date
        """
    )
    job_rows = cur.fetchall()

    today = datetime.date.today()
    jobs = []

    for row in job_rows:
        position, opening_date, deadline_date, max_allowed, form_access, applicant_cnt = row

        # Ensure dates are date objects
        if isinstance(opening_date, datetime.datetime):
            opening_date = opening_date.date()
        if isinstance(deadline_date, datetime.datetime):
            deadline_date = deadline_date.date()

        # Status shown in the "Status" column
        if deadline_date and deadline_date < today:
            status = "Closed"
        elif max_allowed and applicant_cnt >= max_allowed:
            status = "On hold"
        else:
            status = "In progress"

        openings = max_allowed or 0
        closed_fill = min(applicant_cnt, openings) if openings > 0 else 0

        # time_to_hire kept for potential use (not shown in table now)
        if opening_date and deadline_date:
            days = (deadline_date - opening_date).days
            time_to_hire = f"{days:02d} days"
        else:
            time_to_hire = "—"

        jobs.append(
            {
                "title": position,
                "deadline": deadline_date.strftime("%b %d, %Y") if deadline_date else "—",
                "status": status,
                "openings": openings,
                "closed": closed_fill,
                "applicants": applicant_cnt,   # this is now "qualified applicants"
                "time_to_hire": time_to_hire,
                "decision": form_access or "",  # '', 'approved', 'denied'
            }
        )

    cur.close()

    return render_template(
        "Hrpage.html",
        email=email,
        username=username,
        chatbot_total=chatbot_total,
        applicant_count=applicant_count,
        chatbot_progress=chatbot_progress,
        applicant_progress=applicant_progress,
        progress_data=progress_data,
        jobs=jobs,
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


@hr_bp.route("/job-decision", methods=["POST"])
def job_decision():
    """
    Handle Approve / Deny buttons for a job row.
    If approved, send email to all applicants for that position.
    """
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    position = request.form.get("position")
    decision = request.form.get("decision")  # "approve" or "deny"

    if not position or decision not in ("approve", "deny"):
        flash("Invalid action.", "error")
        return redirect(url_for("hr.hr_dashboard"))

    cur = mysql.connection.cursor()

    # Map decision to a status string stored in position_limits.form_access
    status_value = "approved" if decision == "approve" else "denied"

    try:
        cur.execute(
            "UPDATE position_limits SET form_access = %s WHERE position = %s",
            (status_value, position),
        )
        mysql.connection.commit()
    except Exception as e:
        print("[DB ERROR] Updating form_access failed:", e)

    # If approved, notify applicants by email
    if decision == "approve":
        # Use your real fields: name + email
        cur.execute(
            "SELECT email, name FROM applicants WHERE position = %s",
            (position,),
        )
        rows = cur.fetchall()

        for email_addr, full_name in rows:
            subject = f"Application update for {position}"
            body = (
                f"Dear {full_name},\n\n"
                f"Your application for the position '{position}' has been approved "
                f"by our HR team to proceed to the next step of the process.\n\n"
                f"Please wait for further instructions regarding scheduling.\n\n"
                f"Best regards,\nAceView HR"
            )
            send_email(email_addr, subject, body)

        flash(
            f"{position} approved. Notification emails sent to applicants.", "success")
    else:
        flash(f"{position} marked as denied.", "info")

    cur.close()
    return redirect(url_for("hr.hr_dashboard"))
