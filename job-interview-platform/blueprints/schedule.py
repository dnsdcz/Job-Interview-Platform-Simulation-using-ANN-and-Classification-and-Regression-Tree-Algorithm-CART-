# blueprints/schedule.py
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
import json
from extensions import mysql

schedule_bp = Blueprint("schedule", __name__)


def get_progress_data():
    """
    Reusable helper that returns a list of dicts like:
    {
        "position": "Business Analyst",
        "current": 5,
        "max_allowed": 10,
        "percent": 50
    }
    """
    cur = mysql.connection.cursor()

    # Limits per position (main recruitment limits)
    cur.execute("SELECT position, max_allowed FROM position_limits")
    position_limits = cur.fetchall()

    # Limits for chatbot interactions (if you track them separately)
    cur.execute("SELECT position, max_allowed FROM chatbot_limits")
    chatbot_limits = cur.fetchall()

    # Actual applicant counts per position
    cur.execute(
        "SELECT position, COUNT(*) AS current_count FROM applicants GROUP BY position")
    counts = {row[0]: row[1] for row in cur.fetchall()}

    progress_data = []

    # Normal positions
    for pos, max_allowed in position_limits:
        current = counts.get(pos, 0)
        percent = int((current / max_allowed) *
                      100) if max_allowed and max_allowed > 0 else 0
        progress_data.append(
            {
                "position": pos,
                "current": current,
                "max_allowed": max_allowed,
                "percent": percent,
            }
        )

    # Chatbot “position”
    for pos, max_allowed in chatbot_limits:
        current = counts.get(pos, 0)
        percent = int((current / max_allowed) *
                      100) if max_allowed and max_allowed > 0 else 0
        progress_data.append(
            {
                "position": "Chatbot Interactions",
                "current": current,
                "max_allowed": max_allowed,
                "percent": percent,
            }
        )

    cur.close()
    return progress_data


@schedule_bp.route("/schedule")
def schedule_page():
    """If you still want the old blue page to work."""
    progress_data = get_progress_data()
    return render_template("Settinghr.html", progress_data=progress_data)


# --- EXTENDED REQUIREMENTS (Recruitment tab & old page both use this) ---
@schedule_bp.route("/update_requirements", methods=["POST"])
def update_requirements():
    print("--- DEBUG: FORM SUBMITTED ---")

    # Core
    position = request.form.get("position")
    max_allowed = request.form.get("max_allowed")
    form_access = request.form.get("form_access")

    # Timeline
    opening_date = request.form.get("opening_date") or None
    deadline_date = request.form.get("deadline_date") or None

    # Education & experience
    education_level = request.form.get("education_level")
    school = request.form.get("school")
    experience_years = request.form.get("experience_years")

    # Demographics & details
    min_age = request.form.get("min_age")
    location = request.form.get("location")
    employment_type = request.form.get("employment_type")

    # Safe numeric conversion
    exp_val = int(
        experience_years) if experience_years and experience_years.isdigit() else 0
    age_val = int(min_age) if min_age and min_age.isdigit() else 18

    print(
        f"Received Data: Pos={position}, Max={max_allowed}, School={school}, Loc={location}, Open={opening_date}"
    )

    if not position or not max_allowed:
        flash("Position and Max Limit are required fields.", "danger")
        # 🔁 back to where form came from (HR tab or old page)
        return redirect(request.referrer or url_for("hr.hr_dashboard"))

    try:
        cur = mysql.connection.cursor()

        query = """
            INSERT INTO position_limits 
            (position, max_allowed, form_access, opening_date, deadline_date, 
             education_level, target_school, experience_years, min_age, location, employment_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                max_allowed = VALUES(max_allowed),
                form_access = VALUES(form_access),
                opening_date = VALUES(opening_date),
                deadline_date = VALUES(deadline_date),
                education_level = VALUES(education_level),
                target_school = VALUES(target_school),
                experience_years = VALUES(experience_years),
                min_age = VALUES(min_age),
                location = VALUES(location),
                employment_type = VALUES(employment_type)
        """

        cur.execute(
            query,
            (
                position,
                max_allowed,
                form_access,
                opening_date,
                deadline_date,
                education_level,
                school,
                exp_val,
                age_val,
                location,
                employment_type,
            ),
        )

        mysql.connection.commit()
        cur.close()
        print("--- SUCCESS: Database Updated ---")
        flash(f"Requirements for {position} updated successfully!", "success")

    except Exception as e:
        print(f"--- ERROR: {e} ---")
        flash(f"Error updating requirements: {e}", "danger")

    # 🔁 back to HR page or wherever the form came from
    return redirect(request.referrer or url_for("hr.hr_dashboard"))


@schedule_bp.route("/set_pax", methods=["POST"])
def set_pax():
    position = request.form.get("position")
    max_allowed = request.form.get("max_allowed")

    if not position or not max_allowed:
        flash("Position and Max Applicants are required.", "danger")
        return redirect(request.referrer or url_for("hr.hr_dashboard"))

    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            INSERT INTO position_limits (position, max_allowed)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE max_allowed = VALUES(max_allowed)
            """,
            (position, max_allowed),
        )
        mysql.connection.commit()
        cur.close()
        flash("Max applicants limit set successfully!", "success")
    except Exception as e:
        flash(f"Error setting max limit: {e}", "danger")

    return redirect(request.referrer or url_for("hr.hr_dashboard"))


@schedule_bp.route("/set_chat", methods=["POST"])
def set_chat():
    position = request.form.get("position")
    max_allowed = request.form.get("max_allowed")

    if not position or not max_allowed:
        flash("All fields are required!", "danger")
        return redirect(request.referrer or url_for("hr.hr_dashboard"))

    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM chatbot_limits WHERE position = %s", (position,))
    existing = cur.fetchone()
    if existing:
        cur.execute(
            "UPDATE chatbot_limits SET max_allowed = %s WHERE position = %s",
            (max_allowed, position),
        )
    else:
        cur.execute(
            "INSERT INTO chatbot_limits (position, max_allowed) VALUES (%s, %s)",
            (position, max_allowed),
        )

    mysql.connection.commit()
    cur.close()
    flash(f"Limit set for {position} successfully!", "success")
    return redirect(request.referrer or url_for("hr.hr_dashboard"))


@schedule_bp.route("/save_schedule", methods=["POST"])
def save_schedule():
    date = request.form.get("date")
    time = request.form.get("time")
    end_date = request.form.get("endDate") or None
    recurring_days = request.form.getlist("recurring")
    recurring_json = json.dumps(recurring_days) if recurring_days else None

    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            INSERT INTO schedules (schedule_date, schedule_time, recurring_days, end_date)
            VALUES (%s, %s, %s, %s)
            """,
            (date, time, recurring_json, end_date),
        )
        mysql.connection.commit()
        cur.close()
        flash("Schedule saved successfully!", "success")
    except Exception as e:
        flash(f"Error saving schedule: {e}", "danger")

    return redirect(request.referrer or url_for("hr.hr_dashboard"))
