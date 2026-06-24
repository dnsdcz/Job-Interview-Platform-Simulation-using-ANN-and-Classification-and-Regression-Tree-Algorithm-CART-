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

    cur.execute(
        """
        SELECT j.job_name, j.max_applicants, COUNT(a.id) AS current_count
        FROM jobs j
        LEFT JOIN applications a ON a.job_id = j.id
        GROUP BY j.id, j.job_name, j.max_applicants
        ORDER BY j.job_name ASC
        """
    )
    rows = cur.fetchall()
    progress_data = []

    for pos, max_allowed, current in rows:
        max_allowed = max_allowed or 0
        current = current or 0
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

        cur.execute(
            "SELECT id FROM educations WHERE education_name = %s",
            (education_level,),
        )
        edu_row = cur.fetchone()
        education_id = edu_row[0] if edu_row else None

        if not education_id and education_level:
            cur.execute(
                "INSERT INTO educations (education_name, cart_value) VALUES (%s, %s)",
                (education_level, 1),
            )
            education_id = cur.lastrowid

        cur.execute("SELECT id FROM jobs WHERE job_name = %s", (position,))
        existing = cur.fetchone()

        if existing:
            cur.execute(
                """
                UPDATE jobs
                SET max_applicants = %s,
                    status = %s,
                    start_date = %s,
                    end_date = %s,
                    required_education_id = %s,
                    required_experience = %s,
                    required_age = %s,
                    description = %s
                WHERE id = %s
                """,
                (
                    max_allowed,
                    form_access or "Open",
                    opening_date,
                    deadline_date,
                    education_id,
                    exp_val,
                    age_val,
                    location,
                    existing[0],
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO jobs
                (job_name, description, required_education_id, required_age,
                 required_experience, max_applicants, start_date, end_date,
                 status, created_by)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    position,
                    location,
                    education_id,
                    age_val,
                    exp_val,
                    max_allowed,
                    opening_date,
                    deadline_date,
                    form_access or "Open",
                    None,
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
            UPDATE jobs
            SET max_applicants = %s
            WHERE job_name = %s
            """,
            (max_allowed, position),
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
