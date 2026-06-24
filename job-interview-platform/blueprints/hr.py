# blueprints/hr.py
from flask import (
    Blueprint,
    render_template,
    request,
    session,
    redirect,
    url_for,
    jsonify,
    current_app
)
from extensions import mysql, mail
from flask_mail import Message
import datetime
from .schedule import get_progress_data

hr_bp = Blueprint("hr", __name__)

@hr_bp.route("/hr")
def hr_dashboard():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))
        
    user_id = session["user_id"]
    cur = mysql.connection.cursor()
    
    # 1) Logged-in HR user info
    cur.execute(
        "SELECT email, username FROM users WHERE user_id = %s",
        (user_id,)
    )
    user = cur.fetchone()
    
    if not user:
        cur.close()
        return redirect(url_for("auth.login"))

    email, username = user

    # 2) Standard Application Metrics (Replaces old CART/ANN logic)
    cur.execute("SELECT COUNT(*) FROM applications")
    total_requests = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM applications WHERE screening_status = 'Pending'")
    pending_applications = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM applications WHERE screening_status IN ('Approved', 'Eligible')")
    approved_applications = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM applications WHERE screening_status IN ('Rejected', 'Denied', 'Not Qualified')")
    rejected_applicants = cur.fetchone()[0] or 0

    avg_interview_score = 0.0

    # 3) Job Position Distribution
    cur.execute("""
        SELECT j.job_name, COUNT(a.application_id) AS total_count
        FROM applications a
        JOIN jobs j ON j.job_id = a.job_id
        GROUP BY j.job_id, j.job_name
        ORDER BY total_count DESC, j.job_name ASC
    """)
    position_distribution = [
        {"position": row[0] or "Unassigned", "count": row[1] or 0}
        for row in cur.fetchall()
    ]

    # 4) Recent Applicants
    cur.execute("""
        SELECT 
            a.application_id, 
            u.username, 
            u.email, 
            j.job_name, 
            a.screening_status,
            a.applied_at
        FROM applications a
        JOIN applicants ap ON ap.applicant_id = a.applicant_id
        JOIN users u ON u.user_id = ap.user_id
        JOIN jobs j ON j.job_id = a.job_id
        ORDER BY a.application_id DESC
        LIMIT 15
    """)
    applicant_rows = cur.fetchall()
    
    recent_applicants = []
    for row in applicant_rows:
        applied_date = row[5]
        applied_date_val = applied_date if isinstance(applied_date, (datetime.date, datetime.datetime)) else None
        
        recent_applicants.append({
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "role": row[3],
            "status": row[4],
            "score": 0.0, # Placeholder, DB schema lacks scores
            "applied_date": applied_date_val,
        })

    # 5) Upcoming Interviews
    cur.execute("""
        SELECT u.username, j.job_name
        FROM applications a
        JOIN applicants ap ON ap.applicant_id = a.applicant_id
        JOIN users u ON u.user_id = ap.user_id
        JOIN jobs j ON j.job_id = a.job_id
        WHERE a.screening_status IN ('Eligible', 'Approved')
        LIMIT 3
    """)
    interview_rows = cur.fetchall()
    
    upcoming_interviews = []
    for row in interview_rows:
        upcoming_interviews.append({
            "candidate_name": row[0],
            "position": row[1],
            "day": "Today",
            "date_str_short": datetime.date.today().strftime("%b %d"),
            "start_time": "10:00 AM",
        })

    # 6) Machine Learning Metrics Stub
    # Note: Because the DB schema only has a single `screening_status` column,
    # the advanced matrix tracking for ANN/CART is zeroed out to prevent errors.
    cart_ann_matrix = {
        "eligible_qualified": 0, "eligible_not_qualified": 0,
        "not_eligible_qualified": 0, "not_eligible_not_qualified": 0,
    }
    cart_metrics = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.0, "error_rate": 0.0, "total": 0}
    ann_metrics = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.0, "error_rate": 0.0, "total": 0}

    # 7) Recruitment Progress
    try:
        progress_data = get_progress_data()
    except Exception:
        progress_data = []

    # 8) Job Posts List
    cur.execute("""
        SELECT 
            j.job_name, j.max_applicants, j.application_status, 
            j.opening_date, j.application_deadline,
            jd.education_baseline, NULL AS target_school,
            jd.required_exp_years, jd.minimum_age, jd.location,
            jd.employment_type
        FROM jobs j
        LEFT JOIN job_desc jd ON jd.job_id = j.job_id
        ORDER BY j.job_name ASC
    """)
    job_posts = [
        {
            "position": row[0],
            "max_allowed": row[1],
            "form_access": row[2] or "Open",
            "opening_date": row[3],
            "deadline_date": row[4],
            "education_level": row[5],
            "target_school": row[6],
            "experience_years": row[7],
            "min_age": row[8],
            "location": row[9],
            "employment_type": row[10],
        }
        for row in cur.fetchall()
    ]
    cur.close()

    return render_template(
        "Hrpage.html",
        username=username or "HR Manager",
        email=email,
        total_requests=total_requests,
        pending_applications=pending_applications,
        approved_applications=approved_applications,
        rejected_applicants=rejected_applicants,
        avg_interview_score=avg_interview_score,
        position_distribution=position_distribution,
        jobs=job_posts,
        upcoming_interviews=upcoming_interviews,
        recent_applicants=recent_applicants,
        progress_data=progress_data,
        cart_ann_matrix=cart_ann_matrix,
        cart_metrics=cart_metrics,
        ann_metrics=ann_metrics,
    )

@hr_bp.route("/applicant-details/<int:app_id>")
def get_applicant_details(app_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT 
            u.username,
            u.email,
            j.job_name,
            a.screening_status,
            COALESCE(jd.required_exp_years, 0),
            GROUP_CONCAT(DISTINCT sk.skill_name ORDER BY sk.skill_name SEPARATOR ', ') AS skills
        FROM applications a
        JOIN applicants ap ON ap.applicant_id = a.applicant_id
        JOIN users u ON u.user_id = ap.user_id
        JOIN jobs j ON j.job_id = a.job_id
        LEFT JOIN job_desc jd ON jd.job_id = j.job_id
        LEFT JOIN applicant_skills askill ON askill.applicant_id = ap.applicant_id
        LEFT JOIN skills_master sk ON sk.skill_id = askill.skill_id
        WHERE a.application_id = %s
        GROUP BY u.username, u.email, j.job_name, a.screening_status, jd.required_exp_years
    """, (app_id,))
    
    row = cur.fetchone()
    cur.close()
    
    if not row:
        return jsonify({"error": "Not found"}), 404
        
    return jsonify({
        "name": row[0],
        "email": row[1],
        "role": row[2],
        "status": row[3],
        "experience": row[4],
        "skills": row[5] if row[5] else "None listed",
        "qa_data": None
    })

@hr_bp.route("/applicant-decision-json", methods=["POST"])
def applicant_decision_json():
    if "user_id" not in session:
        return jsonify({"ok": False, "msg": "Unauthorized"}), 401

    data = request.get_json() or {}
    app_id = data.get("applicant_id")
    decision = data.get("decision")
    
    if not app_id or decision not in ("approve", "reject"):
        return jsonify({"ok": False, "msg": "Invalid data"}), 400
        
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT u.username, u.email
            FROM applications a
            JOIN applicants ap ON ap.applicant_id = a.applicant_id
            JOIN users u ON u.user_id = ap.user_id
            WHERE a.application_id = %s
        """, (app_id,))
        
        row = cur.fetchone()
        
        if not row:
            cur.close()
            return jsonify({"ok": False, "msg": "Applicant not found"}), 404
            
        app_name, app_email = row
        new_status = "Approved" if decision == "approve" else "Rejected"
        
        cur.execute(
            "UPDATE applications SET screening_status = %s WHERE application_id = %s",
            (new_status, app_id)
        )
        mysql.connection.commit()
        cur.close()

        # Email Notification Logic
        if decision == "approve":
            subject = "AceView Application Update - Congratulations!"
            body = f"""Dear {app_name},
We are pleased to inform you that your application has been APPROVED for the next stage of our recruitment process.
Our HR team will contact you soon with the next steps and schedule details.

Best regards,
AceView Recruitment Team"""
        else:
            subject = "AceView Application Update"
            body = f"""Dear {app_name},
Thank you for taking the time to apply and interview with us.
After careful consideration, we regret to inform you that you have not been selected at this time.
We encourage you to apply again in the future for other opportunities.

Best regards,
AceView Recruitment Team"""

        try:
            default_sender = current_app.config.get("MAIL_DEFAULT_SENDER") or current_app.config.get("MAIL_USERNAME")
            msg = Message(subject=subject, recipients=[app_email], sender=default_sender)
            msg.body = body
            mail.send(msg)
        except Exception as mail_err:
            print("DECISION EMAIL ERROR:", mail_err)
            
    except Exception as e:
        print("DECISION GENERAL ERROR:", e)
        return jsonify({"ok": False, "msg": "Server error"}), 500
        
    return jsonify({"ok": True})

@hr_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))