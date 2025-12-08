# blueprints/hr.py
from flask import (
    Blueprint,
    render_template,
    request,
    session,
    redirect,
    url_for,
    jsonify,
    current_app,
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
        "SELECT email, username, profile_photo FROM users WHERE id = %s",
        (user_id,),
    )
    user = cur.fetchone()
    if not user:
        cur.close()
        return redirect(url_for("auth.login"))

    email, username, profile_photo = user

    # 2) BASIC COUNTS (for cards)
    # Total "applicants" in applicants table
    cur.execute("SELECT COUNT(*) FROM applicants")
    total_requests = cur.fetchone()[0] or 0

    # CART Eligible / Not Eligible
    cur.execute("SELECT COUNT(*) FROM applicants WHERE eligibility = 'Eligible'")
    cart_eligible = cur.fetchone()[0] or 0

    cur.execute(
        "SELECT COUNT(*) FROM applicants WHERE eligibility IS NULL OR eligibility <> 'Eligible'"
    )
    cart_not_eligible = cur.fetchone()[0] or 0

    # ANN Qualified / Not Qualified
    cur.execute(
        "SELECT COUNT(*) FROM chatbot WHERE qualification_status = 'Qualified'"
    )
    ann_qualified = cur.fetchone()[0] or 0

    cur.execute(
        "SELECT COUNT(*) FROM chatbot WHERE qualification_status = 'Not Qualified'"
    )
    ann_not_qualified = cur.fetchone()[0] or 0

    # 3) RECENT APPLICANTS TABLE (Dashboard)
    cur.execute(
        """
        SELECT 
            a.id,
            a.name,
            a.email,
            a.position,
            a.eligibility,
            IFNULL(c.qualification_status, 'Pending') AS ann_status,
            IFNULL(c.average_score, 0) AS avg_score,
            IFNULL(a.status, 'Pending') AS hr_status,
            a.start_date
        FROM applicants a
        LEFT JOIN chatbot c ON a.user_id = c.user_id
        ORDER BY a.id DESC
        LIMIT 15
        """
    )
    applicant_rows = cur.fetchall()

    recent_applicants = []
    for row in applicant_rows:
        # start_date might be None or datetime/date
        applied_date = row[8]
        if isinstance(applied_date, (datetime.date, datetime.datetime)):
            applied_date_val = applied_date
        else:
            applied_date_val = None

        recent_applicants.append(
            {
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "role": row[3],
                "cart_status": row[4],
                "ann_status": row[5],
                "score": float(row[6]) if row[6] is not None else 0.0,
                "hr_status": row[7],
                "applied_date": applied_date_val,
            }
        )

    # 4) UPCOMING INTERVIEWS (simple example)
    cur.execute(
        """
        SELECT a.name, a.position
        FROM chatbot c
        JOIN applicants a ON c.user_id = a.user_id
        WHERE c.qualification_status = 'Qualified'
        LIMIT 3
        """
    )
    interview_rows = cur.fetchall()
    upcoming_interviews = []
    for row in interview_rows:
        upcoming_interviews.append(
            {
                "candidate_name": row[0],
                "position": row[1],
                "day": "Today",
                "date_str_short": datetime.date.today().strftime("%b %d"),
                "start_time": "10:00 AM",
            }
        )

    # 5) CART–ANN RELATIONSHIP COUNTS (for tables + metrics)
    cur.execute(
        """
        SELECT a.eligibility, c.qualification_status, COUNT(*)
        FROM applicants a
        JOIN chatbot c ON a.user_id = c.user_id
        GROUP BY a.eligibility, c.qualification_status
        """
    )
    rel_rows = cur.fetchall()

    cart_ann_matrix = {
        "eligible_qualified": 0,
        "eligible_not_qualified": 0,
        "not_eligible_qualified": 0,
        "not_eligible_not_qualified": 0,
    }

    for eligibility, qual_status, count in rel_rows:
        eligibility = eligibility or "Not Eligible"
        qual_status = qual_status or "Not Qualified"

        if eligibility == "Eligible" and qual_status == "Qualified":
            cart_ann_matrix["eligible_qualified"] += count
        elif eligibility == "Eligible" and qual_status == "Not Qualified":
            cart_ann_matrix["eligible_not_qualified"] += count
        elif eligibility != "Eligible" and qual_status == "Qualified":
            cart_ann_matrix["not_eligible_qualified"] += count
        else:
            cart_ann_matrix["not_eligible_not_qualified"] += count

    # 6) PERFORMANCE METRICS FOR CART (treat ANN result as "ground truth")
    tp_c = cart_ann_matrix["eligible_qualified"]
    fp_c = cart_ann_matrix["eligible_not_qualified"]
    fn_c = cart_ann_matrix["not_eligible_qualified"]
    tn_c = cart_ann_matrix["not_eligible_not_qualified"]

    total_c = tp_c + fp_c + fn_c + tn_c
    if total_c > 0:
        acc_c = (tp_c + tn_c) / total_c
        err_c = 1.0 - acc_c
    else:
        acc_c = 0.0
        err_c = 0.0

    cart_metrics = {
        "tp": tp_c,
        "fp": fp_c,
        "fn": fn_c,
        "tn": tn_c,
        "accuracy": acc_c,
        "error_rate": err_c,
        "total": total_c,
    }

    # 7) PERFORMANCE METRICS FOR ANN (vs HR decision Approved/Rejected)
    cur.execute(
        """
        SELECT c.qualification_status, a.status, COUNT(*)
        FROM chatbot c
        JOIN applicants a ON a.user_id = c.user_id
        WHERE c.qualification_status IN ('Qualified','Not Qualified')
          AND a.status IN ('Approved','Rejected')
        GROUP BY c.qualification_status, a.status
        """
    )
    ann_rel_rows = cur.fetchall()

    tp_a = fp_a = fn_a = tn_a = 0
    for qual_status, hr_status, count in ann_rel_rows:
        if qual_status == "Qualified" and hr_status == "Approved":
            tp_a += count
        elif qual_status == "Qualified" and hr_status == "Rejected":
            fp_a += count
        elif qual_status == "Not Qualified" and hr_status == "Approved":
            fn_a += count
        elif qual_status == "Not Qualified" and hr_status == "Rejected":
            tn_a += count

    total_a = tp_a + fp_a + fn_a + tn_a
    if total_a > 0:
        acc_a = (tp_a + tn_a) / total_a
        err_a = 1.0 - acc_a
    else:
        acc_a = 0.0
        err_a = 0.0

    ann_metrics = {
        "tp": tp_a,
        "fp": fp_a,
        "fn": fn_a,
        "tn": tn_a,
        "accuracy": acc_a,
        "error_rate": err_a,
        "total": total_a,
    }

    cur.close()

    # 8) RECRUITMENT PROGRESS
    progress_data = get_progress_data()

    return render_template(
        "Hrpage.html",
        username=username or "HR Manager",
        email=email,
        total_requests=total_requests,
        cart_eligible=cart_eligible,
        cart_not_eligible=cart_not_eligible,
        ann_qualified=ann_qualified,
        ann_not_qualified=ann_not_qualified,
        jobs=[],  # (you can keep or remove jobs if you use it elsewhere)
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
    cur.execute(
        """
        SELECT 
            a.name,
            a.email,
            a.position,
            a.eligibility,
            a.yearexperience,
            a.skills,
            c.qualification_status,
            c.average_score,
            c.assessment_data
        FROM applicants a
        LEFT JOIN chatbot c ON a.user_id = c.user_id
        WHERE a.id = %s
        """,
        (app_id,),
    )
    row = cur.fetchone()
    cur.close()

    if not row:
        return jsonify({"error": "Not found"}), 404

    return jsonify(
        {
            "name": row[0],
            "email": row[1],
            "role": row[2],
            "cart_status": row[3],
            "experience": row[4],
            "skills": row[5],
            "ann_status": row[6] if row[6] else "Pending",
            "ann_score": float(row[7]) if row[7] else 0,
            "qa_data": row[8],
        }
    )


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
        cur.execute(
            "SELECT name, email FROM applicants WHERE id = %s", (app_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"ok": False, "msg": "Applicant not found"}), 404

        app_name, app_email = row
        new_status = "Approved" if decision == "approve" else "Rejected"

        cur.execute(
            "UPDATE applicants SET status = %s WHERE id = %s",
            (new_status, app_id),
        )
        mysql.connection.commit()
        cur.close()

        # email body
        if decision == "approve":
            subject = "AceView Application Update - Congratulations!"
            body = f"""Dear {app_name},

We are pleased to inform you that your application has been APPROVED for the next stage of our recruitment process.

Our HR team will contact you soon with the next steps and schedule details.

Best regards,
AceView Recruitment Team
"""
        else:
            subject = "AceView Application Update"
            body = f"""Dear {app_name},

Thank you for taking the time to apply and interview with us.

After careful consideration, we regret to inform you that you have not been selected at this time.
We encourage you to apply again in the future for other opportunities.

Best regards,
AceView Recruitment Team
"""

        try:
            default_sender = current_app.config.get("MAIL_DEFAULT_SENDER") or current_app.config.get(
                "MAIL_USERNAME"
            )
            msg = Message(subject=subject, recipients=[
                          app_email], sender=default_sender)
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
