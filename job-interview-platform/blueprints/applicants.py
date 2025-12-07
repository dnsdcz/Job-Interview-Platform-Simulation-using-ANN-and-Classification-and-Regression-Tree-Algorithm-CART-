# blueprints/applicants.py

from datetime import datetime
from urllib.parse import unquote

import numpy as np

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
    current_app,
)
from werkzeug.utils import secure_filename

from extensions import mysql, logger
from services.email_service import send_step1_completed_email  # ⬅️ NEW IMPORT

# CART (Decision Tree)
from sklearn.tree import DecisionTreeClassifier

applicants_bp = Blueprint("applicants", __name__)


# ---------- CART (Decision Tree) MODEL FOR PRESCREEN & APPLICATION ----------

# Education mapping
CART_EDU_MAP = {
    "high_school": 1,
    "vocational": 2,
    "associate": 3,
    "bachelor": 4,
    "master": 5,
    "phd": 6,
}


def cart_normalize_age(age):
    age = max(20, min(age, 65))  # clamp between 20–65
    return (age - 20) / (65 - 20)


def cart_experience_score(exp_years):
    return min(exp_years, 15) / 15.0


def cart_skill_score(skill_count):
    return min(skill_count, 10) / 10.0


# Sample training data (you can replace with your real dataset later)
raw_applicants_cart = [
    ("bachelor",    25,  3,  4, 1),
    ("high_school", 21,  0,  1, 0),
    ("master",      32,  8,  7, 1),
    ("bachelor",    45, 15, 10, 1),
    ("vocational",  28,  2,  2, 0),
    ("associate",   23,  1,  3, 0),
    ("phd",         38, 10,  6, 1),
    ("bachelor",    50,  5,  5, 1),
    ("high_school", 60, 15,  2, 0),
    ("master",      27,  4,  8, 1),
]

X_cart = []
y_cart = []

for edu, age, exp, skills, label in raw_applicants_cart:
    edu_score = CART_EDU_MAP[edu]
    age_norm = cart_normalize_age(age)
    exp_s = cart_experience_score(exp)
    skill_s = cart_skill_score(skills)
    X_cart.append([edu_score, age_norm, exp_s, skill_s])
    y_cart.append(label)

X_cart = np.array(X_cart)
y_cart = np.array(y_cart)

# CART model trained once at app startup
cart_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42,
)
cart_model.fit(X_cart, y_cart)


def cart_predict_from_form(age, education_level, experience, skills_raw: str):
    """
    Use the global cart_model (DecisionTreeClassifier) to compute
    score and predicted status from form data.
    """
    # education
    edu_score = CART_EDU_MAP.get(education_level, 1)

    # normalized & derived scores
    age_norm = cart_normalize_age(age)
    exp_s = cart_experience_score(experience)

    skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]
    skill_count = len(skills_list)
    skill_s = cart_skill_score(skill_count)

    features = np.array([[edu_score, age_norm, exp_s, skill_s]])

    proba = cart_model.predict_proba(features)[0][1]  # prob of class 1
    pred = cart_model.predict(features)[0]

    status = "Eligible" if pred == 1 else "Not Eligible"

    return {
        "status": status,
        "model_score": float(proba),  # 0–1
        "probability_percent": round(float(proba) * 100, 2),
        "features": {
            "edu_score": edu_score,
            "age_norm": float(age_norm),
            "exp_score": float(exp_s),
            "skill_score": float(skill_s),
            "skill_count": skill_count,
        },
    }


# ---------- Dashboard & application ----------


@applicants_bp.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    cur.execute(
        "SELECT email, username, contact_number FROM users WHERE id = %s",
        (user_id,),
    )
    user = cur.fetchone()
    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))
    email, username, contact = user

    # check if user already has an application
    cur.execute("SELECT * FROM applicants WHERE email = %s", (email,))
    row = cur.fetchone()
    applicant = None
    if row:
        # applicants table layout:
        # 0 id, 1 user_id, 2 name, 3 email, 4 contact, 5 position,
        # 6 eligibility, 7 yearexperience, 8 Level, 9 status,
        # 10 confidence, 11 address, 12 start_date, 13 desired_pay,
        # 14 employment_type, 15 school, 16 school_location,
        # 17 years_attended, 18 education_level, 19 degree,
        # 20 major, 21 job_title, 22 company, 23 responsibilities, 24 skills
        applicant = {
            "id": row[0],
            "user_id": row[1],
            "name": row[2],
            "email": row[3],
            "contact": row[4],
            "position": row[5],
            "eligibility": row[6],
            "yearexperience": row[7],
            "level": row[8],
            "status": row[9],
            "confidence": row[10],
            "address": row[11],
            "start_date": row[12],
            "desired_pay": row[13],
            "employment_type": row[14],
            "school": row[15],
            "school_location": row[16],
            "years_attended": row[17],
            "education_level": row[18],
            "degree": row[19],
            "major": row[20],
            "job_title": row[21],
            "company": row[22],
            "responsibilities": row[23],
            "skills": row[24],
        }
        # ensure interview uses same values
        session["position"] = row[5]
        session["experience"] = row[7]
        session["name"] = row[2]

    # position limits info
    cur.execute(
        """
        SELECT pl.id, pl.position, pl.max_allowed,
               COUNT(a.position) AS current_count
        FROM position_limits pl
        LEFT JOIN applicants a ON pl.position = a.position
        GROUP BY pl.id, pl.position, pl.max_allowed
        """
    )
    positions = cur.fetchall()
    position_limits = [
        {
            "id": p[0],
            "position": p[1],
            "max_allowed": p[2],
            "current_count": p[3],
            "is_full": p[3] >= p[2],
        }
        for p in positions
    ]

    name = session.get("name", username)
    result = session.get("result")
    reason = session.get("reason")
    confidence = session.get("confidence")
    position = session.get("position")
    qualification_status = session.get(
        "qualification_status", "")  # from chatbot only
    applied_role = position or "Business Analyst"

    cur.close()

    return render_template(
        "dashboard.html",
        name=name,
        email=email,
        contact=contact,
        username=username,
        result=result,
        reason=reason,
        confidence=confidence,
        position=position,
        applied_role=applied_role,
        qualification_status=qualification_status,
        application_data=applicant,
        has_applied=applicant is not None,
        position_limits=position_limits,
    )


@applicants_bp.route("/submit", methods=["POST"])
def submit_application():
    if "user_id" not in session:
        flash("You must be logged in to apply.", "error")
        return redirect(url_for("auth.login"))

    try:
        # --- 1. GET DATA ---
        form = request.form
        user_id = session["user_id"]

        # Collect Form Data
        name = form.get("name")
        email = form.get("email")
        contact = form.get("contact")
        age = int(form.get("age")) if form.get("age") else 0
        address = form.get("address")

        position = form.get("position")
        start_date = form.get("start_date")
        desired_pay = int(form.get("desired_pay")) if form.get(
            "desired_pay") else 0
        employment_type = form.get("employment_type")

        school = form.get("school")
        school_location = form.get("school_location")
        years_attended = form.get("years_attended")
        education_level = form.get("education_level")
        degree = form.get("degree")
        major = form.get("major")

        job_title = form.get("job_title")
        company = form.get("company")
        experience = int(form.get("experience")) if form.get(
            "experience") else 0
        responsibilities = form.get("responsibilities")
        skills = form.get("skills", "")

        cur = mysql.connection.cursor()

        # --- 2. CHECK DUPLICATES ---
        cur.execute("SELECT id FROM applicants WHERE email = %s", (email,))
        if cur.fetchone():
            flash("This email has already been used to apply.", "error")
            cur.close()
            return redirect(url_for("applicants.dashboard"))

        # --- 3. FETCH HR REQUIREMENTS ---
        cur.execute(
            """
            SELECT max_allowed, form_access, opening_date, deadline_date, 
                   min_age, experience_years
            FROM position_limits 
            WHERE position = %s
            """,
            (position,),
        )

        limits = cur.fetchone()

        # Default values if not set in HR dashboard
        req_age = 18
        req_exp = 0

        if limits:
            max_allowed, form_access, open_date, deadline, db_min_age, db_min_exp = limits
            if db_min_age:
                req_age = db_min_age
            if db_min_exp:
                req_exp = db_min_exp

            # Check Basic Limits (Closed/Full)
            if form_access == "Closed":
                flash("Applications are closed.", "error")
                cur.close()
                return redirect(url_for("applicants.dashboard"))

            cur.execute(
                "SELECT COUNT(*) FROM applicants WHERE position = %s", (position,)
            )
            if cur.fetchone()[0] >= max_allowed:
                flash("Position is full.", "error")
                cur.close()
                return redirect(url_for("applicants.dashboard"))

        # --- 4. CALCULATE SPECIFIC REASON FOR REJECTION ---
        rejection_reasons = []

        # Rule 1: Age
        if age < req_age:
            rejection_reasons.append(
                f"Age ({age}) is below the minimum requirement of {req_age}"
            )

        # Rule 2: Experience
        if experience < req_exp:
            rejection_reasons.append(
                f"Experience ({experience} yrs) is below the required {req_exp} yrs"
            )

        # Rule 3: Skills Count (Basic check)
        skills_list = [s.strip() for s in skills.split(",") if s.strip()]
        if len(skills_list) < 2:
            rejection_reasons.append(
                "Insufficient skills listed (minimum 2 required)"
            )

        # Rule 4: AI Score (Soft Check) - USING DECISION TREE (CART)
        try:
            cart_result = cart_predict_from_form(
                age=age,
                education_level=education_level,
                experience=experience,
                skills_raw=skills,
            )
            model_score = cart_result["model_score"]  # 0–1
            confidence = cart_result["probability_percent"]  # 0–100
            session["cart_details"] = cart_result
        except Exception as e:
            logger.error(f"CART prediction error: {e}")
            model_score = 0.5
            confidence = model_score * 100

        # If they passed hard rules, check AI score
        if not rejection_reasons:
            if model_score < 0.55:
                rejection_reasons.append(
                    "Assessment score below qualification threshold"
                )

        # --- 5. DETERMINE FINAL STATUS (ELIGIBLE / NOT ELIGIBLE ONLY) ---
        if not rejection_reasons:
            eligibility = "Eligible"
            final_reason = "You meet all requirements for this position."
        else:
            eligibility = "Not Eligible"
            final_reason = "Not Eligible: " + "; ".join(rejection_reasons)

        # We'll use the numeric "status" field for application state
        # e.g. 0 = Pending, 1 = Approved, 2 = Denied
        status = 0  # Pending by default

        # Level is required (NOT NULL) in your table, so give it something
        level_value = "N/A"

        # confidence in DB is int, but our model gives float 0–100
        confidence_int = int(round(confidence))

        # --- 6. INSERT INTO DATABASE (NO QUALIFIED COLUMN) ---
        query = """
            INSERT INTO applicants
            (user_id, name, email, contact, position,
             eligibility, yearexperience, Level, status, confidence,
             address, start_date, desired_pay, employment_type,
             school, school_location, years_attended, education_level,
             degree, major, job_title, company, responsibilities, skills)
            VALUES
            (%s, %s, %s, %s, %s,
             %s, %s, %s, %s, %s,
             %s, %s, %s, %s,
             %s, %s, %s, %s,
             %s, %s, %s, %s, %s, %s)
        """

        values = (
            user_id,
            name,
            email,
            contact,
            position,
            eligibility,
            experience,          # maps to yearexperience column
            level_value,         # Level (required)
            status,              # numeric status
            confidence_int,      # confidence as INT
            address,
            start_date,
            desired_pay,
            employment_type,
            school,
            school_location,
            years_attended,
            education_level,
            degree,
            major,
            job_title,
            company,
            responsibilities,
            skills,
        )

        cur.execute(query, values)
        mysql.connection.commit()
        cur.close()

        # --- 7. UPDATE SESSION WITH THE RESULT ---
        session["name"] = name
        session["position"] = position
        session["result"] = eligibility
        session["confidence"] = confidence_int
        session["reason"] = final_reason

        # Flash + email when Eligible (Step 1 complete)
        if eligibility == "Eligible":
            flash(f"Application Submitted! {final_reason}", "success")
            try:
                send_step1_completed_email(email, name, position)
            except Exception as e:
                logger.error(f"Error sending Step 1 email after submit: {e}")
        else:
            flash(f"Application Submitted. Status: {final_reason}", "error")

        return redirect(url_for("applicants.dashboard"))

    except Exception as e:
        logger.error(f"submit_application error: {e}")
        flash(f"Error: {e}", "error")
        return redirect(url_for("applicants.dashboard"))


# ---------- Applicant views ----------


@applicants_bp.route("/applicant")
def applicant_view():
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT email, username, contact_number FROM users WHERE id = %s",
        (user_id,),
    )
    user = cur.fetchone()
    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))

    email, username, contact = user
    cur.execute(
        "SELECT eligibility, position FROM applicants WHERE user_id = %s",
        (user_id,),
    )
    eligibility_row = cur.fetchone()
    cur.close()

    name = session.get("name")
    eligible_applicant = session.get("result")
    position = session.get("position")
    reason = session.get("reason")
    confidence = session.get("confidence")

    return render_template(
        "applicant.html",
        email=email,
        name=name,
        username=username,
        contact=contact,
        result=eligible_applicant,
        position=position,
        reason=reason,
        confidence=confidence,
    )


@applicants_bp.route("/viewapp")
def view_applicants():
    if "user_id" not in session:
        flash("You must be logged in to view applicants.", "error")
        return redirect(url_for("auth.login"))

    cur = mysql.connection.cursor()
    cur.execute("SELECT name, email, contact, position FROM applicants")
    applicants = cur.fetchall()
    cur.close()

    return render_template("view_applicants.html", applicants=applicants)


@applicants_bp.route("/viewchat")
def view_chatbot():
    if "user_id" not in session:
        flash("You must be logged in to view chatbot data.", "error")
        return redirect(url_for("auth.login"))

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT user_name, position, experience, qualification_status FROM chatbot"
    )
    chatbot = cur.fetchall()
    cur.close()

    return render_template("view_chatbot.html", chatbot=chatbot)


# ---------- Pre-application simple form (CART PRESCREEN) ----------


@applicants_bp.route("/prescreenn", methods=["GET", "POST"])
def prescreen():
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT email, username, contact_number FROM users WHERE id = %s",
        (user_id,),
    )
    user = cur.fetchone()
    cur.close()

    if not user:
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))

    email, username, contact = user

    result = None

    if request.method == "POST":
        education_level = request.form.get("education_level")
        age = int(request.form.get("age") or 0)
        experience = int(request.form.get("experience") or 0)
        skills_raw = request.form.get("skills", "")

        try:
            cart_result = cart_predict_from_form(
                age=age,
                education_level=education_level,
                experience=experience,
                skills_raw=skills_raw,
            )
            result = cart_result
            session["prescreen_result"] = result
            flash(
                f"Prescreen result: {result['status']} (confidence {result['probability_percent']}%)",
                "info",
            )
        except Exception as e:
            logger.error(f"CART prescreen error: {e}")
            flash("Error during prescreening.", "error")
            result = None
    else:
        result = session.get("prescreen_result")

    return render_template(
        "prescreen.html",
        email=email,
        username=username,
        contact=contact,
        result=result,
    )


@applicants_bp.route("/preapp", methods=["GET", "POST"])
def preapp():
    """
    Very simple pre-application: creates a minimal applicants row
    with eligibility Pending and NO qualified field.
    """
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    if request.method == "POST":
        position = request.form.get("position")
        yearexperience = request.form.get("yearexperience")

        cur.execute(
            "SELECT username, email, contact_number FROM users WHERE id = %s",
            (user_id,),
        )
        user_info = cur.fetchone()
        if not user_info:
            flash("User not found.", "error")
            cur.close()
            return redirect(url_for("auth.login"))

        name, email, contact = user_info

        eligibility = "Pending"
        level_value = "N/A"
        status = 0        # Pending
        confidence = 0    # no AI score yet

        cur.execute(
            """
            INSERT INTO applicants
            (user_id, name, email, contact, position,
             yearexperience, eligibility, Level, status, confidence)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                user_id,
                name,
                email,
                contact,
                position,
                yearexperience,
                eligibility,
                level_value,
                status,
                confidence,
            ),
        )
        mysql.connection.commit()
        cur.close()
        return redirect(url_for("applicants.preapp"))

    cur.execute(
        """
        SELECT name, email, contact, position, yearexperience, eligibility,
               Level, status, confidence
        FROM applicants
        WHERE user_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    )
    applicant = cur.fetchone()
    cur.close()

    if applicant:
        (
            name,
            email,
            contact,
            position,
            yearexperience,
            eligibility,
            level_value,
            status,
            confidence,
        ) = applicant
        app_needed = False
    else:
        name = email = contact = position = yearexperience = eligibility = level_value = status = confidence = None
        app_needed = True

    return render_template(
        "pre-app.html",
        name=name,
        email=email,
        contact=contact,
        position=position,
        yearexperience=yearexperience,
        eligibility=eligibility,
        level=level_value,
        status=status,
        confidence=confidence,
        app_needed=app_needed,
    )


# ---------- Profile & photo ----------


def _allowed_profile_file(filename: str) -> bool:
    allowed = current_app.config.get(
        "ALLOWED_PROFILE_EXTENSIONS", {"png", "jpg", "jpeg", "gif"}
    )
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@applicants_bp.route("/upload_photo", methods=["POST"])
def upload_photo():
    if "user_id" not in session:
        flash("You must be logged in to upload a profile photo.", "error")
        return redirect(url_for("auth.login"))

    if "profile_photo" not in request.files:
        flash("No file part", "error")
        return redirect(url_for("applicants.profile"))

    file = request.files["profile_photo"]
    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("applicants.profile"))

    if file and _allowed_profile_file(file.filename):
        image_data = file.read()
        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE users SET profile_photo = %s WHERE id = %s",
            (image_data, session["user_id"]),
        )
        mysql.connection.commit()
        cur.close()
        flash("Profile photo saved successfully", "success")
        return redirect(url_for("applicants.profile"))

    flash("Invalid file type", "error")
    return redirect(url_for("applicants.profile"))


@applicants_bp.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    cur.execute(
        "SELECT email, username, contact_number, usertype, profile_photo FROM users WHERE id = %s",
        (user_id,),
    )
    user = cur.fetchone()

    cur.execute("SELECT * FROM applicants WHERE email = %s", (user[0],))
    applicant = cur.fetchone()

    cur.execute("SELECT * FROM applicants")
    applications = cur.fetchall()
    cur.close()

    return render_template(
        "profile.html",
        email=user[0],
        username=user[1],
        contact=user[2],
        profile_photo=user[4],
        position=applicant[5] if applicant else None,
        eligibility=applicant[6] if applicant else None,
        yearexperience=applicant[7] if applicant else None,
        # no qualified field in applicants table anymore
        qualified=None,
        applications=applications,
    )


# ---------- NEW: HR view per job + applicant approve/deny ----------


@applicants_bp.route("/job/<path:position>")
def job_applicants(position):
    """
    When called with ?modal=1 returns JSON list of applicants for that position.
    Can still render full page if you visit /job/<position> directly in browser.

    NOTE:
    - Here we may still use chatbot.qualification_status to show "Qualified" only
      from chatbot (not from applicants table).
    """
    if "user_id" not in session:
        if request.args.get("modal") == "1":
            return jsonify({"error": "not_logged_in"}), 401
        flash("You must be logged in to view applicants.", "error")
        return redirect(url_for("auth.login"))

    # only HR users should see this
    user_id = session["user_id"]
    cur = mysql.connection.cursor()
    cur.execute("SELECT usertype FROM users WHERE id = %s", (user_id,))
    row = cur.fetchone()
    if not row or row[0] != "hrpage":
        cur.close()
        if request.args.get("modal") == "1":
            return jsonify({"error": "not_authorized"}), 403
        flash("You are not authorized to view this page.", "error")
        return redirect(url_for("applicants.dashboard"))

    position = unquote(position)

    # Join with chatbot to read qualification_status from chatbot table ONLY
    cur.execute(
        """
        SELECT 
            a.id,
            a.name,
            a.email,
            a.contact,
            a.yearexperience,
            a.education_level,
            a.`Level`,
            c.qualification_status,
            a.confidence,
            a.eligibility
        FROM applicants a
        LEFT JOIN chatbot c ON c.user_id = a.user_id
        WHERE a.position = %s
        ORDER BY a.confidence DESC, a.name ASC
        """,
        (position,),
    )
    rows = cur.fetchall()
    cur.close()

    applicants = []
    for r in rows:
        applicants.append(
            {
                "id": r[0],
                "name": r[1],
                "email": r[2],
                "contact": r[3],
                "experience": r[4],
                "education_level": r[5],
                "level": r[6],
                # qualified here is from chatbot, not applicants table
                "qualified": r[7] or "Pending",
                "confidence": r[8],
                "eligibility": r[9] or "",
            }
        )

    # If modal=1 → JSON for AJAX
    if request.args.get("modal") == "1":
        return jsonify({"position": position, "applicants": applicants})

    # Optional: still support full-page view if you ever visit this URL directly
    return render_template(
        "job_applicants.html",
        position=position,
        applicants=applicants,
    )


@applicants_bp.route("/applicant-decision-json", methods=["POST"])
def applicant_decision_json():
    """
    Approve / Deny from modal via fetch().

    - 'eligibility' keeps the text (Eligible / Not Eligible)
    - 'status' is the pipeline state:
        0 = Pending      (after application only)
        1 = Approved     (final, after all steps)
        2 = Denied       (final, after all steps)

    When HR clicks approve (Eligible), we also send an email to the applicant
    telling them they have finished Step 1.
    """
    if "user_id" not in session:
        return jsonify({"error": "not_logged_in"}), 401

    data = request.get_json() or {}
    applicant_id = data.get("applicant_id")
    decision = data.get("decision")
    position = data.get("position")

    if not applicant_id or decision not in ("approve", "deny") or not position:
        return jsonify({"error": "invalid_data"}), 400

    # eligibility text
    new_eligibility = "Eligible" if decision == "approve" else "Not Eligible"
    # numeric pipeline status + human label
    new_status_code = 1 if decision == "approve" else 2   # 1=Approved, 2=Denied
    new_status_label = "Approved" if decision == "approve" else "Denied"

    try:
        cur = mysql.connection.cursor()

        # Get applicant info first (for email)
        cur.execute(
            "SELECT name, email, position FROM applicants WHERE id = %s",
            (applicant_id,),
        )
        app_row = cur.fetchone()

        # Update eligibility + numeric status (final decision)
        cur.execute(
            "UPDATE applicants SET eligibility = %s, status = %s WHERE id = %s",
            (new_eligibility, new_status_code, applicant_id),
        )
        mysql.connection.commit()
        cur.close()

        # If approved → send "Step 1 finished" email via service
        if decision == "approve" and app_row:
            name, email, db_position = app_row
            pos_for_email = db_position or position
            try:
                send_step1_completed_email(email, name, pos_for_email)
            except Exception as e:
                logger.error(
                    f"Error sending Step 1 email after HR approve: {e}")

        return jsonify({
            "ok": True,
            "eligibility": new_eligibility,      # Eligible / Not Eligible
            "status_label": new_status_label,    # Approved / Denied
            "status_code": new_status_code       # 1 / 2
        })

    except Exception as e:
        logger.error(f"applicant_decision_json error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------- Misc helpers ----------


@applicants_bp.route("/save_experience", methods=["POST"])
def save_experience():
    user_id = session.get("user_id")
    yearexperience = request.form.get("yearexperience")

    if user_id and yearexperience:
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                """
                UPDATE applicants
                SET yearexperience = %s
                WHERE user_id = %s
                """,
                (yearexperience, user_id),
            )
            mysql.connection.commit()
            cur.close()
            session["experience"] = int(yearexperience)
            return jsonify({"success": "Experience saved successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid data"}), 400


@applicants_bp.route("/progress")
def show_progress():
    """
    Progress view per position based on chatbot qualification only.
    applicants table is not used for 'qualified' anymore.
    """
    cur = mysql.connection.cursor()
    cur.execute(
        """
        SELECT 
            a.position,
            COUNT(
                CASE 
                    WHEN LOWER(c.qualification_status) = 'qualified'
                    THEN 1 ELSE NULL
                END
            ) AS percentage
        FROM applicants a
        LEFT JOIN chatbot c ON c.user_id = a.user_id
        GROUP BY a.position
        """
    )
    progress_data = cur.fetchall()
    cur.close()
    return render_template("progress.html", progress_data=progress_data)


@applicants_bp.route("/check_email")
def check_email():
    email = request.args.get("email")
    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM applicants WHERE email = %s", (email,))
    existing_user = cur.fetchone()
    cur.close()
    return jsonify({"exists": bool(existing_user)})


@applicants_bp.route("/applicants")
def view_applications():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM applicants")
        applications = cur.fetchall()
        cur.close()
        return render_template("applications.html", applications=applications)
    except Exception as e:
        return f"Error: {e}", 500


@applicants_bp.route("/get_applicants")
def get_applicants():
    cur = mysql.connection.cursor()
    cur.execute("SELECT name, position, yearexperience FROM applicants")
    rows = cur.fetchall()
    cur.close()

    applicants = [
        {"name": r[0], "position": r[1], "experience": r[2]} for r in rows
    ]
    return jsonify(applicants)
