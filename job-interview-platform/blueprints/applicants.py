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


# Sample training data
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
    # education mapping normalized fallback
    edu_clean = education_level.lower().replace("'", "").replace(" ", "_") if education_level else "high_school"
    edu_score = CART_EDU_MAP.get(edu_clean, 1)

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


# ---------- Dashboard & Application Pipeline ----------


@applicants_bp.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    # Get baseline profile credentials from 'users' (normalized column name: user_type)
    cur.execute(
        "SELECT email, username, contact_num FROM users WHERE user_id = %s",
        (user_id,),
    )
    user = cur.fetchone()
    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))
    email, username, contact = user

    # Retrieve the candidate's latest normalized application data
    cur.execute(
        """
        SELECT
            app.application_id,
            u.username,
            u.email,
            u.contact_num,
            j.job_name,
            app.screening_status,
            COALESCE(
                (SELECT SUM(TIMESTAMPDIFF(YEAR, start_date, COALESCE(end_date, CURDATE()))) 
                 FROM work_experience 
                 WHERE applicant_id = a.applicant_id), 0
            ) AS years_experience,
            (SELECT degree_level FROM educations WHERE applicant_id = a.applicant_id ORDER BY graduation_year DESC LIMIT 1) AS education_level,
            (SELECT GROUP_CONCAT(sm.skill_name SEPARATOR ', ') 
             FROM applicant_skills ask 
             JOIN skills_master sm ON sm.skill_id = ask.skill_id 
             WHERE ask.applicant_id = a.applicant_id) AS skills,
            TIMESTAMPDIFF(YEAR, a.date_of_birth, CURDATE()) AS age
        FROM applications app
        JOIN applicants a ON a.applicant_id = app.applicant_id
        JOIN users u ON u.user_id = a.user_id
        JOIN jobs j ON j.job_id = app.job_id
        WHERE u.user_id = %s
        ORDER BY app.application_id DESC
        LIMIT 1
        """,
        (user_id,),
    )
    row = cur.fetchone()
    applicant = None
    if row:
        applicant = {
            "id": row[0],
            "user_id": user_id,
            "name": row[1],
            "email": row[2],
            "contact": row[3],
            "position": row[4],
            "eligibility": "Eligible" if row[5] == "Passed Screening" else "Not Eligible",
            "yearexperience": row[6],
            "level": "N/A",
            "status": row[5],  # screening_status string
            "confidence": 75,  # static fallback when not using live prediction state
            "address": None,
            "education_level": row[7] or "N/A",
            "skills": row[8] or "None listed",
            "age": row[9]
        }
        # Sync simple session data for backward compatibility
        session["position"] = row[4]
        session["experience"] = row[6]
        session["name"] = row[1]

    # Fetch available job list and real-time candidate limit statistics
    cur.execute(
        """
        SELECT j.job_id, j.job_name, j.max_applicants, COUNT(app.application_id) AS current_count
        FROM jobs j
        LEFT JOIN applications app ON app.job_id = j.job_id
        GROUP BY j.job_id, j.job_name, j.max_applicants
        ORDER BY j.job_name ASC
        """
    )
    positions = cur.fetchall()
    position_limits = [
        {
            "id": p[0],
            "position": p[1],
            "max_allowed": p[2],
            "current_count": p[3],
            "is_full": bool(p[2] and p[3] >= p[2]),
        }
        for p in positions
    ]

    # Pull dynamic jobs and detailed description requirements
    cur.execute(
        """
        SELECT j.job_name, j.application_status, j.opening_date, j.application_deadline,
               jd.education_baseline, jd.required_exp_years, jd.minimum_age, jd.employment_type
        FROM jobs j
        LEFT JOIN job_desc jd ON jd.job_id = j.job_id
        ORDER BY j.opening_date DESC, j.job_name ASC
        """
    )
    available_jobs = [
        {
            "position": p[0],
            "max_allowed": None,
            "form_access": p[1],
            "opening_date": p[2],
            "deadline_date": p[3],
            "education_level": p[4],
            "experience_years": p[5],
            "min_age": p[6],
            "employment_type": p[7],
        }
        for p in cur.fetchall()
    ]

    # Fetch assessment scores dynamically
    chatbot_data = None
    name = session.get("name", username)
    result = session.get("result")
    reason = session.get("reason")
    confidence = session.get("confidence")
    position = session.get("position")
    qualification_status = session.get("qualification_status", "")
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
        available_jobs=available_jobs,
        chatbot_data=chatbot_data,
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

        name = form.get("name")
        email = form.get("email")
        contact = form.get("contact")
        age = int(form.get("age")) if form.get("age") else 0
        address = form.get("address")

        position = form.get("position")
        start_date_form = form.get("start_date")
        desired_pay = int(form.get("desired_pay")) if form.get("desired_pay") else 0
        employment_type = form.get("employment_type")

        school = form.get("school")
        school_location = form.get("school_location")
        years_attended = form.get("years_attended")
        education_level = form.get("education_level")
        degree = form.get("degree")
        major = form.get("major")

        job_title = form.get("job_title")
        company = form.get("company")
        experience = int(form.get("experience")) if form.get("experience") else 0
        responsibilities = form.get("responsibilities")
        skills = form.get("skills", "")

        cur = mysql.connection.cursor()

        # --- 2. RETRIEVE OR GENERATE ATOMIC APPLICANT RECORD ---
        cur.execute("SELECT applicant_id FROM applicants WHERE user_id = %s", (user_id,))
        app_row = cur.fetchone()
        
        dob_calc = f"{datetime.now().year - age}-01-01"
        if app_row:
            applicant_id = app_row[0]
            # Update basic demography
            cur.execute(
                """
                UPDATE applicants 
                SET full_name = %s, date_of_birth = %s, current_location = %s 
                WHERE applicant_id = %s
                """,
                (name, dob_calc, address, applicant_id)
            )
        else:
            cur.execute(
                """
                INSERT INTO applicants (user_id, full_name, date_of_birth, current_location, preferred_location, resume_url)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, name, dob_calc, address, address, "s3://resumes/placeholder_applicant.pdf")
            )
            applicant_id = cur.lastrowid

        # --- 3. FETCH NORMALIZED JOB SPECIFICATIONS ---
        cur.execute(
            """
            SELECT j.job_id, j.max_applicants, j.application_status, 
                   jd.minimum_age, jd.required_exp_years
            FROM jobs j
            LEFT JOIN job_desc jd ON jd.job_id = j.job_id
            WHERE j.job_name = %s
            """,
            (position,),
        )
        job_info = cur.fetchone()
        if not job_info:
            flash("The selected position does not exist.", "error")
            cur.close()
            return redirect(url_for("applicants.dashboard"))

        job_id, max_allowed, app_status, req_age, req_exp = job_info

        # Check status & applicant limits
        if app_status != "Open":
            flash("Applications for this position are currently closed.", "error")
            cur.close()
            return redirect(url_for("applicants.dashboard"))

        cur.execute("SELECT COUNT(*) FROM applications WHERE job_id = %s", (job_id,))
        if cur.fetchone()[0] >= max_allowed:
            flash("This position has reached its maximum applicant limit.", "error")
            cur.close()
            return redirect(url_for("applicants.dashboard"))

        # Duplicate Application check
        cur.execute("SELECT application_id FROM applications WHERE job_id = %s AND applicant_id = %s", (job_id, applicant_id))
        if cur.fetchone():
            flash("You have already applied for this position.", "error")
            cur.close()
            return redirect(url_for("applicants.dashboard"))

        # --- 4. INSERT EDUCATIONAL ENTITY ---
        if school or degree or major:
            cur.execute("DELETE FROM educations WHERE applicant_id = %s", (applicant_id,))
            cur.execute(
                """
                INSERT INTO educations (applicant_id, degree_level, major, institution, graduation_year)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (applicant_id, education_level or "High School", major or "General Education", school or "N/A", 2024)
            )

        # --- 5. INSERT WORK EXPERIENCE ENTITY ---
        if job_title or company or experience > 0:
            cur.execute("DELETE FROM work_experience WHERE applicant_id = %s", (applicant_id,))
            start_yr = datetime.now().year - experience
            cur.execute(
                """
                INSERT INTO work_experience (applicant_id, job_title, company_name, start_date, end_date, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (applicant_id, job_title or "Employee", company or "Company", f"{start_yr}-01-01", f"{datetime.now().year}-01-01", responsibilities or "")
            )

        # --- 6. ATOMIC SKILL MASTER LINKING ---
        if skills:
            cur.execute("DELETE FROM applicant_skills WHERE applicant_id = %s", (applicant_id,))
            skills_list = [s.strip() for s in skills.split(",") if s.strip()]
            for sk in skills_list:
                cur.execute("SELECT skill_id FROM skills_master WHERE skill_name = %s", (sk,))
                sk_row = cur.fetchone()
                if sk_row:
                    skill_id = sk_row[0]
                else:
                    cur.execute("INSERT INTO skills_master (skill_name) VALUES (%s)", (sk,))
                    skill_id = cur.lastrowid
                cur.execute(
                    "INSERT IGNORE INTO applicant_skills (applicant_id, skill_id) VALUES (%s, %s)",
                    (applicant_id, skill_id)
                )

        # --- 7. PROCESS RULES & MACHINE LEARNING (CART) ---
        rejection_reasons = []

        if age < req_age:
            rejection_reasons.append(f"Age ({age}) is below requirements ({req_age})")

        if experience < req_exp:
            rejection_reasons.append(f"Experience ({experience} yrs) is below requirements ({req_exp} yrs)")

        skills_list_eval = [s.strip() for s in skills.split(",") if s.strip()]
        if len(skills_list_eval) < 2:
            rejection_reasons.append("Insufficient skills listed (minimum 2 required)")

        try:
            cart_result = cart_predict_from_form(
                age=age,
                education_level=education_level,
                experience=experience,
                skills_raw=skills,
            )
            model_score = cart_result["model_score"]
            confidence = cart_result["probability_percent"]
            session["cart_details"] = cart_result
        except Exception as e:
            logger.error(f"CART prediction error: {e}")
            model_score = 0.5
            confidence = 50.0

        if not rejection_reasons and model_score < 0.55:
            rejection_reasons.append("Assessment score below qualification threshold")

        # Set final eligibility evaluation
        if not rejection_reasons:
            eligibility = "Eligible"
            screening_status = "Passed Screening"
            final_reason = "You meet all requirements for this position."
        else:
            eligibility = "Not Eligible"
            screening_status = "Failed Screening"
            final_reason = "Not Eligible: " + "; ".join(rejection_reasons)

        # --- 8. SUBMIT APPLICATION JUNCTION RECORD ---
        cur.execute(
            """
            INSERT INTO applications (job_id, applicant_id, screening_status, applied_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (job_id, applicant_id, screening_status)
        )
        mysql.connection.commit()
        cur.close()

        # Update tracking sessions
        session["name"] = name
        session["position"] = position
        session["result"] = eligibility
        session["confidence"] = int(round(confidence))
        session["reason"] = final_reason

        if eligibility == "Eligible":
            flash(f"Application Submitted! {final_reason}", "success")
            try:
                send_step1_completed_email(email, name, position)
            except Exception as e:
                logger.error(f"Error sending Step 1 email: {e}")
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
        "SELECT email, username, contact_num FROM users WHERE user_id = %s",
        (user_id,),
    )
    user = cur.fetchone()
    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for("auth.login"))

    email, username, contact = user
    
    cur.execute(
        """
        SELECT app.screening_status, j.job_name 
        FROM applications app 
        JOIN applicants a ON a.applicant_id = app.applicant_id
        JOIN jobs j ON j.job_id = app.job_id
        WHERE a.user_id = %s
        ORDER BY app.application_id DESC LIMIT 1
        """,
        (user_id,),
    )
    eligibility_row = cur.fetchone()
    cur.close()

    name = session.get("name")
    eligible_applicant = "Eligible" if eligibility_row and eligibility_row[0] == "Passed Screening" else "Not Eligible"
    position = eligibility_row[1] if eligibility_row else session.get("position")
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
    cur.execute(
        """
        SELECT a.full_name, u.email, u.contact_num, j.job_name 
        FROM applications app
        JOIN applicants a ON a.applicant_id = app.applicant_id
        JOIN users u ON u.user_id = a.user_id
        JOIN jobs j ON j.job_id = app.job_id
        """
    )
    applicants = cur.fetchall()
    cur.close()

    return render_template("view_applicants.html", applicants=applicants)


@applicants_bp.route("/viewchat")
def view_chatbot():
    if "user_id" not in session:
        flash("You must be logged in to view chatbot data.", "error")
        return redirect(url_for("auth.login"))

    # Mocked chatbot interaction for visualization
    chatbot = []
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
        "SELECT email, username, contact_num FROM users WHERE user_id = %s",
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
    if "user_id" not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    if request.method == "POST":
        position = request.form.get("position")
        yearexperience = int(request.form.get("yearexperience") or 0)

        cur.execute(
            "SELECT username, email, contact_num FROM users WHERE user_id = %s",
            (user_id,),
        )
        user_info = cur.fetchone()
        if not user_info:
            flash("User not found.", "error")
            cur.close()
            return redirect(url_for("auth.login"))

        name, email, contact = user_info

        # Create base profile if not existing
        cur.execute("SELECT applicant_id FROM applicants WHERE user_id = %s", (user_id,))
        app_row = cur.fetchone()
        if not app_row:
            cur.execute(
                """
                INSERT INTO applicants (user_id, full_name, date_of_birth, current_location)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, name, "1998-01-01", "Unknown")
            )
            applicant_id = cur.lastrowid
        else:
            applicant_id = app_row[0]

        # Get job id matching positional requirements
        cur.execute("SELECT job_id FROM jobs WHERE job_name = %s", (position,))
        job_row = cur.fetchone()
        if job_row:
            job_id = job_row[0]
            cur.execute(
                """
                INSERT INTO applications (job_id, applicant_id, screening_status)
                VALUES (%s, %s, 'Pending')
                """,
                (job_id, applicant_id)
            )
            
            # Record base work experience 
            cur.execute(
                """
                INSERT INTO work_experience (applicant_id, job_title, company_name, start_date)
                VALUES (%s, %s, %s, %s)
                """,
                (applicant_id, "Candidate", "Company", f"{datetime.now().year - yearexperience}-01-01")
            )
            mysql.connection.commit()

        cur.close()
        return redirect(url_for("applicants.preapp"))

    # Fetch latest preapp
    cur.execute(
        """
        SELECT a.full_name, u.email, u.contact_num, j.job_name, 
               COALESCE((SELECT SUM(TIMESTAMPDIFF(YEAR, start_date, COALESCE(end_date, CURDATE()))) FROM work_experience WHERE applicant_id = a.applicant_id), 0) AS years_experience,
               app.screening_status
        FROM applications app
        JOIN applicants a ON a.applicant_id = app.applicant_id
        JOIN users u ON u.user_id = a.user_id
        JOIN jobs j ON j.job_id = app.job_id
        WHERE u.user_id = %s
        ORDER BY app.application_id DESC
        LIMIT 1
        """,
        (user_id,),
    )
    applicant = cur.fetchone()
    cur.close()

    if applicant:
        name, email, contact, position, yearexperience, status = applicant
        app_needed = False
    else:
        name = email = contact = position = yearexperience = status = None
        app_needed = True

    return render_template(
        "pre-app.html",
        name=name,
        email=email,
        contact=contact,
        position=position,
        yearexperience=yearexperience,
        eligibility="Eligible" if status == "Passed Screening" else ("Pending" if status == "Pending" else "Not Eligible"),
        level="N/A",
        status=status,
        confidence=50,
        app_needed=app_needed,
    )


# ---------- Profile & Photo Management ----------


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
            "UPDATE users SET profile_photo = %s WHERE user_id = %s",
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
        "SELECT email, username, contact_num, user_type FROM users WHERE user_id = %s",
        (user_id,),
    )
    user = cur.fetchone()

    cur.execute("SELECT * FROM applicants WHERE user_id = %s", (user_id,))
    applicant = cur.fetchone()

    cur.execute(
        """
        SELECT j.job_name, app.screening_status, app.applied_at 
        FROM applications app 
        JOIN applicants a ON a.applicant_id = app.applicant_id
        JOIN jobs j ON j.job_id = app.job_id
        WHERE a.user_id = %s
        """,
        (user_id,)
    )
    applications = cur.fetchall()
    cur.close()

    return render_template(
        "profile.html",
        email=user[0],
        username=user[1],
        contact=user[2],
        profile_photo=None,
        position=applications[0][0] if applications else None,
        eligibility=applications[0][1] if applications else None,
        yearexperience=5 if applicant else None,
        qualified=None,
        applications=applications,
    )


# ---------- HR views per job + applicant approve/deny ----------


@applicants_bp.route("/job/<path:position>")
def job_applicants(position):
    if "user_id" not in session:
        if request.args.get("modal") == "1":
            return jsonify({"error": "not_logged_in"}), 401
        flash("You must be logged in to view applicants.", "error")
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()
    cur.execute("SELECT user_type FROM users WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    if not row or row[0] != "HR":
        cur.close()
        if request.args.get("modal") == "1":
            return jsonify({"error": "not_authorized"}), 403
        flash("You are not authorized to view this page.", "error")
        return redirect(url_for("applicants.dashboard"))

    position = unquote(position)

    # Perform structural JOIN to isolate applicants per job description
    cur.execute(
        """
        SELECT 
            app.application_id,
            a.full_name,
            u.email,
            u.contact_num,
            COALESCE((SELECT SUM(TIMESTAMPDIFF(YEAR, start_date, COALESCE(end_date, CURDATE()))) FROM work_experience WHERE applicant_id = a.applicant_id), 0) AS years_experience,
            COALESCE((SELECT degree_level FROM educations WHERE applicant_id = a.applicant_id ORDER BY graduation_year DESC LIMIT 1), 'N/A') AS education_level,
            app.screening_status
        FROM applications app
        JOIN applicants a ON a.applicant_id = app.applicant_id
        JOIN users u ON u.user_id = a.user_id
        JOIN jobs j ON j.job_id = app.job_id
        WHERE j.job_name = %s
        ORDER BY app.application_id DESC
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
                "level": "N/A",
                "qualified": r[6],
                "confidence": 85,
                "eligibility": "Eligible" if r[6] == "Passed Screening" else "Not Eligible",
            }
        )

    if request.args.get("modal") == "1":
        return jsonify({"position": position, "applicants": applicants})

    return render_template(
        "job_applicants.html",
        position=position,
        applicants=applicants,
    )


@applicants_bp.route("/applicant-decision-json", methods=["POST"])
def applicant_decision_json():
    if "user_id" not in session:
        return jsonify({"error": "not_logged_in"}), 401

    data = request.get_json() or {}
    application_id = data.get("applicant_id")  # maps to application junction identity
    decision = data.get("decision")
    position = data.get("position")

    if not application_id or decision not in ("approve", "deny") or not position:
        return jsonify({"error": "invalid_data"}), 400

    new_status = "Passed Screening" if decision == "approve" else "Failed Screening"

    try:
        cur = mysql.connection.cursor()

        # Update applications status directly inside the junction records
        cur.execute(
            "UPDATE applications SET screening_status = %s WHERE application_id = %s",
            (new_status, application_id),
        )

        cur.execute(
            """
            SELECT a.full_name, u.email 
            FROM applications app 
            JOIN applicants a ON a.applicant_id = app.applicant_id
            JOIN users u ON u.user_id = a.user_id
            WHERE app.application_id = %s
            """,
            (application_id,),
        )
        app_row = cur.fetchone()
        mysql.connection.commit()
        cur.close()

        if decision == "approve" and app_row:
            name, email = app_row
            try:
                send_step1_completed_email(email, name, position)
            except Exception as e:
                logger.error(f"Error sending Step 1 email: {e}")

        return jsonify({
            "ok": True,
            "eligibility": "Eligible" if decision == "approve" else "Not Eligible",
            "status_label": "Approved" if decision == "approve" else "Denied",
            "status_code": 1 if decision == "approve" else 2
        })

    except Exception as e:
        logger.error(f"applicant_decision_json error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------- Misc helpers ----------


# ---------- Misc helpers ----------


@applicants_bp.route("/save_experience", methods=["POST"])
def save_experience():
    user_id = session.get("user_id")
    yearexperience = request.form.get("yearexperience")

    if user_id and yearexperience:
        try:
            cur = mysql.connection.cursor()
            
            # Find the applicant_id linked to this user
            cur.execute("SELECT applicant_id FROM applicants WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            
            if row:
                applicant_id = row[0]
                # Check for an existing work experience record
                cur.execute("SELECT work_exp_id FROM work_experience WHERE applicant_id = %s LIMIT 1", (applicant_id,))
                exp_row = cur.fetchone()
                
                start_yr = datetime.now().year - int(yearexperience)
                start_date = f"{start_yr}-01-01"
                
                if exp_row:
                    cur.execute(
                        "UPDATE work_experience SET start_date = %s WHERE work_exp_id = %s",
                        (start_date, exp_row[0])
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO work_experience (applicant_id, job_title, company_name, start_date) 
                        VALUES (%s, 'Candidate', 'Company', %s)
                        """,
                        (applicant_id, start_date)
                    )
                mysql.connection.commit()
                session["experience"] = int(yearexperience)
                cur.close()
                return jsonify({"success": "Experience saved successfully"})
            
            cur.close()
            return jsonify({"error": "Applicant profile not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Invalid data"}), 400


@applicants_bp.route("/progress")
def show_progress():
    """
    Progress view per position based on chatbot qualification status.
    Uses the normalized relational mappings instead of legacy flat tables.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT 
                j.job_name,
                COUNT(
                    CASE 
                        WHEN LOWER(c.qualification_status) = 'qualified'
                        THEN 1 ELSE NULL
                    END
                ) AS percentage
            FROM applications app
            JOIN applicants a ON a.applicant_id = app.applicant_id
            JOIN jobs j ON j.job_id = app.job_id
            LEFT JOIN chatbot c ON c.user_id = a.user_id
            GROUP BY j.job_name
            """
        )
        progress_data = cur.fetchall()
        cur.close()
        return render_template("progress.html", progress_data=progress_data)
    except Exception as e:
        logger.error(f"Error in show_progress: {e}")
        return f"Error: {e}", 500


@applicants_bp.route("/check_email")
def check_email():
    email = request.args.get("email")
    if not email:
        return jsonify({"exists": False})
        
    try:
        cur = mysql.connection.cursor()
        # Relational check: join users with applicants since email is normalized in users table
        cur.execute(
            """
            SELECT a.applicant_id 
            FROM applicants a 
            JOIN users u ON u.user_id = a.user_id 
            WHERE u.email = %s
            """, 
            (email,)
        )
        existing_user = cur.fetchone()
        cur.close()
        return jsonify({"exists": bool(existing_user)})
    except Exception as e:
        logger.error(f"Error in check_email: {e}")
        return jsonify({"error": str(e)}), 500


@applicants_bp.route("/applicants")
def view_applications():
    """
    Comprehensive list of applicants utilizing structural relationships 
    (joining users to fetch normalized user contact information and emails).
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT 
                a.applicant_id,
                a.full_name,
                u.email,
                u.contact_num,
                a.current_location,
                a.preferred_location,
                a.resume_url
            FROM applicants a
            JOIN users u ON u.user_id = a.user_id
            """
        )
        applications = cur.fetchall()
        cur.close()
        return render_template("applications.html", applications=applications)
    except Exception as e:
        logger.error(f"Error in view_applications: {e}")
        return f"Error: {e}", 500


@applicants_bp.route("/get_applicants")
def get_applicants():
    """
    JSON API returning applicant names, jobs, and calculated experience totals 
    directly from work history records.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT 
                a.full_name, 
                j.job_name,
                COALESCE(
                    (SELECT SUM(TIMESTAMPDIFF(YEAR, start_date, COALESCE(end_date, CURDATE()))) 
                     FROM work_experience 
                     WHERE applicant_id = a.applicant_id), 0
                ) AS years_experience
            FROM applications app
            JOIN applicants a ON a.applicant_id = app.applicant_id
            JOIN jobs j ON j.job_id = app.job_id
            """
        )
        rows = cur.fetchall()
        cur.close()

        applicants = [
            {"name": r[0], "position": r[1], "experience": int(r[2])} for r in rows
        ]
        return jsonify(applicants)
    except Exception as e:
        logger.error(f"Error in get_applicants: {e}")
        return jsonify({"error": str(e)}), 500