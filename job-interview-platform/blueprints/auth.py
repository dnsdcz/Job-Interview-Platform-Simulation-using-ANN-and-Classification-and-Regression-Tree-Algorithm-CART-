# blueprints/auth.py
import requests
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import mysql, logger
from services.otp_service import generate_otp, verify_otp
from services.email_service import send_otp_email

auth_bp = Blueprint("auth", __name__)


def _role_target(user_type):
    role = (user_type or "").strip().lower()
    # Separate HR and Admin routing here
    if role in ("hr", "hrpage"):
        return "hr.hr_dashboard"
    if role == "admin":
        return "admin.dashboard"
    if role == "applicant":
        return "applicants.dashboard"
    return None


@auth_bp.route("/")
def index():
    return render_template("index.html")


# ---------- 1. APPLICANT LOGIN ----------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        # Redirect logged-in users based on their usertype
        cur = mysql.connection.cursor()
        cur.execute("SELECT user_type FROM users WHERE user_id = %s", (session["user_id"],))
        result = cur.fetchone()
        cur.close()

        if result:
            target = _role_target(result[0])
            if target:
                return redirect(url_for(target))

    error = None
    if request.method == "POST":
        # --- Verify reCAPTCHA ---
        recaptcha_response = request.form.get('g-recaptcha-response')
        if not recaptcha_response:
            error = "Please complete the reCAPTCHA verification."
            return render_template("login.html", error=error)

        secret_key = "6LfLvBEsAAAAACY2WgJ9qMIEjaNDEWMOPH_Xw73w"
        verify_url = "https://www.google.com/recaptcha/api/siteverify"
        data = {
            'secret': secret_key,
            'response': recaptcha_response
        }

        try:
            verify_response = requests.post(verify_url, data=data).json()
            if not verify_response.get('success'):
                error = "reCAPTCHA verification failed. Please try again."
                return render_template("login.html", error=error)
        except:
            error = "Unable to verify reCAPTCHA. Please try again."
            return render_template("login.html", error=error)

        # --- Login Logic ---
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT user_id, password, user_type FROM users WHERE email = %s",
            (email,),
        )
        result = cur.fetchone()
        cur.close()

        if result:
            if check_password_hash(result[1], password):
                # STRICT ROLE CHECK: Deny HR/Admin here
                if result[2] in ('HR', 'Admin'):
                    error = "Staff members must use the Staff Login portal (/staff-login)."
                else:
                    session["user_id"] = result[0]
                    session["email"] = email
                    flash("Login successful! Welcome, Applicant.", "success")
                    return redirect(url_for("applicants.dashboard"))
            else:
                error = "Incorrect password."
        else:
            error = "Email not found."

    return render_template("login.html", error=error)


# ---------- 2. STAFF LOGIN (HR / ADMIN) ----------
@auth_bp.route("/staff-login", methods=["GET", "POST"])
def staff_login():
    if "user_id" in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT user_type FROM users WHERE user_id = %s", (session["user_id"],))
        result = cur.fetchone()
        cur.close()
        if result:
            target = _role_target(result[0])
            if target:
                return redirect(url_for(target))

    error = None
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT user_id, password, user_type FROM users WHERE email = %s", (email,))
        result = cur.fetchone()
        cur.close()

        if result:
            if check_password_hash(result[1], password):
                # STRICT ROLE CHECK: Deny Applicants here
                if result[2] == 'Applicant':
                    error = "Applicants must use the standard login page (/login)."
                else:
                    session["user_id"] = result[0]
                    session["email"] = email
                    
                    target = _role_target(result[2])
                    
                    # Direct them to their specific dashboards
                    if target == "hr.hr_dashboard":
                        flash("Login successful! Welcome to the HR Portal.", "success")
                        return redirect(url_for("hr.hr_dashboard"))
                    elif target == "admin.dashboard":
                        flash("Login successful! Welcome, Admin.", "success")
                        return redirect(url_for("admin.dashboard"))
                    else:
                        error = "Role recognized, but dashboard not found."
            else:
                error = "Incorrect password."
        else:
            error = "Email not found."

    return render_template("staff_log.html", error=error)


# ---------- 3. STAFF REGISTRATION ----------
@auth_bp.route("/register-staff", methods=["GET", "POST"])
def register_staff():
    if "user_id" in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT user_type FROM users WHERE user_id = %s", (session["user_id"],))
        result = cur.fetchone()
        cur.close()
        if result:
            target = _role_target(result[0])
            if target:
                return redirect(url_for(target))

    if request.method == "POST":
        # --- Verify reCAPTCHA ---
        recaptcha_response = request.form.get("g-recaptcha-response")
        secret_key = "6LfLvBEsAAAAACY2WgJ9qMIEjaNDEWMOPH_Xw73w"

        try:
            recaptcha_verify = requests.post(
                "https://www.google.com/recaptcha/api/siteverify",
                data={"secret": secret_key, "response": recaptcha_response}
            ).json()

            if not recaptcha_verify.get("success"):
                return render_template("register_staff.html", error="Recaptcha verification failed.")
        except:
            return render_template("register_staff.html", error="Unable to verify reCAPTCHA.")

        # --- Capture Form Data ---
        email = request.form["email"]
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        contact_num = request.form.get("contact_num")
        
        # Ensure only HR or Admin can be selected here
        usertype = request.form.get("user_type")
        if usertype not in ["Admin", "HR"]:
            return render_template("register_staff.html", error="Invalid staff role selected.")

        cur = mysql.connection.cursor()
        
        # Check if email or username is taken
        cur.execute("SELECT email, username FROM users WHERE email = %s OR username = %s", (email, username))
        existing_user = cur.fetchone()
        
        if existing_user:
            cur.close()
            existing_email, existing_username = existing_user
            if existing_email == email:
                return render_template("register_staff.html", error="Email already registered.")
            else:
                return render_template("register_staff.html", error="Username is already taken.")

        # Insert new staff member
        cur.execute("""
            INSERT INTO users (email, username, password, user_type, contact_num)
            VALUES (%s, %s, %s, %s, %s)
        """, (email, username, password, usertype, contact_num))

        mysql.connection.commit()
        cur.close()

        flash(f"Staff account created successfully for {username}. Please log in.", "success")
        return redirect(url_for("auth.staff_login"))

    return render_template("register_staff.html")


# ---------- 4. SESSION / LOGOUT / APPLICANT REGISTRATION ----------
@auth_bp.route("/check_session")
def check_session():
    """Check if user is logged in"""
    if "user_id" in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT user_type FROM users WHERE user_id = %s", (session["user_id"],))
        result = cur.fetchone()
        cur.close()

        if result:
            return {"logged_in": True, "user_type": result[0]}

    return {"logged_in": False}


@auth_bp.route("/logout")
def logout():
    session.clear()
    response = redirect(url_for("auth.login"))
    # Prevent caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, post-check=0, pre-check=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT user_type FROM users WHERE user_id = %s", (session["user_id"],))
        result = cur.fetchone()
        cur.close()

        if result:
            target = _role_target(result[0])
            if target:
                return redirect(url_for(target))

    if request.method == "POST":
        # --- Verify reCAPTCHA ---
        recaptcha_response = request.form.get("g-recaptcha-response")
        secret_key = "6LfLvBEsAAAAACY2WgJ9qMIEjaNDEWMOPH_Xw73w"

        recaptcha_verify = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": secret_key, "response": recaptcha_response}
        ).json()

        if not recaptcha_verify.get("success"):
            return render_template("register.html", error="Recaptcha verification failed.")

        # --- Continue registration ---
        email = request.form["email"]
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        usertype = "Applicant" # Hardcoded to Applicant for public registration
        contact_num = request.form.get("contact_num")

        cur = mysql.connection.cursor()
        
        # Check if both email or username are already taken
        cur.execute("SELECT email, username FROM users WHERE email = %s OR username = %s", (email, username))
        existing_user = cur.fetchone()
        
        if existing_user:
            cur.close()
            existing_email, existing_username = existing_user
            if existing_email == email:
                return render_template("register.html", error="Email already registered.")
            else:
                return render_template("register.html", error="Username is already taken. Please choose another.")

        # Insert new user
        cur.execute("""
            INSERT INTO users (email, username, password, user_type, contact_num)
            VALUES (%s, %s, %s, %s, %s)
        """, (email, username, password, usertype, contact_num))

        mysql.connection.commit()
        cur.close()

        flash("Registration successful. You can now log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


# ---------- 5. SUPPORT / PRIVACY ----------
@auth_bp.route("/support")
def support():
    return render_template("support.html")

@auth_bp.route("/landing")
def landing():
    return render_template("index.html")

@auth_bp.route("/privacy")
def privacy():
    return render_template("privacy.html")


# ---------- 6. FORGOT PASSWORD / OTP ----------
@auth_bp.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        cur = mysql.connection.cursor()
        cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if not user:
            flash("Email not found.", "error")
            return redirect(url_for("auth.forgot_password"))

        otp = generate_otp(email)
        if send_otp_email(email, otp):
            flash("OTP sent to your email.", "info")
            return redirect(url_for("auth.verify_otp_route", email=email))
        
        flash("Failed to send OTP.", "error")
        return redirect(url_for("auth.forgot_password"))

    return render_template("forgotpass.html")


@auth_bp.route("/verify_otp/<email>", methods=["GET", "POST"])
def verify_otp_route(email):
    if request.method == "POST":
        otp_input = request.form["otp"]
        if verify_otp(email, otp_input):
            session["verified_email"] = email
            session["verification_time"] = True  # flag only
            flash("OTP verified. You may now reset your password.", "success")
            return redirect(url_for("auth.reset_password", token=email))

        flash("Invalid or expired OTP.", "error")
        return render_template("verify_otp.html", email=email)

    return render_template("verify_otp.html", email=email)


@auth_bp.route("/resend_otp", methods=["POST"])
def resend_otp():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return {"success": False, "message": "Email is required."}

    cur = mysql.connection.cursor()
    cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()

    if not user:
        return {"success": False, "message": "Email not found in our system."}

    otp = generate_otp(email)
    ok = send_otp_email(email, otp)
    
    if ok:
        return {"success": True, "message": "OTP resent successfully."}
    return {"success": False, "message": "Failed to send OTP."}


@auth_bp.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token):
    verified_email = session.get("verified_email")
    if not verified_email or verified_email != token:
        flash("Session expired or invalid. Please verify your OTP again.", "error")
        return redirect(url_for("auth.forgot_password"))

    if request.method == "POST":
        new_password = generate_password_hash(request.form["password"])
        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE users SET password = %s WHERE email = %s",
            (new_password, token),
        )
        mysql.connection.commit()
        cur.close()
        
        session.pop("verified_email", None)
        session.pop("verification_time", None)
        flash("Password has been reset successfully.", "success")
        return redirect(url_for("auth.login"))

    return render_template("reset.html")