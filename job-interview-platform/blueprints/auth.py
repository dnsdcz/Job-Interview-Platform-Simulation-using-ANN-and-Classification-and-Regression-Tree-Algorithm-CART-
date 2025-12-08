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


@auth_bp.route("/")
def index():
    return render_template("index.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    # Add this check at the beginning
    if "user_id" in session:
        # Redirect logged-in users based on their usertype
        cur = mysql.connection.cursor()
        cur.execute("SELECT usertype FROM users WHERE id = %s",
                    (session["user_id"],))
        result = cur.fetchone()
        cur.close()

        if result:
            if result[0] == "hrpage":
                return redirect(url_for("hr.hr_dashboard"))
            elif result[0] == "applicant":
                return redirect(url_for("applicants.dashboard"))

    error = None
    if request.method == "POST":
        # Verify reCAPTCHA
        recaptcha_response = request.form.get('g-recaptcha-response')
        if not recaptcha_response:
            error = "Please complete the reCAPTCHA verification."
            return render_template("login.html", error=error)

        # Verify with Google
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

        # Original login logic
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT id, password, usertype FROM users WHERE email = %s", (
                email,)
        )
        result = cur.fetchone()
        cur.close()

        if result:
            if check_password_hash(result[1], password):
                session["user_id"] = result[0]
                session["email"] = email
                if result[2] == "hrpage":
                    flash("Login successful! Welcome, HR.", "success")
                    return redirect(url_for("hr.hr_dashboard"))
                elif result[2] == "applicant":
                    flash("Login successful! Welcome, Applicant.", "success")
                    return redirect(url_for("applicants.dashboard"))
                else:
                    error = "Usertype not recognized."
            else:
                error = "Incorrect password."
        else:
            error = "Email not found."

    return render_template("login.html", error=error)


@auth_bp.route("/check_session")
def check_session():
    """Check if user is logged in"""
    if "user_id" in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT usertype FROM users WHERE id = %s",
                    (session["user_id"],))
        result = cur.fetchone()
        cur.close()

        if result:
            return {"logged_in": True, "usertype": result[0]}

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
    # Add this check at the beginning
    if "user_id" in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT usertype FROM users WHERE id = %s",
                    (session["user_id"],))
        result = cur.fetchone()
        cur.close()

        if result:
            if result[0] == "hrpage":
                return redirect(url_for("hr.hr_dashboard"))
            elif result[0] == "applicant":
                return redirect(url_for("applicants.dashboard"))

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
        usertype = "applicant"
        contact_number = request.form.get("contact_number")

        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            return render_template("register.html", error="Email already registered.")

        cur.execute("""
            INSERT INTO users (email, username, password, usertype, contact_number)
            VALUES (%s, %s, %s, %s, %s)
        """, (email, username, password, usertype, contact_number))

        mysql.connection.commit()
        cur.close()

        flash("Registration successful. You can now log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


# ---------- Support / Privacy ----------


@auth_bp.route("/support")
def support():
    return render_template("support.html")


@auth_bp.route("/landing")
def landing():
    return render_template("index.html")


@auth_bp.route("/privacy")
def privacy():
    return render_template("privacy.html")


# ---------- Forgot password / OTP ----------


@auth_bp.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
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
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
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
