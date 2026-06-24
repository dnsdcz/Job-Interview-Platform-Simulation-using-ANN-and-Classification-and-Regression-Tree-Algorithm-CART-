# blueprints/admin.py
from flask import Blueprint, render_template, session, redirect, url_for
from extensions import mysql

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/admin/dashboard")
def dashboard():
    # 1. Security Check: Ensure user is logged in
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    # 2. Security Check: Ensure the user is actually an Admin
    cur.execute("SELECT username, user_type FROM users WHERE user_id = %s", (user_id,))
    user = cur.fetchone()
    
    if not user or user[1] != 'Admin':
        cur.close()
        # If they aren't an admin, kick them out to the regular login
        return redirect(url_for("auth.login"))

    username = user[0]

    # 3. Gather Admin-specific Data (Example Queries)
    # Total Users
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    # Total HR Staff
    cur.execute("SELECT COUNT(*) FROM users WHERE user_type = 'HR'")
    total_hr = cur.fetchone()[0]

    # Total Applicants
    cur.execute("SELECT COUNT(*) FROM users WHERE user_type = 'Applicant'")
    total_applicants = cur.fetchone()[0]

    cur.close()

    # 4. Render the Admin Template
    return render_template(
        "admin_dashboard.html",
        username=username,
        total_users=total_users,
        total_hr=total_hr,
        total_applicants=total_applicants
    )