from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from io import BytesIO
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
model = load_model('model.h5') 

# MySQL config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''  # add if you have a password
app.config['MYSQL_DB'] = 'job_applications'
app.config['MYSQL_CURSORCLASS'] = 'binary'  # needed for binary file saving

mysql = MySQL(app)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    form = request.form
    resume = request.files['resume']

    name = form['name']
    age = int(form['age'])
    address = form['address']
    email = form['email']
    education_level = form['education_level']
    degree = form['degree']
    major = form['major']
    experience = int(form['experience'])
    skills = form['skills']
    position = form['position']
    contact = form['contact']

    filename = secure_filename(resume.filename)
    resume_data = resume.read()

    features = preprocess_form_data(age, education_level, experience, skills)

    prediction = model.predict(features)
    model_score = float(prediction[0][0])
    confidence = model_score * 100

    skills_list = [s.strip().lower() for s in skills.split(',') if s.strip()]
    skill_count = len(skills_list)

    rule_eligible = True
    rejection_reason = ""

    if age < 21:
        rule_eligible = False
        rejection_reason = "Minimum age requirement not met"
    if education_level == "high_school" and experience < 2:
        rule_eligible = False
        rejection_reason = "Insufficient experience for education level"
    if skill_count < 2:
        rule_eligible = False
        rejection_reason = "Insufficient skills listed"

    base_threshold = 0.55
    if education_level in ["master", "phd"]:
        base_threshold -= 0.05
    if experience >= 5:
        base_threshold -= 0.05
    if skill_count >= 5:
        base_threshold -= 0.05
    if 25 <= age <= 40:
        base_threshold -= 0.03

    if not rule_eligible and model_score < 0.8:
        result = "Not Eligible"
        result_reason = rejection_reason
    elif rule_eligible and model_score > base_threshold:
        result = "Eligible"
        result_reason = "Meets all qualifications"
    elif model_score > 0.75:
        result = "Eligible"
        result_reason = "Exceptionally strong candidacy"
    else:
        result = "Not Eligible"
        result_reason = "Does not meet overall qualification threshold"

    eligibility = result
    print(f"Eligibility: {result}")

    try:
        cursor = mysql.connection.cursor()
        insert_query = """
            INSERT INTO applicants (name, email, contact, position, resume_filename, resume_data, eligibility)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (name, email, contact, position, filename, resume_data, eligibility))
        mysql.connection.commit()
        applicant_id = cursor.lastrowid
    except Exception as e:
        return f"Database Error: {e}"
    finally:
        cursor.close()

    return render_template('result.html', 
                           result=result,
                           reason=result_reason, 
                           name=name,
                           confidence=round(confidence, 1),
                           position=position,
                           applicant_id=applicant_id)

@app.route('/resume/<int:applicant_id>')
def download_resume(applicant_id):
    admin_key = request.args.get('key')
    if admin_key != 'superadmin123':
        return "Unauthorized access", 403

    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT resume_filename, resume_data FROM applicants WHERE id = %s", (applicant_id,))
        result = cursor.fetchone()
        if result:
            filename, file_data = result
            return send_file(BytesIO(file_data), download_name=filename, as_attachment=True)
        else:
            return "Resume not found", 404
    except Exception as e:
        return f"Database Error: {e}"
    finally:
        cursor.close()

def preprocess_form_data(age, education_level, experience, skills):
    education_map = {
        "high_school": 1,
        "vocational": 2,
        "associate": 3,
        "bachelor": 4,
        "master": 5,
        "phd": 6
    }
    education_score = education_map.get(education_level, 1)
    experience_score = min(15, experience) / 15
    skills_list = [s.strip().lower() for s in skills.split(',') if s.strip()]
    skill_count = len(skills_list)
    skill_score = min(10, skill_count) / 10
    age_normalized = (age - 20) / (65 - 20)
    age_normalized = max(0, min(1, age_normalized))
    interview_score = (skill_score * 0.6) + (experience_score * 0.4)

    return np.array([[experience_score, education_score/6, skill_score, interview_score]])
