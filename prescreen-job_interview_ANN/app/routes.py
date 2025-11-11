from flask import Flask, render_template, request, send_file, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
from mysql.connector import Error
from werkzeug.utils import secure_filename
import numpy as np
import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        database="job_applications"
    )

app = Flask(__name__)
model = load_model('model.h5') 



#application form
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#eligibility result
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

    # Process features for model prediction
    features = preprocess_form_data(age, education_level, experience, skills)

    # Get prediction
    prediction = model.predict(features)
    model_score = float(prediction[0][0])
    confidence = model_score * 100  # For display
    
 
    print(f"Features for {name}: {features}")
    print(f"Raw prediction: {model_score}")
    
    # Get skills list for counting
    skills_list = [s.strip().lower() for s in skills.split(',') if s.strip()]
    skill_count = len(skills_list)
    
    # 1. Apply business rules (hard requirements)
    rule_eligible = True
    rejection_reason = ""
    
    # Minimum requirements
    if age < 21:
        rule_eligible = False
        rejection_reason = "Minimum age requirement not met"
    if education_level == "high_school" and experience < 2:
        rule_eligible = False
        rejection_reason = "Insufficient experience for education level"
    if skill_count < 2:
        rule_eligible = False
        rejection_reason = "Insufficient skills listed"
    
    # 2. Calculate dynamic threshold based on candidate attributes
    base_threshold = 0.55  # Start with base threshold
    
    # Adjust threshold based on candidate qualifications
    if education_level in ["master", "phd"]:
        base_threshold -= 0.05  # Lower threshold (easier to qualify) for highly educated
    
    if experience >= 5:
        base_threshold -= 0.05  # Lower threshold for experienced candidates
    
    if skill_count >= 5:
        base_threshold -= 0.05  # Lower threshold for candidates with many skills
    
    # Age factor - favor prime working age
    if 25 <= age <= 40:
        base_threshold -= 0.03
        
    print(f"Dynamic threshold for {name}: {base_threshold}")
    
    # 3. Combined decision logic
    if not rule_eligible and model_score < 0.8:
        # Failed hard requirements and model isn't extremely confident
        result = "Not Eligible"
        result_reason = rejection_reason
    elif rule_eligible and model_score > base_threshold:
        # Passed hard requirements and over dynamic threshold
        result = "Eligible"
        result_reason = "Meets all qualifications"
    elif model_score > 0.75:
        # Very high model score can override other factors
        result = "Eligible"
        result_reason = "Exceptionally strong candidacy"
    else:
        # Default case - use dynamic threshold
        result = "Not Eligible"
        result_reason = "Does not meet overall qualification threshold"

    
    eligibility = result
    print(f"Eligiblity: {result}")
    

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO applicants (name, email, contact, position, resume_filename, resume_data, eligibility)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (name, email, contact, position, filename, resume_data, eligibility))
        applicant_id = cursor.lastrowid  # <-- Get ID of the newly inserted row
        conn.commit()

    except Error as e:
        return f"Database Error: {e}"
    finally:
        # Ensure cursor and connection are closed only if initialized
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
    
    return render_template('result.html', 
                          result=result,
                          reason=result_reason, 
                          name=name,
                          confidence=round(confidence, 1),
                          position=position,
                          applicant_id=applicant_id)

@app.route('/resume/<int:applicant_id>')
def download_resume(applicant_id):

    # Check secret key from query parameter
    admin_key = request.args.get('key')
    
    # Hardcoded key for demo (you can load from env later)
    if admin_key != 'superadmin123':
        return "Unauthorized access", 403

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT resume_filename, resume_data FROM applicants WHERE id = %s", (applicant_id,))
        result = cursor.fetchone()
        if result:
            filename, file_data = result
            return send_file(BytesIO(file_data), download_name=filename, as_attachment=True)
        else:
            return "Resume not found", 404
    except Error as e:
        return f"Database Error: {e}"
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


# Improved preprocessing function that properly evaluates education levels
def preprocess_form_data(age, education_level, experience, skills):
    # Education scoring based on structured education level
    education_map = {
        "high_school": 1,
        "vocational": 2,
        "associate": 3,
        "bachelor": 4,
        "master": 5,
        "phd": 6
    }
    
    # Get education score directly from dropdown selection
    education_score = education_map.get(education_level, 1)  # Default to 1 if not found
    
    # Experience score - normalize to 0-1 range with cap at 15 years
    experience_score = min(15, experience) / 15  
    
    # Skills assessment - improve by counting number of relevant skills
    skills_list = [s.strip().lower() for s in skills.split(',') if s.strip()]
    
    # Count only non-empty skills and normalize with a reasonable maximum
    skill_count = len(skills_list)
    skill_score = min(10, skill_count) / 10  # Cap at 10 skills
    
    # Age factor - normalize age (assuming working age 20-65)
    # Add more weight to 30-45 age range (peak career years)
    age_normalized = (age - 20) / (65 - 20)
    age_normalized = max(0, min(1, age_normalized))
    
    # Create an interview score proxy (weighted combination of skills and experience)
    interview_score = (skill_score * 0.6) + (experience_score * 0.4)
    
    # Return normalized input features in the expected order for the model
    return np.array([[experience_score, education_score/6, skill_score, interview_score]])


@app.route('/api/applicants', methods=['GET'])
def get_applicants():
    # Check for API key in the request
    api_key = request.args.get('key')
    if api_key != 'partner_access_key':  # Change this to a secure key in production
        return jsonify({"error": "Unauthorized access"}), 403
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute("SELECT id, name, email, contact, position, eligibility FROM applicants")
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch all rows and convert to dictionaries
        rows = cursor.fetchall()
        applicants = []
        for row in rows:
            applicant = dict(zip(column_names, row))
            applicants.append(applicant)
            
        return jsonify({"applicants": applicants})
    
    except Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# API endpoint to get a specific applicant
@app.route('/api/applicant/<int:applicant_id>', methods=['GET'])
def get_applicant(applicant_id):
    api_key = request.args.get('key')
    if api_key != 'partner_access_key':  # Change this to a secure key in production
        return jsonify({"error": "Unauthorized access"}), 403
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Execute the query with parameter
        cursor.execute("SELECT id, name, email, contact, position, eligibility FROM applicants WHERE id = %s", (applicant_id,))
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch the applicant
        row = cursor.fetchone()
        
        if row:
            applicant = dict(zip(column_names, row))
            return jsonify(applicant)
        else:
            return jsonify({"error": "Applicant not found"}), 404
            
    except Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)