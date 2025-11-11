import os
import re
import io
import json
import time
import random
import pdfkit
import pdfplumber
import bleach
import traceback
from io import BytesIO
from pdfminer.high_level import extract_text
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import request, session, jsonify, redirect, url_for, render_template
from datetime import datetime, timedelta
from flask_limiter import Limiter
from redis import Redis
from pdfminer.high_level import extract_text
import warnings
from sentence_transformers import SentenceTransformer, util
import logging



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Numerical and Data Handling
import h5py
import numpy as np

# Flask and Extensions
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, session, flash, make_response,
    send_file, send_from_directory
)
from flask_mysqldb import MySQL
from flask_mail import Mail, Message

# Security and Uploads
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# PDF Parsing and Generation
from pdfminer.high_level import extract_text
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pdfkit

# Machine Learning and NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

from flask import Flask
from flask_mysqldb import MySQL

warnings.filterwarnings("ignore")

app = Flask(__name__)


# Only load models once, globally
print("Loading SentenceTransformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SentenceTransformer model loaded.")

print("Loading KeyBERT...")
kw_model = KeyBERT(embedder)
print("✅ KeyBERT model loaded.")

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  
app.config['MYSQL_PASSWORD'] = '' 
app.config['MYSQL_DB'] = 'auth_db' 

mysql = MySQL(app)

app.secret_key = 'supersecretkey'  

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="redis://localhost:6379"
)

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='aceview18@gmail.com',
    MAIL_PASSWORD='uelmqlulrxbbkikx'  # Make sure this is an App Password
)
mail = Mail(app)


# Folder setup
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create summary reports directory
os.makedirs('summary_reports', exist_ok=True)

# Initialize KeyBERT and SentenceTransformer model
kw_model = KeyBERT()
resume_text_global = None

# In-memory OTP store
otp_store = {}

@app.route('/')
def index():
    """Landing page route"""
    return render_template('index.html')

role_questions = {
    "business_analyst": {
        "junior": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",

            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",

            "How do you prioritize tasks when handling multiple small projects?",
            "How do you handle conflicts between team members?",
            "What is the difference between a project and a program in IT?",
            "How do you stay organized when working on multiple deliverables?"
        ],
        "mid": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",

            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",

            "How do you balance quality, time, and cost in a constrained project?",
            "How do you ensure proper communication between developers, QA, and business stakeholders?",
            "What methods do you use to manage project risks?",
            "How do you ensure cross-functional teams are aligned on goals?"
        ],
        "senior": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",

            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",

            " Describe a time when you had to make a difficult decision that impacted the entire team.",
            "Tell me about a program that failed and how you responded.",
            " What’s your approach to resource allocation across multiple high-priority programs?",
            "How do you evaluate whether a program should be continued, pivoted, or stopped?"
        ],
        "special": [
            "Tell me about yourself.",
            "Why are you leaving your current job?",
            "How do you handle criticism?",
            "Why should we hire you?",

            "Tell me about a time you had to solve a difficult problem at work.",
            "Give an example of a time you supported a teammate under pressure.",
            "Have you ever taken the lead on a project? What happened?",
            "Give an example of a mistake you made and how you handled it.",

            "Describe a time you coached or mentored other program/project managers.",
            "Describe a scenario where your technical understanding of IT architecture helped resolve a program issue.",
            "How do you forecast risk and opportunity over multi-year IT programs?",
            "What innovations have you introduced to improve program delivery or stakeholder engagement?"
        ]
    },
    "project_manager": {
        "junior": [
            "How do you prioritize tasks when handling multiple small projects?",
            "How do you handle conflicts between team members?",
            "What is the difference between a project and a program in IT?",
            "How do you stay organized when working on multiple deliverables?"
        ],
        "mid": [
            "How do you address situations where project requirements change during the development phase?",
            "Could you provide an example of a project where you had to manage multiple stakeholders?",
            "If you had the opportunity to enhance one aspect of the Business Analysis process, what would you focus on and why?",
            "What metrics do you track during a project, and how do you assess whether the project is on the right path toward success?"

        ],
        "senior": [
            "How do you align business analysis with organizational strategy?",
            "Can you describe a time when your analysis influenced the direction or outcome of a project?",
            "How do you assess the effectiveness of a newly implemented business process or change?",
            "What is your experience with Agile methodologies, and how do you adjust your business analysis approach to fit within Agile frameworks?"
        ],
        "special": [
            "What advanced business analysis methodologies or techniques do you employ to manage complex, large-scale projects?",
            "Could you share an example where you successfully led a team of business analysts on a high-profile project?",
            "If you were tasked with creating a new methodology for business analysis, what would it look like and why?",
            "How do you handle conflicting or contradictory data when making critical business recommendations?"
        ]
    },
    "java_developer": {
        "junior": [
            "Explain the difference between int[] arr = new int[5]; and int[] arr = {1, 2, 3, 4, 5};",
            "Can you explain the concept of inheritance and give a simple example?",
            "How would you create and use an ArrayList in Java?",
            "Can you describe a small Java program you’ve written and what it did?"
        ],
        "mid": [
            "What are the main principles of OOP and how does Java implement them?",
            "Explain the differences between ArrayList, LinkedList, and HashMap",
            "How does garbage collection work in Java??",
            "Explain how you would connect a Java application to a database (JDBC or ORM)."
        ],
        "senior": [
            "How do you approach managing multi-threading in Java? Can you provide examples of situations where multi-threading was necessary?",
            "Tell us about a time when you mentored junior developers. What strategies did you use to help them improve their skills?",
            " How would you go about designing a scalable Java application? What potential challenges would you anticipate, and how would you address them?",
            "Explain a time you mentored junior analysts.",
            "What is the difference between a synchronized block and a synchronized method in Java?"   
        ],
        "special": [
            "How do you ensure high availability and fault tolerance in a distributed Java system?",
            "Describe your experience with designing enterprise-level Java applications. What were the most critical design decisions you made?",
            "If Java were to be replaced by a new language tomorrow, what would your transition strategy be?",
            "What is your approach to optimizing Java performance in high-load applications?"
        ]
    }
}



# Keyword-based question templates
keyword_templates = {
    "python": "Tell me about your experience with Python.",
    "django": "Have you used Django in any of your projects?",
    "team": "Describe your role in a team project.",
    "management": "How do you manage responsibilities?",
    "machine learning": "What ML projects have you done?",
    "communication": "How do you ensure good team communication?",
    "sql": "Tell me about your experience with SQL."
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    try:
        text = extract_text(filepath)
        return text
    except Exception as e:
        return f"Error extracting text: {e}"
    
def extract_name(text):
    # Look for "Name: John David Alonzo"
    name_line = re.search(r'Name[:\-]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
    if name_line:
        return name_line.group(1).strip()

    # Fallback: Use email-based pattern
    email_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?=.*@)', text)
    if email_match:
        return email_match.group(1).strip()

    # Final fallback: Try top of resume
    first_lines = text.strip().split('\n')[:2]
    for line in first_lines:
        words = line.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words[:2]):
            return " ".join(words[:2])

    return "Unknown"

def extract_university(text):
    for line in text.splitlines():
        if "University" in line or "College" in line or "Institute" in line:
            return line.strip()
    return "Unknown"

def extract_education(text):
    education_keywords = ['BS', 'Bachelor', 'BA', 'Masters', 'PhD', 'degree']
    lines = text.splitlines()
    matched = [line.strip() for line in lines if any(kw in line for kw in education_keywords)]
    return matched[0] if matched else "Not specified"

def extract_resume_details(text):
    name = extract_name(text)
    skills = extract_skills(text)
    education = extract_education(text)
    university = extract_university(text)
    technologies = extract_technologies(text)
    
    return name, skills, education, university, technologies

def extract_skills(text):
    skills = []
    technical_section = re.search(r'Technical Skills\s*(.*?)\s*(?=Experience|Education|$)', text, re.DOTALL | re.IGNORECASE)
    if technical_section:
        raw_skills = technical_section.group(1).strip().splitlines()
        skills = [skill.strip() for skill in raw_skills if skill.strip()]
    return skills


def extract_technologies(text):
    tech_keywords = ["Python", "Java", "C++", "JavaScript", "Django", "React", "AI", "SQL", "HTML", "CSS"]
    return list({tech for tech in tech_keywords if tech.lower() in text.lower()})


def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

class SimpleNeuralNetwork:
    """Simple neural network implementation without TensorFlow"""
    
    def __init__(self, h5_path):
        """Initialize the model from H5 file"""
        self.weights = []
        self.biases = []
        
        with h5py.File(h5_path, 'r') as f:
            # Extract weights and biases from the H5 file
            model_weights = f['model_weights']
            
            # Get dense layers
            for layer_name in ['dense', 'dense_1', 'dense_2']:
                layer = model_weights[layer_name]
                weight = np.array(layer[layer_name]['kernel:0'])
                bias = np.array(layer[layer_name]['bias:0'])
                
                self.weights.append(weight)
                self.biases.append(bias)
    
    def predict(self, x):
        """Forward pass through the network"""
        # First dense layer with ReLU
        x = np.dot(x, self.weights[0]) + self.biases[0]
        x = relu(x)
        
        # Second dense layer with ReLU
        x = np.dot(x, self.weights[1]) + self.biases[1]
        x = relu(x)
        
        # Output layer with sigmoid
        x = np.dot(x, self.weights[2]) + self.biases[2]
        x = sigmoid(x)
        
        return x

def preprocess_form_data(age, education_level, experience, skills):
    """
    Process applicant form data into model features
    """
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

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']  # ✅ added
        password = generate_password_hash(request.form['password'])
        usertype = request.form['usertype']
        contact_number = request.form.get('contact_number')

        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            return render_template('register.html', error="Email already registered.")
        else:
            cur.execute("""
                INSERT INTO users (email, username, password, usertype, contact_number) 
                VALUES (%s, %s, %s, %s, %s)
            """, (email, username, password, usertype, contact_number))  # ✅ fixed
            mysql.connection.commit()
            cur.close()
            flash("Registration successful. You can now log in.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()  
    return redirect(url_for('index'))

@app.route('/support')
def support():
    # render support page
    return render_template('support.html')

@app.route('/privacy')
def privacy():
    # render privacy page
    return render_template('privacy.html')


@app.route('/applicant')
def applicant():
    if 'user_id' not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for('index'))

    user_id = session['user_id']
    cur = mysql.connection.cursor()

    # Fetch user details
    cur.execute("SELECT email, username, contact_number FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()

    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for('login'))

    email, username, contact = user

    # Check if username is None or empty, and if session has a backup
    if not username:
        fallback_username = session.get('username')
        if fallback_username:
            # Update DB with session username
            cur.execute("UPDATE users SET username = %s WHERE id = %s", (fallback_username, user_id))
            mysql.connection.commit()
            username = fallback_username

    # Check application data
    cur.execute("SELECT eligibility, position FROM applicants WHERE id = %s", (user_id,))
    eligibility_row = cur.fetchone()
    cur.close()

    name = session.get('name')
    eligible_applicant = session.get('result')
    position = session.get('position')
    reason = session.get('reason')
    confidence = session.get('confidence')

    return render_template('applicant.html',
                           email=email,
                           name=name,
                           username=username,
                           contact=contact,
                           result=eligible_applicant,
                           
                           position=position,
                           reason=reason,
                           confidence=confidence)



@app.route('/save_experience', methods=['POST'])
def save_experience():
    user_id = session.get('user_id')  # Get user_id from session
    yearexperience = request.form.get('yearexperience')  # Form data

    if user_id and yearexperience:
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                UPDATE applicants
                SET yearexperience = %s
                WHERE user_id = %s
            """, (yearexperience, user_id))
            mysql.connection.commit()
            cur.close()
            return jsonify({"success": "Experience saved successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid data"}), 400



@app.route('/viewapp')
def view_applicants():
    if 'user_id' not in session:
        flash("You must be logged in to view applicants.", "error")
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT name, email, contact, position FROM applicants")
    applicants = cur.fetchall()
    cur.close()

    return render_template('view_applicants.html', applicants=applicants)

@app.route('/viewchat')
def view_chatbot():
    if 'user_id' not in session:
        flash("You must be logged in to view applicants.", "error")
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT user_name, position, experience, qualification_status FROM chatbot")
    chatbot = cur.fetchall()
    cur.close()

    return render_template('view_chatbot.html', chatbot=chatbot)



@app.route('/hr')
def hr():
    if 'user_id' in session:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        # Get HR user details
        cur.execute("SELECT email, username FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        
        # Get chatbot interaction count
        cur.execute("SELECT COUNT(*) FROM chatbot")
        chatbot_count = cur.fetchone()[0]
        chatbot_progress = 100 if chatbot_count > 0 else 0
        
        # Get total applicant count
        cur.execute("SELECT COUNT(*) FROM applicants")
        applicant_count = cur.fetchone()[0]
        
        # Get completed applications
        cur.execute("SELECT COUNT(*) FROM applicants WHERE status = 'completed'")
        completed_applications = cur.fetchone()[0]
        applicant_progress = (completed_applications / applicant_count) * 100 if applicant_count > 0 else 0
        
        # Get the maximum applicants allowed for each position from the limits table
        cur.execute("""
            SELECT position, max_allowed FROM position_limits
        """)
        limits_results = cur.fetchall()
        position_limits = {position: max_allowed for position, max_allowed in limits_results}
        
        # Get current count of applicants per position
        cur.execute("""
            SELECT position, COUNT(*) as current_count
            FROM applicants
            GROUP BY position
        """)
        counts_results = cur.fetchall()
        position_counts = {position: count for position, count in counts_results}
        
        # Define standard position list
        positions = [
            'Business Analyst',
            'Project Analyst',
            'Java Developer'
        ]
        
        # Prepare progress data based on position limits
        progress_data = []
        
        for pos in positions:
            # Get the current count for this position (default to 0 if not found)
            current_count = position_counts.get(pos, 0)
            
            # Get the maximum allowed for this position (default to 100 if not found)
            max_allowed = position_limits.get(pos, 10)
            
            # Calculate percentage based on current vs max allowed
            if max_allowed > 0:
                percentage = round((current_count / max_allowed) * 10, 2)
                # Cap percentage at 100%
                percentage = min(percentage, 10)
            else:
                percentage = 0.0
            
            # Format the position name and add the data
            progress_data.append({
                'position': pos,
                'percentage': percentage,
                'current': current_count,
                'max': max_allowed
            })
        
        # Let's log the data to help with debugging
        print("Progress data:", progress_data)
        
        cur.close()
        
        if user:
            email, username = user
            return render_template('Hrpage.html', 
                                  email=email, 
                                  username=username,
                                  chatbot_count=chatbot_count,
                                  applicant_count=applicant_count,
                                  chatbot_progress=chatbot_progress,
                                  applicant_progress=applicant_progress,
                                  progress_data=progress_data)
        else:
            flash("User not found.", "error")
            return redirect(url_for('login'))
    else:
        flash("You need to log in first.", "error")
        return redirect(url_for('login'))




@app.route('/set-username', methods=['POST'])
def set_username():
    if 'user_id' not in session:
        flash("You must be logged in to set a username.", "error")
        return redirect(url_for('login'))

    user_id = session['user_id']
    username = request.form['username']

    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET username = %s WHERE id = %s", (username, user_id))
    mysql.connection.commit()
    cur.close()

    flash("Username updated successfully!", "success")
    return redirect(url_for('hr'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None  # <-- Add this line to hold error message

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, password, usertype FROM users WHERE email = %s", (email,))
        result = cur.fetchone()
        cur.close()

        if result:
            if check_password_hash(result[1], password):  # Ensure the password matches
                session['user_id'] = result[0]  # Store user ID in session
                session['email'] = email
                if result[2] == 'hrpage':
                    flash("Login successful! Welcome, HR.", "success")
                    return redirect(url_for('hr'))
                elif result[2] == 'applicant':
                    flash("Login successful! Welcome, Applicant.", "success")
                    return redirect(url_for('dash'))
                else:
                    error = "Usertype not recognized."  # <-- Show this if usertype is invalid
            else:
                error = "Incorrect password."  # <-- Show error if password is wrong
        else:
            error = "Email not found."  # <-- Show error if email does not exist

        return render_template('login.html', error=error)  # <-- Pass error to template

    return render_template('login.html', error=error)  # <-- Pass error even on GET


from flask import request, flash, redirect, url_for

# Allow only PDF files for resume upload
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import MySQLdb.cursors

@app.route('/schedule')
def schedules():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)  # DictCursor for dict rows

    # Get position limits
    cur.execute("SELECT position, max_allowed FROM position_limits")
    position_limits = cur.fetchall()

    # Get chatbot limits
    cur.execute("SELECT position, max_allowed FROM chatbot_limits")
    chatbot_limits = cur.fetchall()

    # Get current applicants count grouped by position
    cur.execute("SELECT position, COUNT(*) as current_count FROM applicants GROUP BY position")
    current_counts = cur.fetchall()

    counts_dict = {row['position']: row['current_count'] for row in current_counts}

    progress_data = []
    for limit in position_limits:
        pos = limit['position']
        max_allowed = limit['max_allowed']
        current = counts_dict.get(pos, 0)
        percent = int((current / max_allowed) * 100) if max_allowed > 0 else 0
        progress_data.append({
            'position': pos,
            'current': current,
            'max_allowed': max_allowed,
            'percent': percent
        })

    for chatbot_limit in chatbot_limits:
        pos = chatbot_limit['position']
        max_allowed = chatbot_limit['max_allowed']
        current = counts_dict.get(pos, 0)
        percent = int((current / max_allowed) * 100) if max_allowed > 0 else 0
        progress_data.append({
            'position': 'Chatbot Interactions',
            'current': current,
            'max_allowed': max_allowed,
            'percent': percent
        })

    cur.close()

    return render_template('Settinghr.html', progress_data=progress_data)


@app.route('/set_pax', methods=['POST'])
def set_pax():
    position = request.form.get('position')
    max_allowed = request.form.get('max_allowed')

    try:
        cursor = mysql.connection.cursor()
        # Update if exists, else insert
        query = """
            INSERT INTO position_limits (position, max_allowed)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE max_allowed = VALUES(max_allowed)
        """
        cursor.execute(query, (position, max_allowed))
        mysql.connection.commit()
        cursor.close()
        flash("Max limit set successfully!", "success")
    except Exception as e:
        flash(f"Error setting max limit: {e}", "danger")

    return redirect('/schedule')  # Or your desired return page

@app.route('/set_chat', methods=['POST'])
def set_chat():
    position = request.form.get('position')
    max_allowed = request.form.get('max_allowed')

    if not position or not max_allowed:
        flash("All fields are required!", "error")
        return redirect(request.referrer)

    cur = mysql.connection.cursor()

    # Check if record exists
    cur.execute("SELECT id FROM chatbot_limits WHERE position = %s", (position,))
    existing = cur.fetchone()

    if existing:
        # Update existing
        cur.execute(
            "UPDATE chatbot_limits SET max_allowed = %s WHERE position = %s",
            (max_allowed, position)
        )
    else:
        # Insert new
        cur.execute(
            "INSERT INTO chatbot_limits (position, max_allowed) VALUES (%s, %s)",
            (position, max_allowed)
        )

    mysql.connection.commit()
    cur.close()

    flash(f"Limit set for {position} successfully!", "success")
    return redirect(request.referrer)


@app.route('/save_schedule', methods=['POST'])
def save_schedule():
    date = request.form.get('date')
    time = request.form.get('time')
    end_date = request.form.get('endDate') or None

    # Get all recurring days as a JSON string or comma-separated
    recurring_days = request.form.getlist('recurring')  # list of selected days
    recurring_json = json.dumps(recurring_days) if recurring_days else None

    try:
        cursor = mysql.connection.cursor()
        query = """
            INSERT INTO schedules (schedule_date, schedule_time, recurring_days, end_date)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (date, time, recurring_json, end_date))
        mysql.connection.commit()
        cursor.close()
        flash("Schedule saved successfully!", "success")
    except Exception as e:
        flash(f"Error saving schedule: {e}", "danger")

    return redirect('/schedule')  # Or wherever you want to go next

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'profile_photo' not in request.files:
        flash('No file part')
        return redirect(url_for('profile'))

    file = request.files['profile_photo']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('profile'))

    if file and allowed_file(file.filename):
        # Read the image content as binary
        image_data = file.read()

        # Save image data to DB for the logged-in user
        cursor = mysql.connection.cursor()
        cursor.execute("""
            UPDATE users SET profile_photo = %s WHERE id = %s
        """, (image_data, session['user_id']))
        mysql.connection.commit()
        cursor.close()

        flash('Profile photo saved to database successfully')
        return redirect(url_for('profile'))

    else:
        flash('Invalid file type')

        


@app.route('/profile')
def prof():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    profile_photo = "user1.jpg" 
    user_id = session['user_id']
    cursor = mysql.connection.cursor()

    # Fetch user info
    cursor.execute("SELECT email, username, contact_number, usertype, profile_photo FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()

    # Fetch applicant info if user is applicant
    cursor.execute("SELECT * FROM applicants WHERE email = %s", (user[0],))
    applicant = cursor.fetchone()

    # Fetch all applications
    cursor.execute("SELECT * FROM applicants")
    applications = cursor.fetchall()

    cursor.close()

    return render_template('profile.html',
        email=user[0],
        username=user[1],
        contact=user[2],
        profile_photo=user[4],
        position=applicant[5] if applicant else None,
        eligibility=applicant[6] if applicant else None,
        yearexperience=applicant[7] if applicant else None,
        qualified=applicant[10] if applicant else None,
        applications=applications
    )

@app.route('/progress')
def show_progress():
    cursor = mysql.connection.cursor(dictionary=True)
    cursor.execute("SELECT position, qualified AS percentage FROM applicants")
    progress_data = cursor.fetchall()
    cursor.close()
    return render_template('progress.html', progress_data=progress_data)


@app.route('/check_email')
def check_email():
    email = request.args.get('email')

    # Check if email exists in the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM applicants WHERE email = %s", (email,))
    existing_user = cur.fetchone()
    cur.close()

    if existing_user:
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})


@app.route('/save', methods=['POST'])
def save():
    # Get the data sent from the frontend
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')

    # Create a cursor to interact with the MySQL database
    cur = mysql.connection.cursor()

    # SQL query to insert the question and answer into the chatbot_interactions table
    query = "INSERT INTO `chatbot` (`user_name`, `position`, `qualifications`, `qualification_status`, `created_at`) VALUES (%s, %s, %s, NOW())"
    
    # Insert data (you can replace 'user_email' with the actual user email if available)
    cur.execute(query, (session.get('email'), question, answer))
    
    # Commit the transaction
    mysql.connection.commit()

    # Close the cursor
    cur.close()

    # Respond to the client
    return jsonify({"message": "Data saved successfully!"})


#Application form 
@app.route('/submit', methods=['POST'])
def submit_application():
    if request.method == 'POST':
        try:
            form = request.form
            name = form['name']
            email = form['email']
            contact = form['contact']
            position = form['position']

            cur = mysql.connection.cursor()

            # Check if email already exists
            cur.execute("SELECT id FROM applicants WHERE email = %s", (email,))
            if cur.fetchone():
                flash("This email has already been used to apply. Please use another.", "error")
                cur.close()
                return redirect(url_for('dash'))

            # Check position limit
            cur.execute("SELECT max_allowed FROM position_limits WHERE position = %s", (position,))
            position_limit = cur.fetchone()
            if position_limit:
                max_allowed = position_limit[0]
                cur.execute("SELECT COUNT(*) FROM applicants WHERE position = %s", (position,))
                current_count = cur.fetchone()[0]
                if current_count >= max_allowed:
                    flash(f"The position '{position}' is already full. Please select another position.", "error")
                    cur.close()
                    return redirect(url_for('dash'))
            else:
                flash(f"Invalid position selected: {position}", "error")
                cur.close()
                return redirect(url_for('dash'))

            age = int(form.get('age', 25))
            education_level = form.get('education_level', 'bachelor')
            experience = int(form.get('experience', 0))
            skills = form.get('skills', '')

            resume_filename = None
            resume_data = None
            if 'resume' in request.files:
                resume = request.files['resume']
                if resume.filename:
                    resume_filename = secure_filename(resume.filename)
                    resume_data = resume.read()

            # Predict eligibility
            features = preprocess_form_data(age, education_level, experience, skills)
            model = SimpleNeuralNetwork('model.h5')
            prediction = model.predict(features)
            model_score = float(prediction[0][0])
            confidence = model_score * 100

            # Rule-based checks
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
            qualified_status = "Qualified" if eligibility == "Eligible" else "Not Qualified"
            status = 'Pending'

            user_id = session.get('user_id')
            if not user_id:
                flash("You must be logged in to apply.", "error")
                return redirect(url_for('login'))

            # Insert application
            cur.execute('''
                INSERT INTO applicants (user_id, name, email, contact, position, eligibility, yearexperience, status, qualified)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (user_id, name, email, contact, position, eligibility, experience, status, qualified_status))
            mysql.connection.commit()
            cur.close()

            session['name'] = name
            session['email'] = email
            session['contact'] = contact
            session['position'] = position
            session['result'] = eligibility
            session['reason'] = result_reason
            session['confidence'] = round(confidence, 1)
            session['qualified'] = qualified_status

            return redirect(url_for('dash'))

        except Exception as e:
            flash(f"Unexpected error: {e}", "error")
            return redirect(url_for('dash'))





@app.route('/prescreenn')
def pre():
    email = session.get('email')
    return render_template('prescreen.html', email=email)









@app.route('/dashboard')
def dash():
    if 'user_id' not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cur = mysql.connection.cursor()
    
    # Fetch user details
    cur.execute("SELECT email, username, contact_number FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for('login'))
    
    email, username, contact = user

    # Check existing application by email
    cur.execute("SELECT * FROM applicants WHERE email = %s", (email,))
    applicant_row = cur.fetchone()
    applicant = None
    if applicant_row:
        applicant = {
            'user_id': applicant_row[0],
            'name': applicant_row[2],
            'email': applicant_row[3],
            'contact': applicant_row[4],
            'position': applicant_row[5],
            'eligibility': applicant_row[6],
            'yearexperience': applicant_row[7],
            'level': applicant_row[8],
            'status': applicant_row[9],
            'qualified': applicant_row[10],
            'confidence': applicant_row[11]
        }

    cur.execute("""
        SELECT user_id, user_name, position, qualification_status, created_at 
        FROM chatbot 
        WHERE user_id = %s AND position != 'resume' 
        ORDER BY created_at DESC LIMIT 1
    """, (user_id,))
    chatbot_data = cur.fetchone()

    if not chatbot_data:
        # Try with email instead of repeating username again
        cur.execute("""
            SELECT user_id, user_name, position, qualification_status, created_at 
            FROM chatbot 
            WHERE user_id = %s AND position != 'resume' 
            ORDER BY created_at DESC LIMIT 1
        """, (user_id,))
        chatbot_data = cur.fetchone()
    # Fetch all position limits with applicant count
    cur.execute("""
        SELECT pl.id, pl.position, pl.max_allowed, 
               COUNT(a.position) AS current_count
        FROM position_limits pl
        LEFT JOIN applicants a ON pl.position = a.position
        GROUP BY pl.id, pl.position, pl.max_allowed
    """)
    positions = cur.fetchall()

    # Convert to list of dicts
    position_limits = [
        {
            'id': p[0],
            'position': p[1],
            'max_allowed': p[2],
            'current_count': p[3],
            'is_full': p[3] >= p[2]
        }
        for p in positions
    ]

    # Prepare chatbot dict if exists
    chatbot_dict = None
    if chatbot_data:
        created_at = chatbot_data[4]
        if isinstance(created_at, str):
            try:
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            except Exception:
                created_at = None

        chatbot_dict = {
            'user_id': chatbot_data[0],
            'user_name': chatbot_data[1],
            'position': chatbot_data[2],
            'qualification_status': chatbot_data[3],
            'created_at': created_at
        }
    
     # Fetch position limits
    cur.execute("SELECT id, position, max_allowed FROM position_limits")
    positions = cur.fetchall()
    position_limits = [{'id': p[0], 'position': p[1], 'max_allowed': p[2]} for p in positions]

    # Optional session values
    name = session.get('name', username)
    result = session.get('result')
    reason = session.get('reason')
    confidence = session.get('confidence')
    position = session.get('position')
    qualification_status = session.get('qualification_status', '')
    applied_role = session.get('position', 'Business Analyst')


    cur.close()

    return render_template('dashboard.html',
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
                           chatbot_data=chatbot_dict,
                           application_data=applicant,
                           has_applied=applicant is not None,
                           has_chatbot_assessment=chatbot_dict is not None,
                           applicant=applicant,
                           position_limits=position_limits)



@app.route('/applicants')
def view_applications():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM applicants")
        applications = cur.fetchall()
        cur.close()
        
        return render_template('applications.html', applications=applications)
        
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/resume/<int:user_id>')
def get_resume(user_id):
    # Check for admin access (simple key-based authentication)
    if request.args.get('key') != 'superadmin123':
        return "Access denied", 403
    
    try:
        # In a real app, you would fetch the resume file path from the database
        # For this demo, we'll assume a static path
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'])
        
        # Check if any file exists for this applicant
        # In a real app, you'd get the exact filename from the database
        if os.path.exists(resume_path):
            # In a real app, replace 'sample.pdf' with the actual filename
            return send_from_directory(resume_path, 'sample.pdf', as_attachment=True)
        else:
            return "Resume not found", 404
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user:
            otp = random.randint(100000, 999999)
            otp_store[email] = {'otp': otp, 'timestamp': time.time()}

            msg = Message(
                subject="Your OTP for Password Reset",
                sender=app.config['MAIL_USERNAME'],
                recipients=[email]
            )
            msg.body = f"Your OTP is {otp}. Use this OTP to reset your password. It is valid for 5 minutes."

            try:
                mail.send(msg)
                flash("OTP sent to your email.", "info")
                return redirect(url_for('verify_otp', email=email))
            except Exception as e:
                flash(f"Error sending OTP: {e}", "error")
                return redirect(url_for('forgot_password'))
        else:
            flash("Email not found.", "error")
            return redirect(url_for('forgot_password'))

    return render_template('forgotpass.html')

@app.route('/verify_otp/<email>', methods=['GET', 'POST'])
def verify_otp(email):
    if request.method == 'POST':
        otp_input = request.form['otp']
        stored = otp_store.get(email)

        if stored and int(otp_input) == stored['otp'] and (time.time() - stored['timestamp']) < 300:
            # Clear the OTP after successful verification
            del otp_store[email]
            
            # Store verification status in session
            session['verified_email'] = email
            session['verification_time'] = time.time()
            
            flash("OTP verified. You may now reset your password.", "success")
            return redirect(url_for('reset_password', token=email))
        else:
            flash("Invalid or expired OTP.", "error")
            return render_template('verify_otp.html', email=email)

    return render_template('verify_otp.html', email=email)

@app.route('/resend_otp', methods=['POST'])
def resend_otp():
    data = request.get_json()
    email = data.get('email')
    
    # Validate email exists in our system
    if not email:
        return jsonify({"success": False, "message": "Email is required."})
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    
    if not user:
        return jsonify({"success": False, "message": "Email not found in our system."})

    # Generate new OTP
    otp = random.randint(100000, 999999)
    otp_store[email] = {'otp': otp, 'timestamp': time.time()}

    msg = Message(
        subject="Your OTP for Password Reset",
        sender=app.config['MAIL_USERNAME'],
        recipients=[email]
    )
    msg.body = f"Your OTP is {otp}. Use this OTP to reset your password. It is valid for 5 minutes."

    try:
        mail.send(msg)
        return jsonify({"success": True, "message": "OTP resent successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": "Failed to send OTP.", "error": str(e)})

@app.route('/reset/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # Validate that the user has been verified recently (within 10 minutes)
    verified_email = session.get('verified_email')
    verification_time = session.get('verification_time')
    
    if (not verified_email or 
        verified_email != token or 
        not verification_time or 
        (time.time() - verification_time) > 600):  # 10 minutes
        flash("Session expired or invalid. Please verify your OTP again.", "error")
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = generate_password_hash(request.form['password'])

        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET password = %s WHERE email = %s", (new_password, token))
        mysql.connection.commit()
        cur.close()
        
        # Clear the session data after successful password reset
        session.pop('verified_email', None)
        session.pop('verification_time', None)

        flash("Password has been reset successfully.", "success")
        return redirect(url_for('login'))

    return render_template('reset.html')


#position
@app.route('/apply', methods=['POST'])
def apply():
    position = request.json.get('position')
    if not position:
        return jsonify({'message': 'Position is required.'}), 400

    # Check the number of applicants for the given position
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM applications WHERE position = %s", (position,))
    filled_count = cur.fetchone()[0]
    max_positions = 50

    # Check if the position is full
    if filled_count >= max_positions:
        cur.close()
        return jsonify({'message': 'Position limit reached.'}), 400

    # Insert the new application into the database
    cur.execute("INSERT INTO applications (position) VALUES (%s)", (position,))
    mysql.connection.commit()
    cur.close()

    return jsonify({'message': 'Application submitted successfully.'})






@app.route('/get_applicants')
def get_applicants():
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, position, experience FROM applicants")
    rows = cursor.fetchall()
    conn.close()

    applicants = [{"name": r[0], "position": r[1], "experience": r[2]} for r in rows]
    return jsonify(applicants)

@app.route('/preapp', methods=['GET', 'POST'])
def preapp():
    if 'user_id' not in session:
        flash("You need to log in first.", "error")
        return redirect(url_for('login'))

    user_id = session['user_id']
    cur = mysql.connection.cursor()

    # Handle new form submission
    if request.method == 'POST':
        position = request.form.get('position')
        yearexperience = request.form.get('yearexperience')

        # Optional: Fetch user info to insert full applicant data
        cur.execute("SELECT username, email, contact_number FROM users WHERE id = %s", (user_id,))
        user_info = cur.fetchone()
        if not user_info:
            flash("User not found.", "error")
            cur.close()
            return redirect(url_for('login'))

        name, email, contact = user_info

        # Default values for qualification
        eligibility = "Pending"
        qualified = "Pending"
        confidence = 0

        # Insert into applicants table
        cur.execute("""
            INSERT INTO applicants 
            (user_id, name, email, contact, position, yearexperience, eligibility, qualified, confidence) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_id, name, email, contact, position, yearexperience, eligibility, qualified, confidence))

        mysql.connection.commit()
        return redirect(url_for('preapp'))

    # Fetch the latest applicant record for the logged-in user
    cur.execute("""
        SELECT name, email, contact, position, yearexperience, eligibility, qualified, confidence 
        FROM applicants 
        WHERE user_id = %s 
        ORDER BY id DESC 
        LIMIT 1
    """, (user_id,))
    applicant = cur.fetchone()

    cur.close()

    if applicant:
        name, email, contact, position, yearexperience, eligibility, qualified, confidence = applicant
        app_needed = False
    else:
        name = email = contact = position = yearexperience = eligibility = qualified = confidence = None
        app_needed = True

    return render_template('pre-app.html',
                           name=name,
                           email=email,
                           contact=contact,
                           position=position,
                           yearexperience=yearexperience,
                           eligibility=eligibility,
                           qualified=qualified,
                           confidence=confidence,
                           app_needed=app_needed)


@app.route('/chatapp')
def capp():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect if not logged in

    user_id = session['user_id']
    cur = mysql.connection.cursor()

    # Fetch username, email, contact from users table (like in /preapp)
    cur.execute("SELECT email, username, contact_number FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()

    if not user:
        cur.close()
        flash("User not found.", "error")
        return redirect(url_for('login'))

    email, username, contact = user

    # Use session fallback if username is missing
    if not username:
        fallback_username = session.get('username')
        if fallback_username:
            cur.execute("UPDATE users SET username = %s WHERE id = %s", (fallback_username, user_id))
            mysql.connection.commit()
            username = fallback_username

    # Fetch the latest chatbot interview result for the current user
    cur.execute("""
        SELECT user_name, position, experience, qualification_status, confidence, average_score, created_at 
        FROM chatbot 
        WHERE user_id = %s 
        ORDER BY created_at DESC 
        LIMIT 1
    """, (user_id,))
    result = cur.fetchone()
    cur.close()

    # If result exists, user has taken chatbot interview
    if result:
        name = result[0]
        position = result[1]
        experience = result[2]
        qualified = result[3]
        confidence = result[4]
        average_score = result[5]
        created_at = result[6]
        chatbot_needed = False
    else:
        name = position = experience = qualified = confidence = average_score = created_at = None
        chatbot_needed = True  # No record means interview not taken

    return render_template('chat-app.html',
                           name=name,
                           email=email,
                           contact=contact,
                           username=username,
                           position=position,
                           experience=experience,
                           qualified=qualified,
                           confidence=confidence,
                           average_score=average_score,
                           created_at=created_at,
                           chatbot_needed=chatbot_needed)
























#Application Initialization and Configuration
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

#Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import os
import re
from werkzeug.utils import secure_filename
import bleach
import logging



@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        resume_file = request.files['resume']

        if resume_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text.strip():
            return jsonify({"error": "Failed to extract text from PDF. Please upload a valid resume."}), 400

        # 5) Parse out name/position/experience
        resume_data = parse_resume(resume_text)
        position   = resume_data.get("position", "unknown")
        experience = resume_data.get("experience", 0)
        name       = resume_data.get("name", "Candidate")

        print("Extracted resume data:", resume_data)

        # 6) Store in session so fetch_questions can see it
        session['resume_data']  = resume_data
        session['position']     = position
        session['experience']   = experience
        session['name']         = name

        # 7) Tell the front end to show questions now
        return jsonify({
            "show_questions": True,
            "position": position,
            "experience": experience,
            "name": name
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500




@app.route('/fetch_questions_after_resume', methods=['GET'])
def fetch_questions_after_resume():
    try:
        resume_data = session.get('resume_data', {})
        role = resume_data.get('position')
        experience = int(resume_data.get('experience', 0))

        level = 'junior' if experience < 3 else 'mid' if experience < 6 else 'senior'
        
        logger.info(f"Extracted role: {role}, experience: {experience} years, level: {level}")
        
        # 👉 Add the logging line here
        logger.info(f"Looking for role_questions[{role}][{level}]")

        questions = role_questions.get(role, {}).get(level, [])

        if not questions:
            logger.warning(f"No questions found for role '{role}' at level '{level}'")

        return jsonify({'questions': questions})
    
    except Exception as e:
        logger.exception("Error fetching questions after resume parsing")
        return jsonify({'error': str(e)}), 500




def parse_resume(resume_text):
    data = {
        "name": "Candidate",
        "position": "unknown",
        "experience": 0
    }

    # Normalize text
    resume_text = resume_text.replace("\r", "\n")
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]

    print("RESUME TEXT PREVIEW:", "\n".join(lines[:10]))  # Preview for debugging

    # --- 1. Try extracting name from the top lines ---
    for line in lines[:5]:
        if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+", line):
            data["name"] = line.strip()
            break

    # --- 2. Position detection (only 3 roles) ---
    joined_text = " ".join(lines).lower()
    if "business analyst" in joined_text:
        data["position"] = "business_analyst"
    elif "project manager" in joined_text or "pmp" in joined_text:
        data["position"] = "project_manager"
    elif "java developer" in joined_text or "software engineer" in joined_text or "developer" in joined_text:
        data["position"] = "java_developer"

    # --- 3. Experience detection ---
    exp_match = re.search(r"(\d{1,2})\s+years?", joined_text)
    if exp_match:
        data["experience"] = int(exp_match.group(1))

    return data


import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    text = ""
    try:
        # Ensure file pointer is at start
        file.stream.seek(0)

        # Open file from stream
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        return text.strip()
    except Exception as e:
        print("PDF extraction failed:", e)
        return ""


from pdf2image import convert_from_bytes
import pytesseract

def extract_text_with_ocr(file):
    try:
        images = convert_from_bytes(file.read())
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print("OCR extraction failed:", e)
        return ""


def clean_resume_text(text):
    text = re.sub(r'\n+', '\n', text)  # remove excessive newlines
    text = re.sub(r'\s{2,}', ' ', text)  # normalize spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII characters
    return text


def get_experience_level(years):
    if years <= 4:
        return "Junior"
    elif 5 <= years <= 7:
        return "Mid"
    elif 8 <= years <= 10:
        return "Senior"
    else:
        return "Special"

def get_role_questions(position, difficulty):
    try:
        questions = role_questions.get(position, {}).get(difficulty.lower(), [])
        return questions
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return []

def get_positions_questions(position, experience_level):
    # Normalize keys
    position = position.lower()
    experience_level = experience_level.lower()
    return role_questions.get(position, {}).get(experience_level, [])

    difficulty_normalized = difficulty.lower()

    role_key = role_questions_normalized.get(role.lower())
    if not role_key:
        return []

    try:
        return role_questions[role_key][difficulty_normalized]
    except KeyError:
        return []


@app.route('/get_questions')
def get_questions():
    position = request.args.get('position')
    exp = request.args.get('experience')

    if not position or not exp:
        return jsonify({"error": "Missing parameters"}), 400

    level = get_experience_level(int(exp))
    questions = get_role_questions(position, level)
    return jsonify({"questions": questions})

@app.route('/start-interview', methods=['POST'])
def start_interview():
    data = request.get_json()
    position = data['position']
    years_of_experience = int(data['years_of_experience'])

    level = get_experience_level(years_of_experience)
    questions = get_role_questions(position, level)

    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session['questions'] = questions
    session['current_question'] = 0
    session['answers'] = []

    return jsonify({
        'session_id': session_id,
        'question': questions[0] if questions else "No questions found for this role and level."
    })











# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize SentenceTransformer
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ SentenceTransformer model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading SentenceTransformer model: {e}")
    model = None

# Initialize KeyBERT
try:
    from keybert import KeyBERT
    kw_model = KeyBERT()
    logger.info("✅ KeyBERT model loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️ KeyBERT model unavailable: {e}")
    kw_model = None

# Flask-Limiter
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["10 per minute"])

# Placeholder model answers (should be populated externally)
model_answers = {}

@app.route('/next_question', methods=['POST'])
def next_question():
    try:
        data = request.json
        answer = data.get('answer', '')
        current_question = data.get('question', '')
        
        # Log the request
        logger.info(f"Next question request received. Current question: {current_question[:30]}...")
        
        # Get position and experience level from the session
        position = session.get('position', 'business_analyst')
        
        experience = session.get('experience', 0)
        if experience >= 7:
            experience_level = "special"
        elif experience >= 5:
            experience_level = "senior"
        elif experience >= 3:
            experience_level = "mid"
        else:
            experience_level = "junior"

        
        # Get question set
        role_question_set = get_positions_questions(position, experience_level)
        
        # Initialize question index if missing
        if 'question_index' not in session:
            session['question_index'] = 0
        
        idx = session['question_index']
        
        # Save the answer if provided
        if answer and current_question:
            if 'answers_history' not in session:
                session['answers_history'] = []
            
            session['answers_history'].append({
                "question": current_question,
                "answer": answer
            })
            session.modified = True
            session['question_index'] += 1
            idx += 1
            
            logger.info(f"Answer saved, new question index: {idx}")
        
        # Check if interview is finished
        if idx >= len(role_question_set):
            logger.info("Interview complete")
            return jsonify({
                'finished': True,
                'summary': 'Interview complete',
                'qualification_status': session.get('qualification_status', 'Pending'),
                'answers_history': session.get('answers_history', [])
            })
        
        # Return the next question
        next_question = role_question_set[idx]
        logger.info(f"Sending next question: {next_question[:30]}...")
        
        return jsonify({
            'next_question': next_question,
            'finished': False,
            'feedback': 'Thank you for your answer!',
            'qualification_status': 'In Progress'
        })
    
    except Exception as e:
        logger.error(f"Error in next_question: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'finished': False
        }), 500


@app.route('/get_dynamic_questions')
def get_dynamic_questions():
    global resume_text_global
    if not resume_text_global:
        return jsonify({"questions": ["Please upload a resume first."]})

    keywords = kw_model.extract_keywords(resume_text_global, top_n=10)
    question_list = []
    for kw, _ in keywords:
        for key in keyword_templates:
            if key.lower() in kw.lower():
                question_list.append(keyword_templates[key])

    if not question_list:
        question_list.append("Tell me more about your most recent role.")

    return jsonify({"questions": question_list})





limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)
from flask import request, jsonify


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

user_answers = []
model_answers = {}  # Replace with actual model answers

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ SentenceTransformer model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading SentenceTransformer model: {e}")
    model = None

# Optional: Initialize KeyBERT if available
try:
    from keybert import KeyBERT
    kw_model = KeyBERT()
    logger.info("✅ KeyBERT model loaded successfully")
except ImportError:
    logger.warning("⚠️ KeyBERT not installed. Keyword extraction will not be available.")
    kw_model = None
except Exception as e:
    logger.error(f"❌ Error loading KeyBERT model: {e}")
    kw_model = None

# Initialize limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per minute"]
)


def score_answer_combined(question, answer):
    try:
        if not model:
            raise RuntimeError("SentenceTransformer model is not loaded.")
        if not kw_model:
            raise RuntimeError("KeyBERT model is not loaded.")

        logger.info("Scoring answer...")
        logger.debug(f"Question: {question}")
        logger.debug(f"Answer: {answer}")

        # Sentence embedding + cosine similarity
        question_embedding = model.encode([question])[0]
        answer_embedding = model.encode([answer])[0]
        cosine_score = cosine_similarity([question_embedding], [answer_embedding])[0][0]

        # Keyword extraction from question
        keywords = kw_model.extract_keywords(question, top_n=5)
        keyword_list = [kw[0].lower() for kw in keywords]
        answer_words = set(answer.lower().split())

        matched_keywords = [kw for kw in keyword_list if kw in answer_words]
        keyword_score = len(matched_keywords) / len(keyword_list) if keyword_list else 0.0

        # Final weighted score
        final_score = round((cosine_score * 0.7) + (keyword_score * 0.3), 2)

        # Qualification logic
        if final_score >= 0.7:
            status = "Qualified"
        elif final_score >= 0.5:
            status = "Partially Qualified"
        else:
            status = "Not Qualified"

        return {
            "score": final_score,
            "qualification_status": status,
            "matched_keywords": matched_keywords,
            "total_keywords": keyword_list,
            "cosine_score": round(cosine_score, 2),
            "keyword_score": round(keyword_score, 2)
        }

    except Exception as e:
        logger.error(f"❌ Error in score_answer_combined: {str(e)}", exc_info=True)
        return {
            "score": 0.0,
            "qualification_status": "Error",
            "matched_keywords": [],
            "total_keywords": [],
            "cosine_score": 0.0,
            "keyword_score": 0.0
        }


def get_similarity_score(ans, ref):
    try:
        emb1 = model.encode(ans, convert_to_tensor=True)
        emb2 = model.encode(ref, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()
    except Exception as e:
        logger.error(f"❌ Error computing similarity: {str(e)}", exc_info=True)
        return 0.0


def extract_keywords_from_text(text, num_keywords=5, kw_model=None):
    """
    Extracts top keywords from the given text using KeyBERT or a fallback method.
    
    Args:
        text (str): Input text for keyword extraction.
        num_keywords (int): Number of keywords to return.
        kw_model (KeyBERT): Initialized KeyBERT model instance.

    Returns:
        List[str]: Extracted keyword strings.
    """
    try:
        if kw_model:
            keywords = kw_model.extract_keywords(text, top_n=num_keywords)
            return [kw for kw, _ in keywords]
        else:
            logger.warning("⚠️ KeyBERT not available. Using fallback keyword extraction.")
            stopwords = {
                'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                'to', 'of', 'in', 'for', 'with', 'on', 'at', 'from', 'by'
            }
            words = [w for w in text.lower().split() if len(w) > 2 and w not in stopwords]
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:num_keywords]]
    except Exception as e:
        logger.error(f"❌ Failed to extract keywords: {str(e)}", exc_info=True)
        return []


def explain_score(score):
    """
    Provides a human-readable explanation based on a similarity score.

    Args:
        score (float): The similarity score between 0 and 1.

    Returns:
        str: Explanation of the score.
    """
    try:
        if score > 0.8:
            return "Excellent and highly relevant answer."
        elif score > 0.6:
            return "Good answer with relevant content."
        elif score > 0.4:
            return "Partially relevant. Consider adding more detail or aligning better with the question."
        else:
            return "Answer lacks relevance. Please focus more on the question and include specific details."
    except Exception as e:
        logger.error(f"❌ Error in explain_score: {str(e)}", exc_info=True)
        return "Unable to explain the score."

def get_model_answer(question):
    """
    Returns the expected model answer for a question.

    Args:
        question (str): The interview question.

    Returns:
        str: Expected model answer or a fallback string.
    """
    try:
        return model_answers.get(question, "Default expected answer to compare against.")
    except Exception as e:
        logger.error(f"❌ Error retrieving model answer: {str(e)}", exc_info=True)
        return "Default expected answer to compare against."

def score_answer_internal(question: str, answer: str) -> float:
    # unchanged, but make sure get_similarity_score is called as:
    return get_similarity_score(question.strip(), answer.strip())
    # i.e. question first, answer second


def compute_answer_score(question: str, answer: str, technique: str = "cosine") -> float:
    # fixed argument order inside
    if not answer or not question:
        logger.warning("⚠️ Empty input provided to scoring.")
        return 0.0

    answer = answer.strip().lower()
    question = question.strip().lower()

    try:
        if technique == "cosine":
            return score_answer_internal(question, answer)
        elif technique == "length":
            score = min(len(answer.split()) / 100.0, 1.0)
            logger.info("📏 Length-based score: %.2f", score)
            return round(score, 2)
        elif technique == "hybrid":
            sim_score = score_answer_internal(question, answer)
            len_score = min(len(answer.split()) / 100.0, 1.0)
            hybrid_score = round((sim_score + len_score) / 2, 2)
            logger.info("🔀 Hybrid score (cosine + length): %.2f", hybrid_score)
            return hybrid_score
        else:
            logger.warning("⚠️ Unknown scoring technique '%s'. Defaulting to cosine.", technique)
            return score_answer_internal(question, answer)
    except Exception as e:
        logger.error("❌ Error in compute_answer_score: %s", str(e))
        traceback.print_exc()
        return 0.0


def score_single_answer(question, answer):
    """
    Score a single answer against a question.
    
    Args:
        question (str): The question text
        answer (str): The answer text
        
    Returns:
        dict: Score details
    """
    try:
        if not question.strip() or not answer.strip():
            return {
                "score": 0,
                "explanation": "Empty question or answer.",
                "feedback": "Please provide both question and answer.",
                "qualification_status": "Not Qualified"
            }

        score = get_similarity_score(question, answer)
        explanation = explain_score(score)

        return {
            "score": score,
            "explanation": explanation,
            "feedback": explanation,
            "qualification_status": "Qualified" if score > 0.6 else "Not Qualified"
        }

    except Exception as e:
        print(f"❌ Error scoring answer:\nQuestion: {question}\nAnswer: {answer}\nError: {e}")
        return {
            "score": 0,
            "explanation": "Error during scoring.",
            "feedback": "An error occurred during scoring. Please retry or check the input.",
            "qualification_status": "Not Qualified"
        }

def score_all_answers(qa_pairs):
    """
    Score multiple question-answer pairs.
    
    Args:
        qa_pairs (list): List of dictionaries with question and answer pairs
        
    Returns:
        dict: Summary of scores and qualification status
    """
    total_score = 0
    results = []

    for pair in qa_pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        result = score_answer_internal(question, answer)
        total_score += result["similarity_score"]
        results.append(result)

    avg_score = total_score / len(qa_pairs) if qa_pairs else 0
    qualification_status = "Qualified" if avg_score > 60 else "Not Qualified"

    return {
        "average_score": round(avg_score, 2),
        "confidence": round(avg_score, 2),
        "qualification_status": qualification_status,
        "answers": results
    }

def generate_detailed_advice(answers_history, avg_score, status_counts):
    """
    Generate detailed feedback based on answers history.
    
    Args:
        answers_history (list): List of previous answers with scores
        avg_score (float): Average score across all answers
        status_counts (dict): Count of different qualification statuses
        
    Returns:
        str: Detailed advice text
    """
    if not answers_history:
        return "No advice available."
    
    # Find questions with lowest scores for targeted advice
    answers_with_scores = [(a.get("question", ""), a.get("score", 0), a.get("answer", "")) 
                          for a in answers_history]
    
    # Sort by score ascending
    answers_with_scores.sort(key=lambda x: x[1])
    
    advice = []
    
    # Overall assessment
    if avg_score >= 70:
        advice.append("You performed very well in this interview! Your answers were thorough and relevant.")
    elif avg_score >= 50:
        advice.append("You did well in this interview. Most of your answers addressed the questions appropriately.")
    elif avg_score >= 40:
        advice.append("Your interview performance was satisfactory, but there's room for improvement.")
    else:
        advice.append("This interview indicates you may need more preparation in several areas.")
    
    # Add specific advice based on lowest scoring answers
    if answers_with_scores:
        lowest_q, lowest_score, lowest_a = answers_with_scores[0]
        if len(lowest_q) > 10:  # Make sure we have a valid question
            advice.append(f"Consider improving your response to: '{lowest_q[:50]}...' Your answer was brief or could use more specific examples.")
        
        if len(answers_with_scores) > 1:
            second_q, second_score, second_a = answers_with_scores[1]
            if second_score < 50 and len(second_q) > 10:
                advice.append(f"Also work on questions like: '{second_q[:50]}...' Try to provide more relevant details.")
    
    # General improvement advice
    if status_counts.get("Not Qualified", 0) > 0:
        advice.append("For future interviews, make sure your answers directly address the questions and include specific examples from your experience.")
    
    if avg_score < 60:
        advice.append("Practice explaining your technical skills more clearly, using industry terminology appropriately.")
    
    return " ".join(advice)

def score_with_keywords(answer: str, keywords: list[str]) -> float:
    try:
        if not answer or not keywords:
            return 0.0
        answer = answer.lower()
        match_count = sum(1 for kw in keywords if kw.lower() in answer)
        score = round(match_count / len(keywords), 2) if keywords else 0.0
        logger.info("🔑 Keyword match score: %.2f", score)
        return score
    except Exception as e:
        logger.error("❌ Error in score_with_keywords: %s", str(e))
        traceback.print_exc()
        return 0.0

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

kw_model = KeyBERT(model='all-MiniLM-L6-v2')
from flask import request, jsonify
from datetime import datetime
import traceback





@app.route('/score_answer', methods=['POST'])
def score_answer():
    try:
        data = request.get_json(force=True)
        user_id = data.get("user_id", "anonymous")
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()

        if not question or not answer:
            logger.warning("❌ Missing question or answer in request.")
            return jsonify({"error": "Missing question or answer."}), 400

        try:
            score = compute_answer_score(question, answer, technique="cosine")
            if score is None:
                raise ValueError("Score is None")
        except Exception as e:
            logger.error(f"Error scoring answer for user {user_id}: {e}")
            return jsonify({"error": "An error occurred while scoring the answer. Please try again."}), 500

        explanation = explain_score(score)
        status = "Qualified" if score >= 0.6 else "Unqualified"
        logger.info(f"[{user_id}] ✅ Scored {score:.2f} ({status}) for question: '{question}'")

        return jsonify({
            "score": score,
            "qualification_status": status,
            "feedback": explanation
        })

    except Exception as e:
        logger.error(f"❌ Exception in /score_answer: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error during scoring."}), 500


@app.route("/submit_interview", methods=["POST"])
def submit_interview():
    try:
        data = request.get_json()
        answers = data.get("answers", [])
        user_answers.extend(answers)

        summary = generate_interview_summary(user_answers)
        advice = generate_advice_based_on_score(summary["average_score"])

        avg_score = summary.get("average_score", 0)
        detailed_feedback = advice

        return jsonify({
            "status": "success",
            "average_score": avg_score,
            "feedback": detailed_feedback
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_interview_summary', methods=['GET'])
def get_interview_summary():
    try:
        if not user_answers:
            return jsonify({"error": "No answers submitted yet"}), 400

        summary = score_all_answers(user_answers)
        status_counts = {
            "Qualified": sum(1 for a in user_answers if a["qualification_status"] == "Qualified"),
            "Partially Qualified": sum(1 for a in user_answers if a["qualification_status"] == "Partially Qualified"),
            "Not Qualified": sum(1 for a in user_answers if a["qualification_status"] == "Not Qualified")
        }

        advice = generate_detailed_advice(user_answers, summary["average_score"], status_counts)

        return jsonify({
            "summary": summary,
            "advice": advice,
            "answers": user_answers
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

def evaluate_answer_logic(question, answer):
    result = score_answer_combined(question, answer)
    score = result.get("score", 0)
    feedback = result.get("feedback", "")
    status = result.get("qualification_status", "Not Qualified")
    return score, feedback, status

def score_answer_accurately(question, answer):
    try:
        if not question or not answer:
            return {"error": "Missing question or answer."}

        # Embed question and answer
        q_emb = model.encode(question, convert_to_tensor=True)
        a_emb = model.encode(answer, convert_to_tensor=True)

        cosine_score = util.pytorch_cos_sim(q_emb, a_emb).item()
        cosine_score = round(cosine_score, 2)

        # Keyword matching
        if kw_model:
            keywords = kw_model.extract_keywords(question, top_n=5)
            keyword_list = [k[0].lower() for k in keywords]
            matched_keywords = [kw for kw in keyword_list if kw in answer.lower()]
            keyword_score = round(len(matched_keywords) / len(keyword_list), 2) if keyword_list else 0
        else:
            matched_keywords = []
            keyword_score = 0

        # Final score
        final_score = round((cosine_score * 0.7) + (keyword_score * 0.3), 2)

        status = (
            "Qualified" if final_score >= 0.7 else
            "Partially Qualified" if final_score >= 0.5 else
            "Not Qualified"
        )

        return {
            "question": question,
            "answer": answer,
            "cosine_score": cosine_score,
            "keyword_score": keyword_score,
            "final_score": final_score,
            "matched_keywords": matched_keywords,
            "qualification_status": status
        }

    except Exception as e:
        logger.error(f"❌ Error in score_answer_accurately: {e}")
        return {
            "question": question,
            "answer": answer,
            "cosine_score": 0.0,
            "keyword_score": 0.0,
            "final_score": 0.0,
            "matched_keywords": [],
            "qualification_status": "Error",
            "error": str(e)
        }

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_answer(answer, question, model, kw_model):
    try:
        if not answer.strip() or not question.strip():
            return 0.0  # Avoid empty input

        # Encode using SentenceTransformer
        emb_question = model.encode([question])[0]
        emb_answer = model.encode([answer])[0]

        similarity = cosine_similarity([emb_question], [emb_answer])[0][0]

        # Keyword extraction and match
        keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        keyword_list = [kw[0].lower() for kw in keywords]
        keyword_hits = sum(1 for word in keyword_list if word in answer.lower())

        keyword_score = keyword_hits / max(len(keyword_list), 1)

        # Combine scores
        final_score = (similarity + keyword_score) / 2
        return round(final_score * 100, 2)

    except Exception as e:
        print("Error in evaluate_answer():", e)
        return 0.0  # Fallback score







@app.route('/score', methods=['POST'])
def score():
    data = request.get_json()
    questions = data.get("questions", [])
    answers = data.get("answers", [])

    if len(questions) != len(answers):
        return jsonify({"error": "Mismatched questions and answers."}), 400

    total_score = 0
    qualified_count = 0
    partially_qualified_count = 0
    not_qualified_count = 0
    detailed_scores = []

    for q, a in zip(questions, answers):
        result = score_answer_combined(q, a)
        total_score += result["score"]

        if result["qualification_status"] == "Qualified":
            qualified_count += 1
        elif result["qualification_status"] == "Partially Qualified":
            partially_qualified_count += 1
        else:
            not_qualified_count += 1

        detailed_scores.append({
            "question": q,
            "answer": a,
            **result
        })

    average_score = round(total_score / len(questions), 2)
    summary = {
        "average_score": average_score,
        "Qualified": qualified_count,
        "Partially Qualified": partially_qualified_count,
        "Not Qualified": not_qualified_count,
        "details": detailed_scores
    }

    return jsonify(summary)


@app.route("/score-all", methods=["POST"])
@limiter.limit("5/minute")
def score_all_route():
    try:
        data = request.get_json()
        qa_pairs = data.get("qa_pairs", [])
        results = []
        total_score = 0

        for pair in qa_pairs:
            q = pair.get("question", "")
            a = pair.get("answer", "")
            scored = score_answer_accurately(q, a)
            results.append(scored)
            total_score += scored.get("final_score", 0)

        avg_score = round(total_score / len(results), 2) if results else 0
        status = "Qualified" if avg_score >= 0.7 else "Partially Qualified" if avg_score >= 0.5 else "Not Qualified"

        return jsonify({
            "average_score": avg_score,
            "qualification_status": status,
            "results": results
        }), 200

    except Exception as e:
        logger.error(f"❌ Error in /score-all route: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


from flask import Flask, request, jsonify

@app.route('/chat')
def chatbot():
    if 'user_id' not in session:
        flash("Please submit your application first.", "error")
        return redirect(url_for('dashboard'))

    user_id = session['user_id']
    name = session.get('name')
    experience = session.get('experience')
    position = session.get('position')

    return render_template('chatbot.html', name=name, experience=experience, position=position, user_id=user_id)


@app.route('/save_chatbot_data', methods=['POST'])
def save_chatbot_data_route():
    data = request.json
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({"error": "User ID not found in session"}), 400

    # Extract data
    user_name = data.get('user_name')
    position = data.get('position')
    experience = data.get('experience')
    skills = data.get('skills')
    qualification_status = data.get('qualification_status')
    advice = data.get('advice')
    assessment_data = json.dumps(data.get('assessment_data', {}))
    confidence = data.get('confidence')
    average_score = data.get('average_score')

    # Call shared saving function
    save_chatbot_data(
        user_id, user_name, position, experience, skills,
        qualification_status, advice, assessment_data,
        confidence, average_score
    )

    return jsonify({"message": "Chatbot data saved successfully"})


def save_chatbot_data(user_id, user_name, position, experience, skills,
                      qualification_status, advice, assessment_data,
                      confidence, average_score):
    try:
        cur = mysql.connection.cursor()
        sql = """
            INSERT INTO chatbot_results 
            (user_id, user_name, position, experience, skills, qualification_status, advice, assessment_data, confidence, average_score, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
        cur.execute(sql, (user_id, user_name, position, experience, skills,
                          qualification_status, advice, assessment_data,
                          confidence, average_score))
        mysql.connection.commit()
        cur.close()
        logger.info(f"Saved chatbot data for user_id {user_id}")
    except Exception as e:
        logger.error(f"Error saving chatbot data: {e}")
        mysql.connection.rollback()
        raise


@app.route('/insert_chatbot_data', methods=['POST'])
def insert_chatbot_data():
    user_id = session.get('user_id')  # Get from session, not from form

    if not user_id:
        return "User ID not found in session. Please login or start application.", 400

    user_name = request.form.get('user_name')
    position = request.form.get('position')
    experience = request.form.get('experience')
    skills = request.form.get('skills')
    qualification_status = request.form.get('qualification_status')

    advice_raw = request.form.get('advice')
    assessment_data_raw = request.form.get('assessment_data')
    confidence = request.form.get('confidence')
    average_score = request.form.get('average_score')

    # Parse advice and assessment safely
    try:
        advice = json.loads(advice_raw) if advice_raw else []
    except json.JSONDecodeError:
        advice = []

    try:
        assessment_data = json.loads(assessment_data_raw) if assessment_data_raw else []
    except json.JSONDecodeError:
        assessment_data = []

    new_id = save_chatbot_data(
        user_id, user_name, position, experience, skills,
        qualification_status, advice, assessment_data,
        confidence, average_score
    )

    if new_id:
        return redirect(url_for('summary'))
    else:
        return "Failed to save data", 500































@app.route('/summary/<int:user_id>')
def summary_with_id(user_id):
    with mysql.connection.cursor() as cur:
        cur.execute("SELECT user_name, position FROM applicants WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        if not row:
            return "Applicant not found", 404
        user_name, position = row

    return render_template('summary.html', user_id=user_id, user_name=user_name, position=position)


@app.route('/summary')
def summary():
    try:
        return render_template('summary.html')
    except Exception as e:
        print("⚠️ Error rendering summary.html:")
        traceback.print_exc()
        return f"<h1>Error rendering summary:</h1><pre>{str(e)}</pre>", 500




import json

def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        return []


import json

def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        return []


@app.route("/generate_summary", methods=["POST"])
def generate_summary():
    user_id = request.form.get("user_id")
    answers_json = request.form.get("answers")

    if not user_id or not answers_json:
        return "Missing user_id or answers", 400

    try:
        user_answers = json.loads(answers_json)
    except Exception as e:
        return f"Invalid answers data: {str(e)}", 400

    with mysql.connection.cursor() as cur:
        cur.execute("SELECT name, position, yearexperience FROM applicants WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        if not row:
            return "Applicant not found", 404
        columns = [desc[0] for desc in cur.description]
        applicant = dict(zip(columns, row))

    scores_data = score_all_answers(user_answers)
    avg_score = scores_data["average_score"]
    qualification_status = scores_data["qualification_status"]

    resume_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{user_id}.pdf")
    if os.path.exists(resume_path):
        top_skills, suggested_role = extract_skills_and_suggest_role(resume_path)
    else:
        top_skills, suggested_role = "", applicant["position"]

    session["user_id"] = user_id
    session["name"] = applicant["name"]
    session["role"] = suggested_role
    session["skills"] = top_skills
    session["qualification_status"] = qualification_status
    session["summary"] = [res["explanation"] for res in scores_data["answers"]]
    session["overall_qualification"] = qualification_status

    return render_template("summary.html",
        name=applicant["name"],
        role=suggested_role,
        skills_str=top_skills,
        qualification_status=qualification_status,
        advice_list=[res["explanation"] for res in scores_data["answers"]],
        assessment_data=user_answers,
        confidence=avg_score,
        average_score=avg_score,
        score_class=determine_score_class(avg_score)
    )

@app.route("/save_summary_report", methods=["POST"])
@limiter.limit("10/minute")
def save_summary_report():
    try:
        data = request.get_json(force=True)
        
        # Get user_id from session
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not logged in or session expired."}), 403
        
        # (rest of your existing code)
        
        user_name = data.get("user_name")
        position = data.get("position")
        experience = data.get("experience", "")
        skills = data.get("skills", [])
        qualification_status = data.get("qualification_status", "")
        advice = data.get("advice", [])
        assessment_data = data.get("assessment_data", [])
        confidence = float(data.get("confidence", 0))
        average_score = float(data.get("average_score", 0))

        if not user_name or not position:
            return jsonify({"error": "Missing required fields: user_name and position"}), 400

        # Save to DB
        conn = mysql.connection
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO chatbot (
                user_id, user_name, position, experience, skills,
                qualification_status, advice, assessment_data,
                confidence, average_score, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """

        cursor.execute(insert_query, (
            user_id,
            user_name,
            position,
            experience,
            json.dumps(skills),
            qualification_status,
            json.dumps(advice),
            json.dumps(assessment_data),
            confidence,
            average_score
        ))
        conn.commit()
        cursor.close()

        return jsonify({
            "message": "Summary report saved successfully.",
            "redirect": url_for('summary')
        }), 201

    except Exception as e:
        logger.error(f"❌ Error in /save_summary_report: {e}")
        return jsonify({
            "error": "Failed to save interview summary.",
            "details": str(e)
        }), 500


def parse_confidence(value):
    try:
        if isinstance(value, str) and '%' in value:
            value = value.replace('%', '')
        return float(value)
    except (ValueError, TypeError):
        return 0.0


@app.route('/summary_report')
def summary_report():
    user_id = session.get('user_id')
    if not user_id:
        return "User session not found.", 400

    with mysql.connection.cursor() as cursor:
        cursor.execute("SELECT * FROM chatbot WHERE user_id = %s ORDER BY id DESC LIMIT 1", (user_id,))
        row = cursor.fetchone()

        if not row:
            return "No summary found for this user.", 404

        keys = [desc[0] for desc in cursor.description]
        data = dict(zip(keys, row))

        assessment_data = json.loads(data.get("assessment_data") or "[]")
        advice_list = json.loads(data.get("advice") or "[]")

        return render_template("summary.html", 
            name=data.get("user_name"),
            position=data.get("position"),
            skills=data.get("skills"),
            qualification_status=data.get("qualification_status"),
            confidence=data.get("confidence"),
            assessment_data=assessment_data,
            advice_list=advice_list
        )


@app.route('/finalize_interview', methods=['POST'])
def finalize_interview():
    try:
        # Replace with actual method to get the score data
        score_data = get_score_data_somehow()

        if isinstance(score_data, dict):
            qualification_status = score_data.get("qualification_status")
        else:
            # If score_data is just an int or something else
            qualification_status = "Qualified" if score_data >= 0.75 else "Not Qualified"

        # Example: save qualification_status or other logic here

        return jsonify({"status": "success", "qualification_status": qualification_status})

    except Exception as e:
        app.logger.error(f"Error in finalize_interview: {e}")
        return jsonify({"error": str(e)}), 500



import textwrap

import textwrap

def generate_summary(name, position, experience, answers, qualification_status, confidence, reason):
    answer_str = ''.join(f"- {ans}\n" for ans in answers)

    summary = f"""
Interview Summary Report

Name: {user_name}
Position Applied: {position}
Experience: {experience} years

Answers Provided:
{answer_str}
Qualification Status: {qualification_status}
Confidence Level: {confidence}
Reason for Qualification Status: {reason}
"""
    return textwrap.dedent(summary).strip()

def generate_pdf_summary(user_name, position, qualifications, status):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_name}_{position}_{timestamp}.pdf"
    filepath = os.path.join("summary_reports", filename)
    os.makedirs('summary_reports', exist_ok=True)

    # Use fallback if qualifications is None
    qualifications = qualifications or "No qualifications provided"

    c = canvas.Canvas(filepath, pagesize=letter)
    text = c.beginText(50, 750)
    text.setFont("Helvetica", 12)
    text.textLine(f"Candidate Name: {user_name}")
    text.textLine(f"Position: {position}")
    text.textLine(f"Qualification Status: {status}")
    text.textLine("Top Skills / Remarks:")
    for line in qualifications.split(','):
        text.textLine(f"- {line.strip()}")

    c.drawText(text)
    c.showPage()
    c.save()

    return filepath

@app.route('/download_summary')
def download_summary():
    user_name = session.get('name', 'Candidate')
    position = session.get('role', 'Unknown')
    qualifications = session.get('skills', [])
    qualifications_str = ', '.join(qualifications) if isinstance(qualifications, list) else qualifications
    status = session.get('qualification_status', 'Not Qualified')

    filepath = generate_pdf_summary(user_name, position, qualifications_str, status)
    
    return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))


@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()

    user_name = data.get('user_name', 'Candidate')
    role = data.get('role', 'Unknown')
    skills = data.get('skills', '')
    qualification_status = data.get('qualification_status', 'Not Qualified')
    advice_list = data.get('advice_list', [])
    assessment_data = data.get('assessment_data', [])

    rendered = render_template('summary.html',
                               user_name=user_name,
                               role=role,
                               skills_str=skills,
                               qualification_status=qualification_status,
                               advice_list=advice_list,
                               assessment_data=assessment_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'summary_reports/{name}_summary_{timestamp}.pdf'
    os.makedirs('summary_reports', exist_ok=True)

    config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
    pdfkit.from_string(rendered, filename, configuration=config)

    with open(filename, 'rb') as f:
        pdf_data = f.read()

    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={os.path.basename(filename)}'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)