# blueprints/__init__.py
from flask import Blueprint

from .auth import auth_bp
from .applicants import applicants_bp
from .hr import hr_bp
from .interview import interview_bp
from .schedule import schedule_bp
from .summary import summary_bp


def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(applicants_bp)
    app.register_blueprint(hr_bp)
    app.register_blueprint(interview_bp)
    app.register_blueprint(schedule_bp)
    app.register_blueprint(summary_bp)
