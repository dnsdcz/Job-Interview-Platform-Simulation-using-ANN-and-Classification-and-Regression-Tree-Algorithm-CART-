# config.py
import os


class Config:
    # core
    SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")

    # MySQL
    MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
    MYSQL_USER = os.environ.get("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
    MYSQL_DB = os.environ.get("MYSQL_DB", "auth_db")

    # uploads
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    PROFILE_UPLOAD_FOLDER = os.environ.get(
        "PROFILE_UPLOAD_FOLDER", "static/uploads")
    ALLOWED_RESUME_EXTENSIONS = {"pdf"}          # kept for possible future
    ALLOWED_PROFILE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

    # mail
    MAIL_SERVER = "smtp.gmail.com"
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "aceview18@gmail.com")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "uelmqlulrxbbkikx")

    # limiter
    RATELIMIT_DEFAULT = "10 per minute"

    # pdf / summary
    SUMMARY_REPORT_DIR = os.environ.get(
        "SUMMARY_REPORT_DIR", "summary_reports")
