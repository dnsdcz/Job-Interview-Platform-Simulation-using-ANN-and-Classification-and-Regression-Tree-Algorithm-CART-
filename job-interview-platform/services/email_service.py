# services/email_service.py
from flask_mail import Message
from flask import current_app
from extensions import mail, logger


def send_otp_email(email: str, otp: int) -> bool:
    msg = Message(
        subject="Your OTP for Password Reset",
        sender=current_app.config["MAIL_USERNAME"],
        recipients=[email],
    )
    msg.body = (
        f"Your OTP is {otp}. Use this OTP to reset your password. "
        "It is valid for 5 minutes."
    )
    try:
        mail.send(msg)
        logger.info(f"📧 OTP sent to {email}")
        return True
    except Exception as e:
        logger.error(f"❌ Error sending OTP email: {e}")
        return False


def send_step1_completed_email(email: str, name: str, position: str) -> bool:
    """
    Sends an email telling the applicant they have finished Step 1
    (Eligible after initial screening / HR approval).
    """
    subject = "You’ve completed Step 1 of your application"
    sender = current_app.config.get("MAIL_USERNAME")

    body = (
        f"Hi {name},\n\n"
        f"Good news! Your application for the position of {position} "
        f"has successfully completed Step 1 of the hiring process.\n\n"
        "You are now marked as Eligible for the next stage.\n"
        "Please log in to your account to view the next steps.\n\n"
        "Best regards,\n"
        "HR Team"
    )

    msg = Message(
        subject=subject,
        sender=sender,
        recipients=[email],
    )
    msg.body = body

    try:
        mail.send(msg)
        logger.info(f"📧 Step 1 completed email sent to {email}")
        return True
    except Exception as e:
        logger.error(f"❌ Error sending Step 1 completed email: {e}")
        return False
