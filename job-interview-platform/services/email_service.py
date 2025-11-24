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
    msg.body = f"Your OTP is {otp}. Use this OTP to reset your password. It is valid for 5 minutes."
    try:
        mail.send(msg)
        logger.info(f"📧 OTP sent to {email}")
        return True
    except Exception as e:
        logger.error(f"❌ Error sending OTP email: {e}")
        return False
