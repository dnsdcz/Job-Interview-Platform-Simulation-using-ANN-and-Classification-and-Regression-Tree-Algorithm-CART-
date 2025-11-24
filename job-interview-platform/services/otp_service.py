# services/otp_service.py
import random
import time


# simple in-memory OTP store
otp_store = {}  # {email: {"otp": 123456, "timestamp": 1234567890}}


def generate_otp(email: str) -> int:
    otp = random.randint(100000, 999999)
    otp_store[email] = {"otp": otp, "timestamp": time.time()}
    return otp


def verify_otp(email: str, otp_input: str, ttl_seconds: int = 300) -> bool:
    data = otp_store.get(email)
    if not data:
        return False
    try:
        if int(otp_input) != data["otp"]:
            return False
    except ValueError:
        return False

    if time.time() - data["timestamp"] > ttl_seconds:
        return False

    # one-time use
    del otp_store[email]
    return True
