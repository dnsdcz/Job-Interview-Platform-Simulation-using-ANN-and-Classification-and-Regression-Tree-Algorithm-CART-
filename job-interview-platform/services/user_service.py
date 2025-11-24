# services/user_service.py
from typing import Optional, Tuple
from flask import current_app
from extensions import mysql


def get_user_by_email(email: str) -> Optional[Tuple]:
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, email, username, contact_number, usertype, profile_photo FROM users WHERE email = %s",
                (email,))
    user = cur.fetchone()
    cur.close()
    return user


def get_user_by_id(user_id: int) -> Optional[Tuple]:
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, email, username, contact_number, usertype, profile_photo FROM users WHERE id = %s",
                (user_id,))
    user = cur.fetchone()
    cur.close()
    return user
