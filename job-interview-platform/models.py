# models.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    id: int
    email: str
    username: Optional[str]
    contact_number: Optional[str]
    usertype: str
    profile_photo: Optional[bytes] = None


@dataclass
class Applicant:
    id: int
    user_id: int
    name: str
    email: str
    contact: str
    position: str
    eligibility: str
    yearexperience: int
    level: str
    status: str
    qualified: str
    confidence: float
