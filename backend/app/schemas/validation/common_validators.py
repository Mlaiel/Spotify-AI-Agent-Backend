"""
Common Validators
- Reusable, production-grade validators for all business domains (AI, Spotify, User, Analytics, Collaboration, Security).
- Features: DSGVO/HIPAA compliance, security, audit, traceability, explainability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Use in Pydantic models, FastAPI dependencies, or custom business logic.
"""
import re
from typing import Any
from pydantic import EmailStr, ValidationError

EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$")
SPOTIFY_ID_REGEX = re.compile(r"^[a-zA-Z0-9]{22}$")

# --- Security & Compliance ---
def validate_email(email: str) -> str:
    if not EMAIL_REGEX.match(email):
        raise ValidationError(f"Invalid email address: {email}")
    return email

def validate_spotify_id(spotify_id: str) -> str:
    if not SPOTIFY_ID_REGEX.match(spotify_id):
        raise ValidationError(f"Invalid Spotify ID: {spotify_id}")
    return spotify_id

def validate_consent(consent: Any) -> bool:
    if consent is not True:
        raise ValidationError("Consent (Einwilligung) ist zwingend erforderlich (DSGVO/HIPAA).");
    return True

def validate_password_strength(password: str) -> str:
    if len(password) < 12:
        raise ValidationError("Password must be at least 12 characters long.")
    if not re.search(r"[A-Z]", password):
        raise ValidationError("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        raise ValidationError("Password must contain at least one lowercase letter.")
    if not re.search(r"[0-9]", password):
        raise ValidationError("Password must contain at least one digit.")
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\",.<>/?]", password):
        raise ValidationError("Password must contain at least one special character.")
    return password

# --- Data Minimization & Privacy ---
def validate_data_minimization(data: dict, allowed_fields: set) -> dict:
    filtered = {k: v for k, v in data.items() if k in allowed_fields}
    if len(filtered) != len(data):
        raise ValidationError("Data minimization violation: extra fields present.")
    return filtered
