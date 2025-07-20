"""
Module: validators.py
Description: Validateurs industriels (email, url, phone, IBAN, custom, FastAPI/Pydantic-ready).
"""
import re
from typing import Optional

def is_email(value: str) -> bool:
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", value))

def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://[\w\.-]+", value))

def is_phone(value: str) -> bool:
    return bool(re.match(r"^\+?\d{7,15}$", value))

def is_iban(value: str) -> bool:
    return bool(re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$", value))

# Exemples d'utilisation
# is_email("foo@bar.com")
# is_url("https://foo.com")
# is_phone("+33612345678")
# is_iban("FR7612345678901234567890123")
