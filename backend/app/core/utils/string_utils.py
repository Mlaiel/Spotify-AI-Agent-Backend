"""
Module: string_utils.py
Description: Utilitaires industriels pour la manipulation de chaînes (slugify, random, clean, truncate, normalize, case).
"""
import re
import unicodedata
import secrets
import string

def slugify(value: str) -> str:
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

def random_string(length: int = 12) -> str:
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))

def clean_string(value: str) -> str:
    return re.sub(r'\s+', ' ', value).strip()

def truncate(value: str, length: int = 100) -> str:
    return value[:length] + ('...' if len(value) > length else '')

# Exemples d'utilisation
# slugify("Hello World!")
# random_string(8)
# clean_string("  foo   bar ")
# truncate("Lorem ipsum...", 10)
