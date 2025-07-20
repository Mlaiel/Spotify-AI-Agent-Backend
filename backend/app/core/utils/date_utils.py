"""
Module: date_utils.py
Description: Utilitaires industriels pour la gestion des dates (parsing, formatage, diff, humanize, timezone, ISO8601).
"""
import datetime
import pytz
from dateutil import parser
from babel.dates import format_datetime, format_timedelta
from typing import Optional

def parse_date(date_str: str) -> datetime.datetime:
    return parser.parse(date_str)

def format_date(dt: datetime.datetime, locale: str = "en", tz: Optional[str] = None, fmt: str = "medium") -> str:
    if tz:
        dt = dt.astimezone(pytz.timezone(tz))
    return format_datetime(dt, locale=locale, format=fmt)

def humanize_delta(delta: datetime.timedelta, locale: str = "en") -> str:
    return format_timedelta(delta, locale=locale)

def now_utc() -> datetime.datetime:
    # Python 3.9+: datetime.now(datetime.timezone.utc) ist der empfohlene Weg
    return datetime.datetime.now(datetime.timezone.utc)

# Exemples d'utilisation
# parse_date("2025-07-10T12:00:00Z")
# format_date(now_utc(), locale="fr", tz="Europe/Paris")
# humanize_delta(datetime.timedelta(hours=2), locale="de")
