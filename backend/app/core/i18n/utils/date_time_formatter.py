"""
Module: date_time_formatter.py
Description: Fournit des fonctions avancées pour la gestion, la localisation et le formatage des dates et heures dans un contexte multi-langue et multi-région.
"""
import datetime
import pytz
import babel.dates
from typing import Optional

class DateTimeFormatter:
    """
    Classe utilitaire pour le formatage avancé des dates et heures avec prise en charge de la localisation, des fuseaux horaires et des formats personnalisés.
    """
    @staticmethod
    def format_datetime(dt: datetime.datetime, locale: str = "en", tz: Optional[str] = None, format: str = "medium") -> str:
        """
        Formate un objet datetime selon la locale, le fuseau horaire et le format spécifiés.
        """
        if tz:
            tzinfo = pytz.timezone(tz)
            dt = dt.astimezone(tzinfo)
        return babel.dates.format_datetime(dt, locale=locale, format=format)

    @staticmethod
    def parse_datetime(date_str: str, locale: str = "en", format: str = "medium") -> datetime.datetime:
        """
        Parse une chaîne de date/heure localisée en objet datetime.
        """
        # Pour une robustesse industrielle, utiliser dateutil.parser ou Babel selon le format attendu
        from dateutil import parser
        return parser.parse(date_str)

    @staticmethod
    def get_timezone_offset(tz: str) -> int:
        """
        Retourne l'offset en minutes pour un fuseau horaire donné.
        """
        tzinfo = pytz.timezone(tz)
        now = datetime.datetime.now(tzinfo)
        return int(now.utcoffset().total_seconds() / 60)

    @staticmethod
    def humanize_timedelta(delta: datetime.timedelta, locale: str = "en") -> str:
        """
        Retourne une représentation humaine d'un timedelta (ex: "il y a 2 heures").
        """
        from babel.dates import format_timedelta
        return format_timedelta(delta, locale=locale)

# Exemples d'utilisation
# dt = DateTimeFormatter.format_datetime(datetime.datetime.now(), locale="fr", tz="Europe/Paris", format="full")
# print(dt)
