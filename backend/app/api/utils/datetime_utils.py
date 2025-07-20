"""
🎵 Spotify AI Agent - DateTime Utilities
=======================================

Utilitaires enterprise pour la gestion avancée des dates et heures
avec support des fuseaux horaires et formatage intelligent.

Architecture:
- Parsing et formatage de dates
- Gestion des fuseaux horaires
- Calculs de durées et intervalles
- Humanisation des dates
- Validation temporelle
- Calendrier et horaires métier

🎖️ Développé par l'équipe d'experts enterprise
"""

import re
from datetime import datetime, date, time, timedelta, timezone
from typing import Optional, Union, List, Dict, Any
from zoneinfo import ZoneInfo
import calendar


# =============================================================================
# FORMATAGE ET PARSING
# =============================================================================

def format_datetime(dt: datetime, format_style: str = 'iso', 
                   timezone_display: bool = True, locale: str = 'en') -> str:
    """
    Formate une datetime selon différents styles
    
    Args:
        dt: DateTime à formater
        format_style: Style de formatage
        timezone_display: Afficher le fuseau horaire
        locale: Locale pour l'affichage
        
    Returns:
        Date formatée
    """
    if format_style == 'iso':
        return dt.isoformat()
    elif format_style == 'rfc3339':
        return dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    elif format_style == 'short':
        return dt.strftime('%Y-%m-%d %H:%M')
    elif format_style == 'long':
        return dt.strftime('%A, %B %d, %Y at %I:%M %p')
    elif format_style == 'date_only':
        return dt.strftime('%Y-%m-%d')
    elif format_style == 'time_only':
        return dt.strftime('%H:%M:%S')
    elif format_style == 'human':
        return humanize_datetime(dt)
    elif format_style == 'business':
        return dt.strftime('%d/%m/%Y %H:%M')
    else:
        return dt.strftime(format_style)


def parse_datetime(date_string: str, timezone_name: Optional[str] = None) -> Optional[datetime]:
    """
    Parse une chaîne de date avec détection automatique du format
    
    Args:
        date_string: Chaîne à parser
        timezone_name: Fuseau horaire à appliquer
        
    Returns:
        DateTime parsée ou None
    """
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%d %H:%M:%S%z',
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%m/%d/%Y',
        '%d/%m/%Y %H:%M',
        '%d-%m-%Y %H:%M',
        '%B %d, %Y',
        '%d %B %Y',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%fZ'
    ]
    
    # Nettoyer la chaîne
    date_string = date_string.strip()
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            
            # Ajouter le fuseau horaire si spécifié
            if timezone_name and dt.tzinfo is None:
                tz = ZoneInfo(timezone_name)
                dt = dt.replace(tzinfo=tz)
            
            return dt
        except ValueError:
            continue
    
    return None


def get_timezone_offset(timezone_name: str, dt: Optional[datetime] = None) -> timedelta:
    """
    Obtient le décalage d'un fuseau horaire par rapport à UTC
    
    Args:
        timezone_name: Nom du fuseau horaire
        dt: DateTime de référence (par défaut maintenant)
        
    Returns:
        Décalage sous forme de timedelta
    """
    if dt is None:
        dt = datetime.now()
    
    tz = ZoneInfo(timezone_name)
    dt_with_tz = dt.replace(tzinfo=tz)
    return dt_with_tz.utcoffset()


def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """
    Convertit une datetime d'un fuseau horaire à un autre
    
    Args:
        dt: DateTime à convertir
        from_tz: Fuseau horaire source
        to_tz: Fuseau horaire cible
        
    Returns:
        DateTime convertie
    """
    # Ajouter le fuseau horaire source si absent
    if dt.tzinfo is None:
        source_tz = ZoneInfo(from_tz)
        dt = dt.replace(tzinfo=source_tz)
    
    # Convertir vers le fuseau cible
    target_tz = ZoneInfo(to_tz)
    return dt.astimezone(target_tz)


# =============================================================================
# HUMANISATION
# =============================================================================

def humanize_datetime(dt: datetime, reference: Optional[datetime] = None, 
                     locale: str = 'en') -> str:
    """
    Humanise une datetime (ex: "il y a 2 heures")
    
    Args:
        dt: DateTime à humaniser
        reference: DateTime de référence (par défaut maintenant)
        locale: Locale pour l'affichage
        
    Returns:
        Chaîne humanisée
    """
    if reference is None:
        reference = datetime.now(dt.tzinfo if dt.tzinfo else timezone.utc)
    
    delta = reference - dt
    
    # Traductions selon la locale
    if locale == 'fr':
        translations = {
            'now': 'maintenant',
            'seconds_ago': 'il y a {} secondes',
            'minute_ago': 'il y a une minute',
            'minutes_ago': 'il y a {} minutes',
            'hour_ago': 'il y a une heure',
            'hours_ago': 'il y a {} heures',
            'day_ago': 'hier',
            'days_ago': 'il y a {} jours',
            'week_ago': 'il y a une semaine',
            'weeks_ago': 'il y a {} semaines',
            'month_ago': 'il y a un mois',
            'months_ago': 'il y a {} mois',
            'year_ago': 'il y a un an',
            'years_ago': 'il y a {} ans'
        }
    else:  # Anglais par défaut
        translations = {
            'now': 'now',
            'seconds_ago': '{} seconds ago',
            'minute_ago': 'a minute ago',
            'minutes_ago': '{} minutes ago',
            'hour_ago': 'an hour ago',
            'hours_ago': '{} hours ago',
            'day_ago': 'yesterday',
            'days_ago': '{} days ago',
            'week_ago': 'a week ago',
            'weeks_ago': '{} weeks ago',
            'month_ago': 'a month ago',
            'months_ago': '{} months ago',
            'year_ago': 'a year ago',
            'years_ago': '{} years ago'
        }
    
    if delta.total_seconds() < 60:
        if delta.total_seconds() < 5:
            return translations['now']
        return translations['seconds_ago'].format(int(delta.total_seconds()))
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() / 60)
        if minutes == 1:
            return translations['minute_ago']
        return translations['minutes_ago'].format(minutes)
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        if hours == 1:
            return translations['hour_ago']
        return translations['hours_ago'].format(hours)
    elif delta.days < 7:
        if delta.days == 1:
            return translations['day_ago']
        return translations['days_ago'].format(delta.days)
    elif delta.days < 30:
        weeks = delta.days // 7
        if weeks == 1:
            return translations['week_ago']
        return translations['weeks_ago'].format(weeks)
    elif delta.days < 365:
        months = delta.days // 30
        if months == 1:
            return translations['month_ago']
        return translations['months_ago'].format(months)
    else:
        years = delta.days // 365
        if years == 1:
            return translations['year_ago']
        return translations['years_ago'].format(years)


def calculate_duration(start: datetime, end: datetime, 
                      unit: str = 'auto', precision: int = 2) -> Union[str, float]:
    """
    Calcule la durée entre deux dates
    
    Args:
        start: Date de début
        end: Date de fin
        unit: Unité de retour (auto, seconds, minutes, hours, days)
        precision: Précision décimale
        
    Returns:
        Durée formatée ou numérique
    """
    delta = end - start
    total_seconds = delta.total_seconds()
    
    if unit == 'seconds':
        return round(total_seconds, precision)
    elif unit == 'minutes':
        return round(total_seconds / 60, precision)
    elif unit == 'hours':
        return round(total_seconds / 3600, precision)
    elif unit == 'days':
        return round(total_seconds / 86400, precision)
    else:  # auto
        if total_seconds < 60:
            return f"{round(total_seconds, precision)} seconds"
        elif total_seconds < 3600:
            return f"{round(total_seconds / 60, precision)} minutes"
        elif total_seconds < 86400:
            return f"{round(total_seconds / 3600, precision)} hours"
        else:
            return f"{round(total_seconds / 86400, precision)} days"


# =============================================================================
# VALIDATION ET VÉRIFICATION
# =============================================================================

def is_business_day(dt: date, country: str = 'US') -> bool:
    """
    Vérifie si une date est un jour ouvrable
    
    Args:
        dt: Date à vérifier
        country: Pays pour les jours fériés
        
    Returns:
        True si jour ouvrable
    """
    # Vérifier le jour de la semaine (0=lundi, 6=dimanche)
    if dt.weekday() >= 5:  # Samedi ou dimanche
        return False
    
    # Vérifier les jours fériés (implémentation basique pour les US)
    if country == 'US':
        holidays = get_us_holidays(dt.year)
        return dt not in holidays
    
    return True


def get_us_holidays(year: int) -> List[date]:
    """
    Retourne la liste des jours fériés américains pour une année
    
    Args:
        year: Année
        
    Returns:
        Liste des jours fériés
    """
    holidays = []
    
    # Jour de l'an
    holidays.append(date(year, 1, 1))
    
    # Martin Luther King Day (3e lundi de janvier)
    jan_first = date(year, 1, 1)
    days_to_first_monday = (7 - jan_first.weekday()) % 7
    first_monday = jan_first + timedelta(days=days_to_first_monday)
    mlk_day = first_monday + timedelta(days=14)
    holidays.append(mlk_day)
    
    # Independence Day
    holidays.append(date(year, 7, 4))
    
    # Christmas Day
    holidays.append(date(year, 12, 25))
    
    return holidays


def is_valid_date_range(start: datetime, end: datetime, 
                       max_range: Optional[timedelta] = None) -> bool:
    """
    Valide une plage de dates
    
    Args:
        start: Date de début
        end: Date de fin
        max_range: Plage maximale autorisée
        
    Returns:
        True si valide
    """
    if start >= end:
        return False
    
    if max_range and (end - start) > max_range:
        return False
    
    return True


def is_future_date(dt: datetime, reference: Optional[datetime] = None) -> bool:
    """
    Vérifie si une date est dans le futur
    
    Args:
        dt: Date à vérifier
        reference: Date de référence
        
    Returns:
        True si future
    """
    if reference is None:
        reference = datetime.now(dt.tzinfo if dt.tzinfo else timezone.utc)
    
    return dt > reference


# =============================================================================
# CALENDRIER ET PÉRIODES
# =============================================================================

def get_week_boundaries(dt: datetime) -> tuple[datetime, datetime]:
    """
    Obtient le début et la fin de la semaine pour une date
    
    Args:
        dt: Date de référence
        
    Returns:
        Tuple (début_semaine, fin_semaine)
    """
    start_of_week = dt - timedelta(days=dt.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    
    end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    return start_of_week, end_of_week


def get_month_boundaries(dt: datetime) -> tuple[datetime, datetime]:
    """
    Obtient le début et la fin du mois pour une date
    
    Args:
        dt: Date de référence
        
    Returns:
        Tuple (début_mois, fin_mois)
    """
    start_of_month = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Dernier jour du mois
    _, last_day = calendar.monthrange(dt.year, dt.month)
    end_of_month = dt.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)
    
    return start_of_month, end_of_month


def get_quarter_boundaries(dt: datetime) -> tuple[datetime, datetime]:
    """
    Obtient le début et la fin du trimestre pour une date
    
    Args:
        dt: Date de référence
        
    Returns:
        Tuple (début_trimestre, fin_trimestre)
    """
    quarter = (dt.month - 1) // 3 + 1
    start_month = (quarter - 1) * 3 + 1
    
    start_of_quarter = dt.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    end_month = start_month + 2
    _, last_day = calendar.monthrange(dt.year, end_month)
    end_of_quarter = dt.replace(month=end_month, day=last_day, hour=23, minute=59, second=59, microsecond=999999)
    
    return start_of_quarter, end_of_quarter


def generate_date_range(start: date, end: date, 
                       step: timedelta = timedelta(days=1)) -> List[date]:
    """
    Génère une plage de dates
    
    Args:
        start: Date de début
        end: Date de fin
        step: Pas d'incrémentation
        
    Returns:
        Liste des dates
    """
    dates = []
    current = start
    
    while current <= end:
        dates.append(current)
        current += step
    
    return dates


# =============================================================================
# HORAIRES MÉTIER
# =============================================================================

def is_within_business_hours(dt: datetime, 
                           start_hour: int = 9, end_hour: int = 17,
                           timezone_name: str = 'UTC') -> bool:
    """
    Vérifie si une heure est dans les horaires d'ouverture
    
    Args:
        dt: DateTime à vérifier
        start_hour: Heure d'ouverture
        end_hour: Heure de fermeture
        timezone_name: Fuseau horaire
        
    Returns:
        True si dans les horaires
    """
    # Convertir vers le fuseau horaire métier
    if dt.tzinfo is None:
        tz = ZoneInfo(timezone_name)
        dt = dt.replace(tzinfo=tz)
    else:
        tz = ZoneInfo(timezone_name)
        dt = dt.astimezone(tz)
    
    # Vérifier l'heure
    return start_hour <= dt.hour < end_hour


def next_business_day(dt: date, country: str = 'US') -> date:
    """
    Trouve le prochain jour ouvrable
    
    Args:
        dt: Date de référence
        country: Pays pour les jours fériés
        
    Returns:
        Prochain jour ouvrable
    """
    next_day = dt + timedelta(days=1)
    
    while not is_business_day(next_day, country):
        next_day += timedelta(days=1)
    
    return next_day


def add_business_days(dt: date, days: int, country: str = 'US') -> date:
    """
    Ajoute des jours ouvrables à une date
    
    Args:
        dt: Date de référence
        days: Nombre de jours ouvrables à ajouter
        country: Pays pour les jours fériés
        
    Returns:
        Date résultante
    """
    current = dt
    added_days = 0
    
    while added_days < days:
        current = next_business_day(current, country)
        added_days += 1
    
    return current


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "format_datetime",
    "parse_datetime",
    "get_timezone_offset",
    "convert_timezone",
    "humanize_datetime",
    "calculate_duration",
    "is_business_day",
    "get_us_holidays",
    "is_valid_date_range",
    "is_future_date",
    "get_week_boundaries",
    "get_month_boundaries",
    "get_quarter_boundaries",
    "generate_date_range",
    "is_within_business_hours",
    "next_business_day",
    "add_business_days"
]
