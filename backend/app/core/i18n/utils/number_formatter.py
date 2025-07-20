"""
Module: number_formatter.py
Description: Fournit des fonctions avancées pour le formatage, la localisation et la validation des nombres, pour tous les cas d'usage métier (statistiques, analytics, monétaires, etc).
"""
import babel.numbers
from typing import Optional

class NumberFormatter:
    @staticmethod
    def format_number(value: float, locale: str = "en") -> str:
        """
        Formate un nombre selon la locale (ex: séparateur de milliers, décimales).
        """
        return babel.numbers.format_decimal(value, locale=locale)

    @staticmethod
    def format_percent(value: float, locale: str = "en", decimals: int = 2) -> str:
        """
        Formate un pourcentage localisé.
        """
        return babel.numbers.format_percent(value, locale=locale, format=f"#,##0.{''.join(['0']*decimals)}%")

    @staticmethod
    def format_currency(value: float, currency: str = "EUR", locale: str = "en") -> str:
        """
        Formate une valeur monétaire localisée.
        """
        return babel.numbers.format_currency(value, currency, locale=locale)

    @staticmethod
    def parse_number(number_str: str, locale: str = "en") -> float:
        """
        Parse une chaîne localisée en float.
        """
        return babel.numbers.parse_decimal(number_str, locale=locale)

# Exemples d'utilisation
# print(NumberFormatter.format_number(1234567.89, locale="fr")
# print(NumberFormatter.format_currency(99.99, currency="USD", locale="en")
