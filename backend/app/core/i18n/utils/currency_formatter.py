"""
Module: currency_formatter.py
Description: Fonctions avancées pour le formatage, la conversion et la validation des devises, avec support multi-locale et multi-devise.
"""
import babel.numbers
from typing import Optional

class CurrencyFormatter:
    @staticmethod
    def format_currency(value: float, currency: str = "EUR", locale: str = "en") -> str:
        """
        Formate une valeur monétaire selon la devise et la locale.
        """
        return babel.numbers.format_currency(value, currency, locale=locale)

    @staticmethod
    def parse_currency(currency_str: str, locale: str = "en") -> float:
        """
        Parse une chaîne monétaire localisée en float.
        """
        return babel.numbers.parse_decimal(currency_str, locale=locale)

    @staticmethod
    def convert_currency(value: float, from_currency: str, to_currency: str, rate: float) -> float:
        """
        Convertit une valeur d'une devise à une autre selon un taux de change fourni.
        """
        return value * rate

# Exemples d'utilisation
# print(CurrencyFormatter.format_currency(123.45, currency="USD", locale="en")
# print(CurrencyFormatter.convert_currency(100, "EUR", "USD", 1.08)
