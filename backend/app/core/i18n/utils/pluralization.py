"""
Module: pluralization.py
Description: Gestion avancée des règles de pluriel pour toutes les langues supportées, avec API clé en main pour l'analytics, les notifications, et la génération de contenu IA.
"""
from babel.plural import PluralRule
from babel import Locale
from typing import Dict, Any

# Règles de pluriel pré-compilées pour performance
_PLURAL_RULES: Dict[str, PluralRule] = {}

SUPPORTED_LOCALES = [
    "en", "fr", "de", "es", "it", "pt", "ru", "ja", "ar", "zh_CN", # ...
]

def get_plural_rule(locale: str) -> PluralRule:
    if locale not in _PLURAL_RULES:
        _PLURAL_RULES[locale] = PluralRule(Locale.parse(locale).plural_form)
    return _PLURAL_RULES[locale]

def pluralize(count: int, forms: Dict[str, str], locale: str = "en") -> str:
    """
    Retourne la forme correcte selon la locale et le nombre (ex: {"one": "track", "other": "tracks"}).
    """
    rule = get_plural_rule(locale)
    form = rule(count)
    return forms.get(form, forms.get("other", ""))

# Exemples d'utilisation
# forms = {"one": "artiste", "other": "artistes"}
# print(pluralize(2, forms, locale="fr")
