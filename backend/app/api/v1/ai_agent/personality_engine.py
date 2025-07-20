"""
PersonalityEngine : Moteur de gestion de la personnalité IA
- Profils dynamiques, adaptation au style utilisateur
- Personnalités prêtes à l’emploi (coach, analyste, créatif, etc.)
- Sécurité : validation, audit, conformité
- Intégration microservices, scalable

Auteur : Architecte IA, ML, Backend
"""

from typing import Dict, Any

class PersonalityEngine:
    """
    Gère la personnalité de l’agent IA selon l’utilisateur, le contexte et la stratégie business.
    """
    DEFAULT_PERSONAS = {
        "default": {"tone": "neutre", "style": "informatif", "lang": "fr"},
        "coach": {"tone": "motivant", "style": "coaching", "lang": "fr"},
        "analyste": {"tone": "analytique", "style": "données", "lang": "fr"},
        "créatif": {"tone": "créatif", "style": "inspirant", "lang": "fr"}
    }

    def get_persona(self, user_id: str) -> Dict[str, Any]:
        """
        Retourne le profil de personnalité à utiliser pour l’utilisateur.
        (En prod : récupérer depuis DB ou préférences utilisateur)
        """
        # Logique avancée possible ici (analyse historique, préférences, AB testing...)
        if user_id.endswith("coach"):
            return self.DEFAULT_PERSONAS["coach"]
        if user_id.endswith("analyste"):
            return self.DEFAULT_PERSONAS["analyste"]
        if user_id.endswith("creatif"):
            return self.DEFAULT_PERSONAS["créatif"]
        return self.DEFAULT_PERSONAS["default"]

# Exemple d’utilisation :
# engine = PersonalityEngine()
# print(engine.get_persona("user123coach")
