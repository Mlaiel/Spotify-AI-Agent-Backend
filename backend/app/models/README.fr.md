# Documentation Modèles Métier (FR)

**Spotify AI Agent – ORM Entreprise pour les modèles métier**

## Objectif
Ce package fournit tous les modèles métier avancés et prêts pour la production :
- Contenu IA, Analytics, Collaboration, Données Spotify, Utilisateur
- Classes de base, gouvernance, sécurité, conformité, data lineage, multi-tenancy

Tous les sous-modules (orm, ai_content, analytics, collaboration, spotify_data, user) sont optimisés pour PostgreSQL, MongoDB et architectures hybrides.

## Bonnes pratiques
- Tous les modèles héritent des classes de base ORM et utilisent les mixins pertinents
- Sécurité, audit et conformité sont appliqués à tous les niveaux
- L’utilisation est loggée pour audit et traçabilité

## Exemple d’utilisation
```python
from .ai_content import AIContent
content = AIContent.create(user_id=1, content_type="lyrics", content="Bonjour le monde")
```

## Gouvernance & Extension
- Toute modification doit respecter la convention de nommage/version et inclure des docstrings
- Sécurité, audit et conformité sont appliqués à tous les niveaux

---
*Cette documentation est générée et maintenue automatiquement dans le CI/CD. Dernière mise à jour : juillet 2025.*

