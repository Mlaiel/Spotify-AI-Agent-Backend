# Documentation ORM Utilisateur (FR)

**Spotify AI Agent – ORM Entreprise pour les données utilisateur**

## Objectif
Ce module fournit tous les modèles ORM avancés et prêts pour la production pour les données utilisateur :
- User, UserProfile, UserPreferences, UserSpotifyData, UserSubscription
- Optimisé pour l’analytics, l’IA, la recommandation, la monétisation, la data lineage, la multi-tenancy

## Fonctionnalités
- Validation, sécurité, audit, soft-delete, timestamps, attribution utilisateur, multi-tenancy
- CI/CD-ready, gouvernance, conformité, logging, monitoring, data lineage
- Extensible pour nouveaux modèles utilisateur, pipelines, intégrations
- Optimisé pour PostgreSQL, MongoDB, architectures hybrides

## Bonnes pratiques
- Tous les modèles sont relus et validés par le Core Team
- Les contrôles de sécurité et conformité sont obligatoires
- L’utilisation est loggée pour audit et traçabilité

## Exemple d’utilisation
```python
from .user import User
user = User.create(email="user@email.com", password_hash="...", role="artist")
```

## Gouvernance & Extension
- Toute modification doit respecter la convention de nommage/version et inclure des docstrings
- Sécurité, audit et conformité sont appliqués à tous les niveaux

---
*Cette documentation est générée et maintenue automatiquement dans le CI/CD. Dernière mise à jour : juillet 2025.*

