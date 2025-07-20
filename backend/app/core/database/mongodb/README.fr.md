# Module MongoDB pour Spotify AI Agent (FR)

Ce module fournit une intégration MongoDB sécurisée, scalable et extensible pour les applications IA musicales et analytiques.

**Fonctionnalités :**
- Connexion sécurisée (TLS, Auth, Pooling, Health-Check, Tracing)
- CRUD, validation, transactions, soft-delete, versioning
- Pipelines d’agrégation dynamiques (ex : Top Artists, Audience Segmentation)
- Gestion automatisée des index & recommandations
- Logging, audit, gestion des exceptions
- Prêt pour l’injection de dépendances et microservices

**Exemple :**
```python
from .mongodb import DocumentManager
user_mgr = DocumentManager("users")
user_id = user_mgr.create({"name": "Alice", "email": "alice@music.com"})
user = user_mgr.get(user_id)
```

**Sécurité :**
- Ne jamais hardcoder les credentials
- Toujours activer TLS & Auth
- Backups réguliers & monitoring

**Dépannage :**
- Voir les logs en cas de problème de connexion
- Utiliser le health-check : `MongoConnectionManager().health_check()`

