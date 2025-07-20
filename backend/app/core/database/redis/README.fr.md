# Module Redis pour Spotify AI Agent (FR)

Ce module fournit une intégration Redis sécurisée, scalable et industrialisée pour les applications IA musicales et analytiques.

## Équipe créatrice (rôles)
- Lead Dev & Architecte IA : [À compléter]
- Développeur Backend Senior (Python/FastAPI/Django) : [À compléter]
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face) : [À compléter]
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB) : [À compléter]
- Spécialiste Sécurité Backend : [À compléter]
- Architecte Microservices : [À compléter]

## Fonctionnalités :
- Cache manager sécurisé, TTL, invalidation, logs, DI ready
- Cluster manager (auto-discovery, failover, monitoring)
- Pub/Sub manager (channels, events, hooks, logs)
- Rate limiter (token bucket, sliding window, anti-abus, logs)
- Session store (chiffrement, expiration, audit, sécurité)

**Exemple :**
```python
from .redis import get_cache
cache = get_cache()
cache.set("user:1", {"name": "Alice"})
```

## Sécurité :
- Ne jamais hardcoder les credentials
- Toujours activer TLS & Auth
- Monitoring, audit, rotation des clés

## Dépannage :
- Voir les logs en cas de problème de connexion ou de cluster
- Utiliser les health-checks intégrés

