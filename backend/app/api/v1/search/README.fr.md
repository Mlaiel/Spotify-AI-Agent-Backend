# 🔍 Module Recherche Avancée IA (FR)

Ce module fournit des API de recherche ultra-avancées pour artistes Spotify : recherche fulltext, vectorielle, sémantique, facettes, analytics.

## Fonctionnalités principales
- Recherche fulltext (Elasticsearch/OpenSearch)
- Recherche vectorielle (embeddings, FAISS, OpenSearch)
- Recherche sémantique (transformers HuggingFace)
- Recherche à facettes, suggestions, autocomplétion
- Analytics de recherche, logs, audit
- Sécurité, validation stricte, monitoring

## Endpoints principaux
- `POST /search/fulltext` : Recherche texte classique
- `POST /search/vector` : Recherche vectorielle (embeddings)
- `POST /search/semantic` : Recherche sémantique IA
- `POST /search/facets` : Recherche à facettes
- `GET /search/analytics` : Statistiques et logs de recherche

## Sécurité & Authentification
- OAuth2 obligatoire, rate limiting, audit trail
- Permissions RBAC (artiste, admin)

## Exemples d'intégration
```python
import requests
resp = requests.post('https://api.monsite.com/search/semantic', json={"query": "playlist chill lofi"}, headers={"Authorization": "Bearer ..."})
```

## Cas d'usage
- Recherche de morceaux, playlists, artistes, analytics, recommandations, autocomplétion.

## Monitoring & Qualité
- Logs centralisés, alertes Sentry, tests unitaires/CI/CD, conformité RGPD.

Pour plus de détails, voir la documentation technique et les exemples dans ce dossier.

