# 📚 API v1 – Documentation (FR)

Cette racine regroupe tous les modules d'API avancés pour l'agent IA Spotify : authentification, gestion Spotify, IA, génération de contenu, recherche, analytics, collaboration.

## Modules principaux
- `auth` : Authentification OAuth2, gestion tokens, sécurité
- `spotify` : Intégration Spotify, stats, playlists, webhooks, analytics
- `ai_agent` : Services IA, recommandations, prompts, NLP
- `content_generation` : Génération de contenu IA (textes, posts, descriptions)
- `music_generation` : Génération musicale IA, mastering, stems
- `search` : Recherche avancée (fulltext, vectorielle, sémantique, facettes)
- `analytics` : Analytics, monitoring, stats, logs
- `collaboration` : Matching artistes, suggestions, scoring

## Sécurité & Authentification
- OAuth2 obligatoire, rate limiting, audit trail, RBAC
- Validation stricte (Pydantic), logs, monitoring Sentry

## Exemples d'intégration
```python
import requests
resp = requests.get('https://api.monsite.com/api/v1/spotify/artist/insights', headers={"Authorization": "Bearer ..."})
```

## Monitoring & Qualité
- Logs centralisés, alertes Sentry, tests unitaires/CI/CD, conformité RGPD.

Voir chaque sous-module pour la documentation détaillée et les exemples.

