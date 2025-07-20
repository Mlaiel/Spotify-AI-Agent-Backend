# 🎧 Module Intégration Spotify (FR)

Ce module fournit des API avancées pour l'intégration, l'analyse et la gestion des données Spotify pour artistes : stats, playlists, webhooks, synchronisation, analytics.

## Fonctionnalités principales
- Authentification OAuth2, gestion tokens, refresh sécurisé
- Récupération stats, audience, analytics, streaming
- Gestion et synchronisation des playlists
- Analyse avancée des morceaux et performances
- Webhooks Spotify (écoutes, likes, playlists…)
- Monitoring, logs, audit, sécurité RGPD

## Endpoints principaux
- `GET /spotify/artist/insights` : Statistiques et audience artiste
- `GET /spotify/streaming/analytics` : Analytics streaming
- `POST /spotify/playlists/sync` : Synchroniser playlists
- `POST /spotify/webhook` : Webhook Spotify (événements)
- `GET /spotify/tracks/analyze` : Analyse avancée de morceaux
- `POST /spotify/user/sync` : Synchronisation données utilisateur

## Sécurité & Authentification
- OAuth2 obligatoire, gestion tokens, rate limiting, audit trail
- Permissions RBAC (artiste, admin)

## Exemples d'intégration
```python
import requests
resp = requests.get('https://api.monsite.com/spotify/artist/insights', headers={"Authorization": "Bearer ..."})
```

## Cas d'usage
- Dashboard artiste, analytics, synchronisation, monitoring audience, alertes IA.

## Monitoring & Qualité
- Logs centralisés, alertes Sentry, tests unitaires/CI/CD, conformité RGPD.

Pour plus de détails, voir la documentation technique et les exemples dans ce dossier.

