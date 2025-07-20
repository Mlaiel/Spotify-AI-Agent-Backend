# 🎵 Module Génération Musicale IA (FR)

Ce module fournit des API avancées pour la génération, l'analyse, le mastering et la manipulation audio par IA, à destination des artistes Spotify.

## Fonctionnalités principales
- Génération musicale IA (Stable Audio, Riffusion, HuggingFace)
- Génération de beats, synthèse audio, effets, mastering automatique
- Analyse harmonique, séparation de stems, prévisualisation
- API RESTful sécurisée, endpoints testés
- Prise en charge asynchrone (FastAPI, Celery)
- Stockage cloud (S3, Cloudinary), formats multiples

## Endpoints principaux
- `POST /music/generate` : Générer un morceau IA à partir d'un prompt ou d'un style
- `POST /music/remix` : Créer un remix automatique
- `POST /music/master` : Masteriser un audio
- `POST /music/stems` : Séparer un morceau en stems
- `POST /music/effects` : Appliquer des effets audio
- `GET /music/preview/{track_id}` : Prévisualiser un audio généré

## Sécurité & Authentification
- OAuth2 obligatoire (Spotify, Auth0)
- Rate limiting, audit trail, validation stricte (Pydantic)
- Permissions RBAC (artiste, admin)

## Exemples d'intégration
```python
import requests
resp = requests.post('https://api.monsite.com/music/generate', json={"prompt": "lofi chill beat"}, headers={"Authorization": "Bearer ..."})
```

## Cas d'usage
- Génération de démos, remix, mastering rapide, analyse harmonique, création de stems pour remix/collab.

## Monitoring & Qualité
- Logs centralisés, alertes Sentry, tests unitaires/CI/CD, conformité RGPD.

Pour plus de détails, voir la documentation technique et les exemples dans ce dossier.

