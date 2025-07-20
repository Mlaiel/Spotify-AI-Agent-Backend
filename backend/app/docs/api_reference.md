# Référence API – Backend IA Spotify

Cette section centralise la documentation des endpoints, schémas, flux d’authentification et exemples d’utilisation de l’API backend.

## 1. Authentification & Sécurité
- OAuth2 (PKCE), JWT, gestion des scopes
- Exemples d’intégration (curl, Python, JS)

## 2. Endpoints principaux
| Méthode | Endpoint                | Description                        |
|---------|-------------------------|------------------------------------|
| POST    | /api/v1/auth/login      | Authentification utilisateur       |
| GET     | /api/v1/spotify/me      | Infos profil Spotify connecté      |
| POST    | /api/v1/ai_agent/query  | Requête IA (analyse, génération)   |
| POST    | /api/v1/content/generate| Génération de contenu musical      |
| GET     | /api/v1/analytics/stats | Statistiques avancées              |
| POST    | /api/v1/collab/match    | Matching collaboratif IA           |

## 3. Exemples d’appels
```bash
curl -X POST https://.../api/v1/auth/login -d '{"email": "...", "password": "..."}'
```

## 4. Schémas de données (extraits)
- User, SpotifyData, AIContent, Collaboration, Analytics
- Validation Pydantic, exemples de payloads

## 5. Webhooks & temps réel
- Webhooks Spotify (écoute, playlist, analytics)
- Websockets pour notifications IA

## 6. Versioning & compatibilité
- Versionnement d’API (v1, v2)
- Stratégie de dépréciation

Pour chaque endpoint, voir détails dans la documentation interactive générée (Swagger/OpenAPI).
