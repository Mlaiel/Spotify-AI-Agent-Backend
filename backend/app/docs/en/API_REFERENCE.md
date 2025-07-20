# API Reference – Spotify AI Backend (EN)

This section details all endpoints, schemas, authentication flows, and usage examples for the backend API.

## 1. Authentication & Security
- OAuth2 (PKCE), JWT, scope management
- Integration examples (curl, Python, JS)

## 2. Main Endpoints
| Method | Endpoint                | Description                        |
|--------|-------------------------|------------------------------------|
| POST   | /api/v1/auth/login      | User authentication                |
| GET    | /api/v1/spotify/me      | Connected Spotify profile info     |
| POST   | /api/v1/ai_agent/query  | AI request (analysis, generation)  |
| POST   | /api/v1/content/generate| Music content generation           |
| GET    | /api/v1/analytics/stats | Advanced statistics                |
| POST   | /api/v1/collab/match    | AI-powered collaboration matching  |

## 3. Example Calls
```bash
curl -X POST https://.../api/v1/auth/login -d '{"email": "...", "password": "..."}'
```

## 4. Data Schemas (extracts)
- User, SpotifyData, AIContent, Collaboration, Analytics
- Pydantic validation, payload examples

## 5. Webhooks & Real-Time
- Spotify webhooks (listening, playlist, analytics)
- Websockets for AI notifications

## 6. Versioning & Compatibility
- API versioning (v1, v2)
- Deprecation strategy

For each endpoint, see details in the generated interactive documentation (Swagger/OpenAPI).
