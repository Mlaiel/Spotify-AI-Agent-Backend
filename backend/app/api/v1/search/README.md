# 🔍 Advanced AI Search Module (EN)

This module provides ultra-advanced search APIs for Spotify artists: fulltext, vector, semantic, faceted search, and analytics.

## Key Features
- Fulltext search (Elasticsearch/OpenSearch)
- Vector search (embeddings, FAISS, OpenSearch)
- Semantic search (HuggingFace transformers)
- Faceted search, suggestions, autocomplete
- Search analytics, logs, audit
- Security, strict validation, monitoring

## Main Endpoints
- `POST /search/fulltext`: Classic text search
- `POST /search/vector`: Vector (embedding) search
- `POST /search/semantic`: AI semantic search
- `POST /search/facets`: Faceted search
- `GET /search/analytics`: Search stats and logs

## Security & Authentication
- OAuth2 required, rate limiting, audit trail
- RBAC permissions (artist, admin)

## Integration Example
```python
import requests
resp = requests.post('https://api.mysite.com/search/semantic', json={"query": "chill lofi playlist"}, headers={"Authorization": "Bearer ..."})
```

## Use Cases
- Track, playlist, artist search, analytics, recommendations, autocomplete.

## Monitoring & Quality
- Centralized logs, Sentry alerts, unit tests/CI/CD, GDPR compliance.

See technical docs and examples in this folder for more details.

