# 🔍 KI-Suchmodul (DE)

Dieses Modul bietet hochmoderne Such-APIs für Spotify-Künstler: Volltextsuche, Vektorsuche, semantische Suche, Facettensuche, Analytics.

## Hauptfunktionen
- Volltextsuche (Elasticsearch/OpenSearch)
- Vektorsuche (Embeddings, FAISS, OpenSearch)
- Semantische Suche (HuggingFace Transformers)
- Facettensuche, Vorschläge, Autocomplete
- Such-Analytics, Logs, Audit
- Sicherheit, strikte Validierung, Monitoring

## Wichtige Endpunkte
- `POST /search/fulltext`: Klassische Textsuche
- `POST /search/vector`: Vektorsuche (Embeddings)
- `POST /search/semantic`: KI-Semantiksuche
- `POST /search/facets`: Facettensuche
- `GET /search/analytics`: Suchstatistiken und Logs

## Sicherheit & Authentifizierung
- OAuth2 erforderlich, Rate Limiting, Audit Trail
- RBAC-Berechtigungen (Künstler, Admin)

## Integrationsbeispiel
```python
import requests
resp = requests.post('https://api.meineseite.com/search/semantic', json={"query": "chill lofi playlist"}, headers={"Authorization": "Bearer ..."})
```

## Anwendungsfälle
- Suche nach Tracks, Playlists, Künstlern, Analytics, Empfehlungen, Autocomplete.

## Monitoring & Qualität
- Zentrale Logs, Sentry-Alerts, Unittests/CI/CD, DSGVO-Konformität.

Weitere Details und Beispiele finden Sie in der technischen Dokumentation in diesem Ordner.

