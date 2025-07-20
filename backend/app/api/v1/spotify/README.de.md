# 🎧 Spotify-Integrationsmodul (DE)

Dieses Modul bietet fortschrittliche APIs für Spotify-Datenintegration, Analytics und Management für Künstler: Statistiken, Playlists, Webhooks, Sync, Analytics.

## Hauptfunktionen
- OAuth2-Authentifizierung, Token-Management, sicheres Refresh
- Abruf von Künstlerstatistiken, Audience, Analytics, Streaming
- Playlist-Management und Synchronisierung
- Erweiterte Track- und Performance-Analyse
- Spotify-Webhooks (Plays, Likes, Playlists…)
- Monitoring, Logs, Audit, DSGVO-Sicherheit

## Wichtige Endpunkte
- `GET /spotify/artist/insights`: Künstlerstatistiken und Audience
- `GET /spotify/streaming/analytics`: Streaming-Analytics
- `POST /spotify/playlists/sync`: Playlists synchronisieren
- `POST /spotify/webhook`: Spotify-Webhook (Events)
- `GET /spotify/tracks/analyze`: Erweiterte Track-Analyse
- `POST /spotify/user/sync`: Nutzerdaten-Sync

## Sicherheit & Authentifizierung
- OAuth2 erforderlich, Token-Management, Rate Limiting, Audit Trail
- RBAC-Berechtigungen (Künstler, Admin)

## Integrationsbeispiel
```python
import requests
resp = requests.get('https://api.meineseite.com/spotify/artist/insights', headers={"Authorization": "Bearer ..."})
```

## Anwendungsfälle
- Künstler-Dashboard, Analytics, Sync, Audience-Monitoring, KI-Alerts.

## Monitoring & Qualität
- Zentrale Logs, Sentry-Alerts, Unittests/CI/CD, DSGVO-Konformität.

Weitere Details und Beispiele finden Sie in der technischen Dokumentation in diesem Ordner.

