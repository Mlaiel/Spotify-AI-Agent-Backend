# PostgreSQL-Skripte – Spotify AI Agent (DE)

Dieses Verzeichnis enthält industrielle Skripte für Automatisierung, Audit, Monitoring, Backup und Business-Daten-Seeding.

## Enthaltene Skripte

- **seed_users.py**: Testnutzer hinzufügen (KI-Profile, Rollen, E-Mails)
- **seed_analytics.py**: Simulierte Analytics-Daten für KI generieren
- **seed_music.py**: Musik-Tracks, Künstler, Genres, Dauer, Popularität hinzufügen
- **audit_log.py**: PostgreSQL-Logs analysieren, verdächtige Zugriffe erkennen, Audit-Report erzeugen
- **monitoring.py**: Health, Größe, Verbindungen, Locks, langsame Queries prüfen
- **auto_backup.py**: Sicheres automatisches Backup mit Rotation, Logs, bereit für Cron/CI/CD
- **alert_slow_queries.py**: Langsame Queries (>5min) erkennen und alarmieren

## Nutzung

```bash
python3 seed_users.py
python3 seed_analytics.py
python3 seed_music.py
python3 audit_log.py
python3 monitoring.py
python3 auto_backup.py
python3 alert_slow_queries.py
```

> Alle Skripte sind für Automatisierung (CI/CD, Cron etc.) bereit und folgen Security Best Practices (keine Hardcoded Credentials, Logs, Audit).
