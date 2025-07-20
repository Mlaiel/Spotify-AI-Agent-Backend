# Logging-Modul – Spotify AI Agent

## Übersicht
Dieses Modul stellt zentrale Logging-Konfigurationen für alle Umgebungen bereit (Development, Production, Testing). Es ist optimiert für Security, Audit, ML/AI, Observability und Enterprise-Compliance.

### Funktionen
- JSON-Logs (Production), strukturierte Logs (Dev/Test)
- Log-Rotation, File- und Console-Handler
- Sentry-Integration (Prod), Prometheus/OTEL-Trace
- Maskierung sensibler Daten, Audit, Trace- und User-IDs
- ML/AI-spezifische Logger

### Best Practices
- Niemals sensible Daten im Klartext loggen
- Rotation und Aufbewahrung für Compliance (GDPR, SOC2)
- Sentry/Alerting nur für Prod aktivieren
- Logging-Konfiguration pro Umgebung anpassen

### Autoren & Rollen
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
Für Details siehe die einzelnen Logging-Konfigurationsdateien und die Projekt-Checkliste.
