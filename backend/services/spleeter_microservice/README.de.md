# Spleeter Microservice – Spotify AI Agent

## Beschreibung
Fortschrittlicher Microservice für die Trennung von Audio-Stems, entwickelt für industrielle KI-Architekturen.

## Funktionen
- Stem-Trennung (2, 4, 5)
- Sichere REST-API (API-Schlüssel)
- Prometheus Monitoring
- Strukturierte Logs
- Bereit für CI/CD und Cloud

## Sicherheit
- Authentifizierung per API-Schlüssel
- Größen-/Typbeschränkung für Dateien
- Audit-Logs


## Erweiterte Bereitstellung
Siehe Haupt-README.md für Docker Compose, Kubernetes, CI/CD.

## Erweiterte Sicherheit
- Authentifizierung per API-Schlüssel (X-API-KEY)
- Erweiterbar für JWT/OAuth2
- Strikte Begrenzung von Dateityp und -größe
- Audit-Logs, Prometheus-Monitoring, Alerting
- Platzhalter für Virenscan (z.B. ClamAV)

## Monitoring & Observability
- Endpoint `/metrics` für Prometheus
- JSON-Logs (structlog)
- Endpoint `/health` für Überwachung

## FAQ
**F: Wie ändere ich den API-Schlüssel?**
A: Die Umgebungsvariable `SPLEETER_API_KEY` anpassen.

**F: Wie kann ich einen Virenscan hinzufügen?**
A: Die Funktion `scan_antivirus` in `utils.py` implementieren (z.B. ClamAV).

**F: Wie produktiv deployen?**
A: Docker Compose oder Kubernetes nutzen, Umgebungsvariablen setzen, Monitoring aktivieren.

**F: Wie Fehler überwachen?**
A: Prometheus-Metriken und JSON-Logs prüfen.

**F: Wo sind die Tests?**
A: Integrationstests werden im Haupt-Backend durchgeführt, nicht in diesem Microservice.

## Best Practices
- Service niemals ohne Authentifizierung betreiben
- Immer Dateityp und -größe begrenzen
- Monitoring und Logging aktivieren
- Temporäre Dateien bereinigen
- Netzwerkzugriffe absichern (Firewall, VPC, etc)

## Rollen & Autor
- **Lead Dev & KI-Architekt**: Fahed Mlaiel
- **Senior Backend Entwickler (Python/FastAPI/Django)**: Fahed Mlaiel
- **Machine Learning Ingenieur (TensorFlow/PyTorch/Hugging Face)**: Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)**: Fahed Mlaiel
- **Backend Security Spezialist**: Fahed Mlaiel
- **Microservices Architekt**: Fahed Mlaiel

---
© 2025 Fahed Mlaiel – Alle Rechte vorbehalten

## Rollen & Autor
- **Lead Dev & KI-Architekt**: Fahed Mlaiel
- **Senior Backend Entwickler (Python/FastAPI/Django)**: Fahed Mlaiel
- **Machine Learning Ingenieur (TensorFlow/PyTorch/Hugging Face)**: Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)**: Fahed Mlaiel
- **Backend Security Spezialist**: Fahed Mlaiel
- **Microservices Architekt**: Fahed Mlaiel

---
© 2025 Fahed Mlaiel – Alle Rechte vorbehalten
