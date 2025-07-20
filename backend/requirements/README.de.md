# Spotify AI Agent – Requirements

## Übersicht
Dieses Verzeichnis enthält alle Abhängigkeitslisten für Entwicklung, Produktion, Security, Testing und Compliance. Die Trennung nach Umgebungen und Rollen garantiert maximale Sicherheit, Wartbarkeit, Skalierbarkeit und Business-Compliance.

### Struktur & Best Practices
- **base.txt**: Kernabhängigkeiten für Backend, ML/AI, Security, DB, API, Microservices
- **development.txt**: Dev-Tools, Linter, Debugger, Test, Mocking
- **production.txt**: Nur geprüfte, produktionsreife Pakete
- **security.txt**: Security, Audit, Compliance, Secrets-Management, Vulnerability-Scan
- **testing.txt**: Test, Mocking, Coverage, Integration, Load, ML/AI-Tests

#### Empfehlungen
- Abhängigkeiten regelmäßig mit `safety` und `pip-audit` prüfen
- Dev/Prod strikt trennen (kein Debug/Dev-Tool in Produktion!)
- ML/AI-Serving und Data-Engineering-Pakete nach Bedarf modularisieren
- Security- und Compliance-Scans automatisieren (CI/CD)
- Alle Änderungen versionieren und reviewen (Code-Review, Audit)

### Autoren & Rollen
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
**Siehe die einzelnen requirements-Dateien und die Projekt-Checkliste für Details.**
