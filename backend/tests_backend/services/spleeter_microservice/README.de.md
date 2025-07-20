# Erweiterte Tests – Spleeter Microservice

Dieses Verzeichnis enthält alle Unit- und Integrationstests für den Spleeter-Microservice des Spotify AI Agent Backends.

## Struktur
- `test_config.py`: Tests für Konfiguration und Umgebungsvariablen
- `test_utils.py`: Tests für Utilities (Validierung, temporäre Dateien)
- `test_security.py`: Tests für Sicherheit (API-Key, JWT ready)
- `test_monitoring.py`: Tests für Prometheus-Metriken
- `test_health.py`: Tests für den /health-Endpoint
- `__init__.py`: Initialisierung des Testmoduls

## Ausführung
Alle Tests ausführen mit:
```bash
pytest
```

## Best Practices
- Nur lizenzfreie Audio-Fixtures verwenden
- Niemals echte Business-Logik mocken (Tests auf realem Code)
- Jeden Testfall dokumentieren
- Integrationstests für die API hier zentralisieren

## Rollen & Autor
- **Lead Dev & KI-Architekt**: Fahed Mlaiel
- **Senior Backend Entwickler (Python/FastAPI/Django)**: Fahed Mlaiel
- **Machine Learning Ingenieur (TensorFlow/PyTorch/Hugging Face)**: Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)**: Fahed Mlaiel
- **Backend Security Spezialist**: Fahed Mlaiel
- **Microservices Architekt**: Fahed Mlaiel

© 2025 Fahed Mlaiel – Alle Rechte vorbehalten
