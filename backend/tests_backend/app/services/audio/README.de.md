# Erweiterte Tests – Audio-Modul

Dieses Verzeichnis enthält alle Unit- und Integrationstests für das `audio`-Modul des Spotify AI Agent Backends.

## Struktur
- `test_audio_utils.py`: Tests für Normalisierung, Konvertierung, Feature-Extraktion
- `test_audio_analyzer.py`: Tests für ML-Analyse, Klassifikation, Tagging
- `test_spleeter_client.py`: Tests für den HTTP-Client des Spleeter-Microservice
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
