# 🚀 Spotify AI Agent - API Core Modul

## Überblick

Das **API Core Modul** ist die grundlegende Schicht des Spotify AI Agent Backends und bietet Enterprise-Level-Infrastrukturkomponenten für API-Entwicklung, Request/Response-Handling, Konfigurationsmanagement und Anwendungskontext.

## 🏗️ Architektur

```
app/api/core/
├── __init__.py           # Modul-Exports und Initialisierung
├── config.py             # Konfigurationsmanagement und Validierung
├── context.py            # Request-Kontext und Dependency Injection
├── exceptions.py         # Benutzerdefinierte Exception-Hierarchie
├── factory.py            # Factory-Patterns für Komponentenerstellung
├── response.py           # Response-Standardisierung und Formatierung
└── README.de.md          # Diese Dokumentation
```

## 🔧 Schlüsselkomponenten

### Konfigurationsmanagement (`config.py`)
- **APIConfig**: Zentrale API-Konfiguration mit Validierung
- **DatabaseConfig**: Datenbankverbindungseinstellungen
- **SecurityConfig**: Sicherheitsrichtlinien und Authentifizierung
- **MonitoringConfig**: Observability- und Metriken-Konfiguration
- Umgebungsspezifische Konfigurationen (dev, staging, prod)

### Request-Kontext (`context.py`)
- **RequestContext**: Thread-sicheres Request-Kontext-Management
- **DependencyInjector**: Service-Dependency-Injection
- **ContextualLogger**: Kontextbewusstes Logging
- Request-Tracking und Korrelations-IDs

### Exception-Handling (`exceptions.py`)
- **APIException**: Basis-Exception für alle API-Fehler
- **ValidationError**: Eingabevalidierungsfehler
- **AuthenticationError**: Authentifizierungsfehler
- **BusinessLogicError**: Domänenspezifische Fehler
- Strukturierte Fehlerantworten mit i18n-Unterstützung

### Factory-Patterns (`factory.py`)
- **ComponentFactory**: Generische Komponentenerstellung
- **ServiceFactory**: Service-Instanz-Management
- **MiddlewareFactory**: Middleware-Chain-Konstruktion
- **DatabaseFactory**: Datenbankverbindungs-Pooling

### Response-Standardisierung (`response.py`)
- **APIResponse**: Standardisiertes Response-Format
- **PaginatedResponse**: Paginierte Datenantworten
- **ErrorResponse**: Fehlerantwort-Formatierung
- **SuccessResponse**: Erfolgsantwort-Hilfsfunktionen
- Response-Komprimierung und Caching-Header

## 🚀 Schnellstart

### Grundlegende Verwendung

```python
from app.api.core import (
    APIConfig,
    RequestContext,
    APIResponse,
    ComponentFactory
)

# Konfiguration initialisieren
config = APIConfig.from_environment()

# Request-Kontext erstellen
with RequestContext() as ctx:
    ctx.set_user_id("user_123")
    ctx.set_correlation_id("req_456")
    
    # Factory zur Komponentenerstellung verwenden
    service = ComponentFactory.create_service("user_service")
    
    # Standardisierte Antwort erstellen
    response = APIResponse.success(
        data={"message": "Hallo Welt"},
        meta={"version": "1.0.0"}
    )
```

### Konfigurationsmanagement

```python
from app.api.core.config import APIConfig, get_config

# Aktuelle Konfiguration abrufen
config = get_config()

# Auf spezifische Einstellungen zugreifen
database_url = config.database.url
redis_url = config.cache.redis_url
log_level = config.logging.level

# Konfiguration validieren
config.validate()
```

### Exception-Handling

```python
from app.api.core.exceptions import ValidationError, APIException

@app.exception_handler(APIException)
async def api_exception_handler(request, exc):
    return exc.to_response()

# Validierungsfehler auslösen
if not user_id:
    raise ValidationError(
        message="Benutzer-ID ist erforderlich",
        field="user_id",
        code="MISSING_USER_ID"
    )
```

## 🔒 Sicherheitsfeatures

- **Eingabevalidierung**: Umfassende Request-Validierung
- **Authentifizierung**: JWT- und API-Key-Authentifizierung
- **Autorisierung**: Rollenbasierte Zugriffskontrolle (RBAC)
- **Rate Limiting**: Per-Benutzer- und Per-Endpoint-Limits
- **CORS**: Cross-Origin Resource Sharing-Richtlinien
- **Sicherheits-Header**: OWASP-konforme Sicherheits-Header

## 📊 Monitoring & Observability

- **Metriken**: Prometheus-kompatible Metriken
- **Tracing**: OpenTelemetry Distributed Tracing
- **Logging**: Strukturiertes JSON-Logging mit Korrelations-IDs
- **Health Checks**: Anwendungs- und Abhängigkeits-Health-Monitoring
- **Performance**: Request/Response-Timing und Profiling

## 🧪 Tests

```bash
# Core-Modul-Tests ausführen
pytest tests_backend/app/api/core/ -v

# Mit Coverage ausführen
pytest tests_backend/app/api/core/ --cov=app.api.core --cov-report=html

# Performance-Tests ausführen
pytest tests_backend/app/api/core/ -m performance
```

## 📈 Performance

- **Antwortzeit**: < 10ms für Konfigurationszugriff
- **Speicherverbrauch**: Optimiert für geringen Speicher-Footprint
- **Durchsatz**: Unterstützt 10.000+ Requests pro Sekunde
- **Caching**: Intelligentes Konfigurations- und Response-Caching

## 🔧 Konfiguration

### Umgebungsvariablen

```env
# API-Konfiguration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_VERSION=1.0.0

# Datenbank
DATABASE_URL=postgresql://user:pass@localhost:5432/spotify_ai
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Sicherheit
JWT_SECRET_KEY=ihr-geheimer-schlüssel
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
LOG_LEVEL=INFO
```

## 🌐 Internationalisierung

Das Core-Modul unterstützt mehrere Sprachen:
- **Deutsch** (Standard)
- **Englisch** (English)
- **Französisch** (français)
- **Spanisch** (español)

## 🤝 Mitwirken

1. Befolgen Sie den etablierten Code-Stil und Patterns
2. Fügen Sie umfassende Tests für neue Features hinzu
3. Aktualisieren Sie die Dokumentation für API-Änderungen
4. Stellen Sie sicher, dass alle Sicherheitsprüfungen bestehen
5. Wahren Sie die Rückwärtskompatibilität

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz - siehe [LICENSE](../../../LICENSE)-Datei für Details.

## 👥 Autoren

- **Fahed Mlaiel** - Lead Developer & Enterprise Architect
- **Spotify AI Agent Team** - Core Development Team

---

**Enterprise-Grade API-Infrastruktur** | Mit ❤️ für Skalierbarkeit und Performance entwickelt
