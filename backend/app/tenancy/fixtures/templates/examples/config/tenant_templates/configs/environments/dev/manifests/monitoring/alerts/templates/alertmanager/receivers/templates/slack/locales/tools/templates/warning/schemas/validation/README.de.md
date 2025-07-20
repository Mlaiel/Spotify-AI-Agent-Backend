# Validierungsmodul - Spotify AI Agent

## Überblick

Das Validierungsmodul bietet umfassende Datenvalidierungs- und Sanitisierungsfunktionen für das Schema-System des Spotify AI Agent. Es umfasst benutzerdefinierte Validatoren, Sicherheitsregeln und Leistungsoptimierungen zur Gewährleistung der Datenintegrität und -sicherheit in allen Systemkomponenten.

## Hauptfunktionen

### 🔒 Erweiterte Validierungsregeln
- **Tenant-ID-Validierung**: Muster-Matching, Prüfung reservierter Wörter, Längenbegrenzungen
- **Alert-Nachricht-Validierung**: Inhalts-Sanitisierung, Erkennung bösartigen Codes
- **Korrelations-ID-Validierung**: Format-Verifikation und Konsistenzprüfungen
- **Metadaten-Validierung**: Größenbegrenzungen, JSON-Strukturverifikation
- **Tag-Validierung**: Schlüssel-Wert-Paar-Regeln, Namenskonventionen
- **Empfänger-Validierung**: Kanalspezifische Formatvalidierung (E-Mail, Slack, SMS)

### 🛡️ Sicherheits-Sanitisierung
- **HTML-Sanitisierung**: Entfernung gefährlicher Tags und Attribute
- **SQL-Injection-Schutz**: Keyword-Erkennung und -Prävention
- **XSS-Prävention**: Entfernung von Skripten und Event-Handlern
- **Input-Normalisierung**: Behandlung von Leerzeichen und Steuerzeichen

### ⚙️ Konfigurations-Validatoren
- **JSON-Template-Validierung**: Jinja2-Syntaxverifikation
- **Cron-Ausdruck-Validierung**: Zeitplan-Formatprüfung
- **URL-Validierung**: Protokoll- und Domain-Beschränkungen
- **Zeitbereich-Validierung**: Logische Konsistenzprüfungen

### 🎵 Spotify-Domain-Validatoren
- **Spotify-IDs**: Validierung von Track-, Künstler-, Album-, Playlist-Identifikatoren
- **Spotify-URIs**: Validierung vollständiger Spotify-URIs
- **Audio-Features**: Validierung von Audio-Features mit gültigen Bereichen
- **Musik-Genres**: Validierung von Genre-Namen und Inhaltsfilterung

### 🤖 ML/Audio-Validatoren
- **Audio-Formate**: Validierung von Spezifikationen (Sample Rate, Kanäle, Bit-Tiefe)
- **ML-Modell-Konfiguration**: Modelltyp-spezifische Validierung
- **Performance-Metriken**: Validierung von Latenz- und Durchsatz-Metriken
- **Ressourcennutzung**: Validierung von System-Metriken

### 🔐 Sicherheitsregeln
- **Passwort-Stärke**: Komplexitätskriterien, verbotene Muster
- **API-Schlüssel**: Format- und Längenvalidierung
- **JWT-Token**: Grundlegende Strukturverifikation
- **Verschlüsselungsschlüssel**: Fernet- und AES-Validierung
- **IP-Adressen**: Validierung mit privaten Domain-Beschränkungen
- **Sicherheits-Header**: Validierung empfohlener Sicherheits-Header

### ⚡ Leistungsoptimierungen
- **Validierungs-Cache**: LRU-Cache mit TTL für häufige Validierungen
- **Batch-Verarbeitung**: Parallele Validierung für große Volumen
- **Adaptive Validierung**: Automatische Anpassung je nach Last
- **Profiling**: Detaillierte Leistungsmetriken

## Architektur

```
validation/
├── __init__.py              # Hauptklassen und Dekoratoren
├── custom_validators.py     # Domain-spezifische Validatoren
├── security_rules.py       # Sichere Validierungsregeln
├── configuration.py        # System-Konfigurationsschemas
├── performance.py          # Leistungsoptimierte Validatoren
├── README.md              # Hauptdokumentation
├── README.fr.md          # Französische Dokumentation
└── README.de.md          # Deutsche Dokumentation
```

## Nutzungsbeispiele

### Grundvalidierung

```python
from pydantic import BaseModel
from schemas.validation import (
    validate_tenant_id_field, 
    validate_alert_message_field,
    validate_metadata_field
)

class AlertSchema(BaseModel):
    tenant_id: str
    message: str
    metadata: Dict[str, Any]
    
    # Validatoren anwenden
    _validate_tenant_id = validate_tenant_id_field()
    _validate_message = validate_alert_message_field()
    _validate_metadata = validate_metadata_field()
```

### Benutzerdefinierte Validierungsregeln

```python
from schemas.validation import ValidationRules

# Tenant-ID validieren
try:
    clean_tenant_id = ValidationRules.validate_tenant_id("mein-tenant-2024")
except ValueError as e:
    print(f"Validierungsfehler: {e}")

# Alert-Nachricht validieren
clean_message = ValidationRules.validate_alert_message(
    "System-Alert: Hohe CPU-Auslastung erkannt"
)

# Empfänger nach Kanal validieren
email_recipients = ValidationRules.validate_recipients_list(
    ["admin@example.com", "ops@company.com"],
    NotificationChannel.EMAIL
)
```

### Daten-Sanitisierung

```python
from schemas.validation import DataSanitizer

# HTML-Inhalt sanitisieren
clean_html = DataSanitizer.sanitize_html(
    "<p>Sicherer Inhalt</p><script>alert('xss')</script>"
)
# Ergebnis: "<p>Sicherer Inhalt</p>"

# Leerzeichen normalisieren
clean_text = DataSanitizer.normalize_whitespace(
    "  Mehrere   Leerzeichen\t\nund\r\nZeilenumbrüche  "
)
# Ergebnis: "Mehrere Leerzeichen und Zeilenumbrüche"

# Langen Text kürzen
short_text = DataSanitizer.truncate_text(
    "Sehr langer Textinhalt hier...", 
    max_length=20, 
    suffix="..."
)
```

### Spotify-Validatoren

```python
from schemas.validation.custom_validators import SpotifyDomainValidators

# Spotify-Track-ID validieren
track_id = SpotifyDomainValidators.validate_spotify_id(
    "4iV5W9uYEdYUVa79Axb7Rh", "track"
)

# Spotify-URI validieren
uri = SpotifyDomainValidators.validate_spotify_uri(
    "spotify:track:4iV5W9uYEdYUVa79Axb7Rh"
)

# Audio-Features validieren
features = SpotifyDomainValidators.validate_audio_features({
    "danceability": 0.8,
    "energy": 0.7,
    "valence": 0.6,
    "tempo": 120.0
})
```

### Sicherheitsvalidierung

```python
from schemas.validation.security_rules import SecurityValidationRules

# Passwort-Stärke validieren
password = SecurityValidationRules.validate_password_strength(
    "MeinSicheresPasswort123!"
)

# API-Schlüssel validieren
api_key = SecurityValidationRules.validate_api_key(
    "sk_live_51234567890abcdef1234567890abcdef"
)

# IP-Adresse validieren
ip = SecurityValidationRules.validate_ip_address(
    "192.168.1.100", allow_private=True
)

# Sicheren Token generieren
secure_token = SecurityValidationRules.generate_secure_token(32)
```

### System-Konfiguration

```python
from schemas.validation.configuration import EnvironmentConfig, DatabaseConfig

# Datenbank-Konfiguration
db_config = DatabaseConfig(
    type="postgresql",
    host="localhost",
    port=5432,
    database="spotify_ai_agent",
    username="app_user",
    password="secure_password_123",
    min_connections=5,
    max_connections=20
)

# Vollständige Umgebungskonfiguration
env_config = EnvironmentConfig(
    environment="production",
    version="1.0.0",
    tenant_id="main",
    database=db_config,
    # ... weitere Konfigurationen
)
```

### Leistungsoptimierungen

```python
from schemas.validation.performance import (
    OptimizedValidationRules,
    BatchValidationProcessor,
    adaptive_validator
)

# Optimierte Tenant-ID-Validierung
tenant_id = OptimizedValidationRules.validate_tenant_id_fast("mein-tenant")

# Batch-Validierung
processor = BatchValidationProcessor()
results = await processor.validate_batch(
    tenant_ids_list,
    OptimizedValidationRules.validate_tenant_id_fast
)

# Adaptive Validierung
result = adaptive_validator.validate_adaptive(
    data, ValidationRules.validate_tenant_id
)
```

## Validierungsmuster

Das Modul umfasst vordefinierte Muster für gängige Validierungsszenarien:

- **TENANT_ID**: `^[a-z0-9_-]+$`
- **CORRELATION_ID**: `^[a-zA-Z0-9_-]+$`
- **EMAIL**: RFC 5322-konformes Muster
- **PHONE**: Internationales Format-Support
- **URL**: HTTP/HTTPS mit Domain-Beschränkungen
- **VERSION**: Semantische Versionierung
- **HEX_COLOR**: 6-stellige Hexadezimal-Farben

## Leistungsbetrachtungen

### Kompilierte Muster
Alle Regex-Muster sind für optimale Leistung in hochfrequenten Szenarien vorkompiliert.

### Caching
Validierungsergebnisse werden bei Bedarf zwischengespeichert, um redundante Verarbeitung zu vermeiden.

### Batch-Validierung
Unterstützung für die Validierung mehrerer Elemente in einem einzigen Vorgang:

```python
# Mehrere Empfänger auf einmal validieren
recipients = ["user1@example.com", "user2@example.com", "user3@example.com"]
validated = ValidationRules.validate_recipients_list(recipients, NotificationChannel.EMAIL)
```

## Fehlerbehandlung

### Validierungsfehler
Alle Validierungsfehler bieten detaillierte Meldungen für das Debugging:

```python
try:
    ValidationRules.validate_tenant_id("UNGÜLTIGER-TENANT!")
except ValueError as e:
    # Fehler: "Tenant ID must contain only letters, numbers, hyphens and underscores"
    handle_validation_error(e)
```

### Sicherheitsverletzungen
Sicherheitsbezogene Validierungsfehler werden protokolliert und können Alerts auslösen:

```python
# Dies löst einen Fehler aus und protokolliert ein Sicherheitsereignis
ValidationRules.validate_alert_message("<script>bösartiger_code()</script>")
```

## Pydantic-Integration

### Benutzerdefinierte Validatoren
Verwenden Sie die bereitgestellten Dekoratoren für nahtlose Pydantic-Integration:

```python
from pydantic import BaseModel
from schemas.validation import validate_tenant_id_field, validate_time_range_fields

class TenantAlert(BaseModel):
    tenant_id: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    
    _validate_tenant = validate_tenant_id_field()
    _validate_time_range = validate_time_range_fields('start_time', 'end_time')
```

### Root-Validatoren
Für feldübergreifende Validierungslogik:

```python
from pydantic import root_validator
from schemas.validation import ValidationRules

class ComplexSchema(BaseModel):
    field1: str
    field2: str
    
    @root_validator
    def validate_consistency(cls, values):
        # Benutzerdefinierte feldübergreifende Validierungslogik
        if values.get('field1') and values.get('field2'):
            # Konsistenz zwischen Feldern sicherstellen
            pass
        return values
```

## Konfiguration

### Umgebungsvariablen
- `VALIDATION_STRICT_MODE`: Strikte Validierung aktivieren/deaktivieren
- `MAX_VALIDATION_CACHE_SIZE`: Validierungs-Cache-Größenbegrenzung
- `VALIDATION_LOG_LEVEL`: Logging-Level für Validierungsereignisse

### Anpassung
Validierungsregeln für spezifische Anwendungsfälle erweitern:

```python
from schemas.validation import ValidationRules

class CustomValidationRules(ValidationRules):
    @classmethod
    def validate_custom_field(cls, value: str) -> str:
        # Benutzerdefinierte Validierungslogik
        return super().validate_tenant_id(value)
```

## Sicherheits-Best-Practices

1. **Input-Sanitisierung**: Benutzereingaben vor der Verarbeitung immer sanitisieren
2. **Whitelist-Validierung**: Positive Validierung (erlaubte Muster) statt Blacklists verwenden
3. **Längenbegrenzungen**: Vernünftige Grenzen für alle String-Felder durchsetzen
4. **Inhaltsfilterung**: Bösartige Muster in Benutzerinhalten prüfen
5. **URL-Beschränkungen**: Webhook-URLs validieren und beschränken zur SSRF-Prävention
6. **Template-Sicherheit**: Template-Syntax validieren zur Injection-Angriff-Prävention

## Monitoring und Metriken

Das Validierungsmodul umfasst integrierte Monitoring-Fähigkeiten:

```python
from schemas.validation.performance import get_performance_stats

# Leistungsstatistiken abrufen
stats = get_performance_stats()
print(f"Cache-Trefferrate: {stats['cache_stats']}")
print(f"Validierungs-Timing: {stats['profiler_stats']}")
```

## Mitwirkung

Beim Hinzufügen neuer Validierungsregeln:

1. Der bestehenden Musterstruktur folgen
2. Umfassende Fehlermeldungen einschließen
3. Leistungs-Benchmarks für komplexe Validatoren hinzufügen
4. Sicherheitsimplikationen dokumentieren
5. Nutzungsbeispiele bereitstellen

---

Dieses Modul wurde als Teil des Spotify AI Agent-Projekts entwickelt und konzentriert sich auf Datenvalidierung und -sicherheit auf Unternehmensebene.
