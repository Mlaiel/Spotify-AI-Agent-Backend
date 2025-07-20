# =============================================================================
# Automatisierungsskripts Dokumentation - Entwicklungsumgebung (Deutsch)
# =============================================================================
# 
# Entwickelt vom Expertenteam unter der Leitung von Fahed Mlaiel
# Lead Dev + AI Architect, Senior Backend Developer, ML Engineer,
# DBA & Data Engineer, Backend Security Specialist, Microservices Architect
# =============================================================================

## Überblick

Dieses Verzeichnis enthält eine vollständige Suite von Automatisierungsskripts für die Entwicklungsumgebung des Spotify AI Agent Projekts. Diese professionellen Skripts ermöglichen ein industrialisiertes und effizientes Management der Entwicklungsumgebung.

## Verfügbare Skripts

### 🚀 **setup_dev.sh** - Initiale Konfiguration
Automatisiertes Konfigurationsskript für die vollständige Entwicklungsumgebung.

**Hauptfunktionen:**
- Automatische Installation der Python Virtual Environment
- Konfiguration von PostgreSQL und Redis Datenbanken
- Installation von Abhängigkeiten und Entwicklungstools
- Vollständige Validierung der Installation

**Typische Verwendung:**
```bash
./setup_dev.sh                    # Standard-Installation
./setup_dev.sh --force            # Erzwungene Neuinstallation
./setup_dev.sh --minimal          # Minimale Installation
```

### 🗄️ **reset_db.sh** - Datenbank-Management
Vollständiges Skript für Zurücksetzen und Wartung der Datenbank.

**Hauptfunktionen:**
- Automatische Sicherung vor Zurücksetzen
- Vollständige Neuerststellung des Schemas
- Ausführung von Alembic-Migrationen
- Laden von Testdaten

**Typische Verwendung:**
```bash
./reset_db.sh                     # Vollständiges Zurücksetzen
./reset_db.sh --no-backup         # Ohne vorherige Sicherung
./reset_db.sh --data-only          # Nur Daten
```

### 🔧 **start_services.sh** - Service-Start
Startet alle für die Entwicklung notwendigen Services mit integrierter Überwachung.

**Hauptfunktionen:**
- Automatischer Start der Services (API, PostgreSQL, Redis)
- Verwaltung von Datenbankmigrationen
- Hintergrund-Modus mit Prozessverfolgung
- Gesundheitsprüfung der Services

**Typische Verwendung:**
```bash
./start_services.sh                      # Interaktiver Modus
./start_services.sh --background          # Hintergrund-Modus
./start_services.sh --start-db --start-redis  # Mit lokalen Services
```

### 📊 **monitor_health.sh** - Überwachung und Monitoring
Vollständiges Überwachungssystem für Services mit Berichtsgenerierung.

**Hauptfunktionen:**
- Kontinuierliche Service-Überwachung
- Sammlung von Systemmetriken
- Automatisierte Performance-Tests
- Generierung von Alarmen und Berichten

**Typische Verwendung:**
```bash
./monitor_health.sh                  # Kontinuierliches Monitoring
./monitor_health.sh --dashboard       # Interaktives Dashboard
./monitor_health.sh --performance     # Performance-Tests
```

### 📋 **manage_logs.sh** - Log-Verwaltung
Industrielles System für Verwaltung, Rotation und Analyse von Logs.

**Hauptfunktionen:**
- Automatische Rotation großer Log-Dateien
- Archivierung mit Kompression
- Intelligente Fehleranalyse
- Generierung von Analyseberichten

**Typische Verwendung:**
```bash
./manage_logs.sh --status            # Log-Status
./manage_logs.sh --analyze-all       # Vollständige Analysen
./manage_logs.sh --full              # Vollständige Wartung
```

## Empfohlene Konfiguration

### Wesentliche Umgebungsvariablen
```bash
# API-Konfiguration
DEV_API_HOST=0.0.0.0
DEV_API_PORT=8000
DEV_HOT_RELOAD=true

# Datenbank-Konfiguration
DEV_DB_HOST=localhost
DEV_DB_PORT=5432
DEV_DB_NAME=spotify_ai_dev

# Monitoring-Konfiguration
MONITORING_INTERVAL=30
DEBUG=true
```

## Verwendungs-Workflows

### 🔄 **Täglicher Start**
```bash
# 1. Erststart (initiale Konfiguration)
./setup_dev.sh

# 2. Service-Start
./start_services.sh --start-db --start-redis

# 3. Status-Überprüfung
./start_services.sh --status
```

### 🔄 **Wöchentliche Wartung**
```bash
# 1. Log-Verwaltung
./manage_logs.sh --full

# 2. Abhängigkeiten aktualisieren
./start_services.sh --update-deps

# 3. Performance-Tests
./monitor_health.sh --performance
```

### 🔄 **Problemlösung**
```bash
# 1. Service-Diagnose
./monitor_health.sh --once

# 2. Fehleranalyse
./manage_logs.sh --analyze-errors

# 3. Zurücksetzen falls notwendig
./reset_db.sh
```

## Überwachung und Alarme

### Überwachte Metriken
- **Services**: API-Verfügbarkeit, PostgreSQL, Redis
- **Performance**: Antwortzeiten, Durchsatz
- **System**: CPU, Speicher, Festplattenspeicher
- **Sicherheit**: Zugangsversuche, Authentifizierungsfehler

### Konfigurierbare Alarm-Schwellenwerte
- CPU > 80%
- Speicher > 85%
- Festplattenspeicher > 90%
- Antwortzeit > 5000ms

## Log-Verwaltung

### Log-Typen
- **Anwendung**: Business- und funktionale Logs
- **Fehler**: System- und Anwendungsfehler
- **Zugang**: API- und Web-Zugangs-Logs
- **Performance**: Performance-Metriken
- **Sicherheit**: Sicherheitsereignisse

### Rotation und Archivierung
- Automatische Rotation bei 100MB
- Archivierung nach 7 Tagen
- Archiv-Aufbewahrung: 30 Tage
- Automatische gzip-Kompression

## Sicherheit

### Implementierte Best Practices
- Verschlüsselung sensibler Geheimnisse
- Validierung von Benutzereingaben
- Sicheres Logging mit Maskierung
- Restriktive Dateiberechtigungen

### Audit und Compliance
- Vollständige Protokollierung von Operationen
- Nachverfolgbarkeit von Änderungen
- Automatische Compliance-Berichte
- Integration mit Audit-Systemen

## Häufige Problembehebung

### Häufige Probleme und Lösungen

#### Services starten nicht
```bash
# Diagnose
./monitor_health.sh --once
./manage_logs.sh --analyze-errors

# Lösung
./reset_db.sh
./start_services.sh --start-db --start-redis
```

#### Verschlechterte Performance
```bash
# Analyse
./monitor_health.sh --performance
./manage_logs.sh --analyze-performance

# Optimierung
./manage_logs.sh --cleanup
```

#### Umfangreiche Logs
```bash
# Wartung
./manage_logs.sh --rotate --cleanup
```

## Technischer Support

### Kontakte
- **Fahed Mlaiel** - Lead Dev + AI Architect
- **Backend-Team** - Täglicher technischer Support

### Zusätzliche Dokumentation
- Detaillierte Logs in `/logs/`
- Analyseberichte in `/logs/reports/`
- Monitoring-Metriken in `/logs/monitoring/`

### Mitwirkung
Verbesserungen und Vorschläge sind erwünscht. Kontaktieren Sie das Entwicklungsteam, um zum Projekt beizutragen.

---

*Diese Dokumentation wird vom Team unter der Leitung von Fahed Mlaiel gepflegt und regelmäßig aktualisiert.*
