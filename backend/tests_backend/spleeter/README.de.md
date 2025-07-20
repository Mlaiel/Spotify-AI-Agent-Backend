# 🎵 Spotify AI Agent - Spleeter Tests Vollständiger Leitfaden
# ==========================================================
# 
# Vollständiger Leitfaden für die Ausführung und Wartung
# der Tests für das Enterprise Spleeter-Modul.
#
# 🎖️ Entwickelt vom Enterprise-Expertenteam

# 🎵 Spleeter Tests - Vollständiger Leitfaden

Willkommen zur vollständigen Test-Suite für das **Spleeter**-Modul des Spotify AI-Agenten! 🚀

Diese Dokumentation führt Sie durch Installation, Konfiguration und Ausführung aller verfügbaren Testtypen.

## 📋 Inhaltsverzeichnis

- [🎯 Überblick](#-überblick)
- [🛠️ Installation](#️-installation)
- [🚀 Schnellstart](#-schnellstart)
- [📊 Testtypen](#-testtypen)
- [⚙️ Konfiguration](#️-konfiguration)
- [🔧 Makefile-Befehle](#-makefile-befehle)
- [🐍 Python-Skript](#-python-skript)
- [📈 Code-Abdeckung](#-code-abdeckung)
- [🏭 CI/CD](#-cicd)
- [🔍 Debugging](#-debugging)
- [📝 Mitwirken](#-mitwirken)

## 🎯 Überblick

Die Spleeter-Test-Suite umfasst **11 Testmodule**, die alle Systemaspekte abdecken:

### 📁 Teststruktur

```
tests_backend/spleeter/
├── 📋 conftest.py              # Globale Konfiguration & Fixtures
├── 🧪 test_core.py             # Kern-Engine-Tests
├── 🤖 test_models.py           # Modellverwaltungs-Tests
├── 🎵 test_processor.py        # Audio-Verarbeitungs-Tests
├── 💾 test_cache.py            # Cache-System-Tests
├── 🔧 test_utils.py            # Hilfsprogramm-Tests
├── 📊 test_monitoring.py       # Überwachungs-Tests
├── ⚠️ test_exceptions.py       # Fehlerbehandlungs-Tests
├── 🔗 test_integration.py      # Integrations-Tests
├── ⚡ test_performance.py      # Leistungs-Tests
├── 🛠️ test_helpers.py          # Test-Hilfsprogramme
├── 📋 Makefile                 # Make-Automatisierung
├── 🐍 run_tests.sh             # Bash-Automatisierungsskript
├── ⚙️ pyproject.toml           # Pytest-Konfiguration
└── 📖 README.de.md             # Diese Dokumentation
```

### 🎖️ Test-Abdeckung

- **Unit-Tests**: Validierung einzelner Komponenten
- **Integrations-Tests**: Validierung vollständiger Workflows
- **Leistungs-Tests**: Benchmarks und Optimierungen
- **Stress-Tests**: Validierung unter hoher Last
- **Sicherheits-Tests**: Validierung von Sicherheitskontrollen
- **Regressions-Tests**: Regressionsprävention

## 🛠️ Installation

### System-Voraussetzungen

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg libsndfile1

# macOS
brew install python ffmpeg libsndfile

# Windows (mit chocolatey)
choco install python ffmpeg
```

### Python-Abhängigkeiten Installation

```bash
# Automatische Installation über Skript
cd /workspaces/Achiri/spotify-ai-agent/backend/tests_backend/spleeter
chmod +x run_tests.sh
./run_tests.sh --install-deps

# Oder manuelle Installation
pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark
pip install black flake8 mypy isort coverage[toml]
pip install pytest-html pytest-xdist pytest-timeout
```

### Installation über Makefile

```bash
# Vollständige Abhängigkeits-Installation
make install-deps

# Konfigurationsüberprüfung
make show-config
```

## 🚀 Schnellstart

### Schnelle Tests (< 1 Minute)

```bash
# Über Makefile
make test-fast

# Über Bash-Skript
./run_tests.sh --smoke

# Über direktes pytest
pytest -m "not slow and not performance and not stress" --timeout=30
```

### Vollständige Tests

```bash
# Vollständige CI/CD-Pipeline
./run_tests.sh --ci

# Alle Tests über Makefile
make test-all

# Tests mit Abdeckung
make coverage-html
```

### Spezifische Tests

```bash
# Tests nach Modul
make test-core          # Kern-Engine
make test-models        # Modellverwaltung
make test-processor     # Audio-Verarbeitung
make test-cache         # Cache-System

# Tests nach Typ
make test-unit          # Unit-Tests
make test-integration   # Integrations-Tests
make test-performance   # Leistungs-Tests
```

## 📊 Testtypen

### 🧪 Unit-Tests

Individuelle Validierung jeder Komponente:

```bash
# Unit-Test-Ausführung
pytest test_core.py test_models.py test_processor.py \
       test_cache.py test_utils.py test_monitoring.py test_exceptions.py \
       -v -m "not slow"
```

**Abdeckung**:
- ✅ Engine-Initialisierung
- ✅ Grundlegende Audio-Trennung
- ✅ ML-Modellverwaltung
- ✅ Mehrstufiger Cache
- ✅ Hilfsprogramme und Validierung
- ✅ System-Überwachung
- ✅ Exception-Handling

### 🔗 Integrations-Tests

End-to-End-Workflow-Validierung:

```bash
# Integrations-Tests
pytest test_integration.py -v -m "integration" --timeout=120
```

**Szenarien**:
- ✅ Vollständige Trennung mit Cache
- ✅ Batch-Verarbeitung mit Überwachung
- ✅ Robuste Fehlerbehandlung
- ✅ Nebenläufige Workflows

### ⚡ Leistungs-Tests

Benchmarks und Optimierungen:

```bash
# Leistungs-Tests
pytest test_performance.py -v -m "performance" --benchmark-only
```

**Metriken**:
- ⏱️ Audio-Trennungszeit
- 📊 Speicherverbrauch
- 🚀 Batch-Durchsatz
- 💾 Cache-Leistung
- 📈 Überwachungs-Overhead

### 💪 Stress-Tests

Validierung unter hoher Last:

```bash
# Stress-Tests
pytest test_performance.py -v -m "stress" --timeout=600
```

**Szenarien**:
- 🔥 Maximale CPU-Last
- 💧 Speicher-Stress
- 🌊 Nebenläufige Verarbeitung
- 🎯 Langzeit-Stabilität

## ⚙️ Konfiguration

### Pytest-Konfiguration (pyproject.toml)

Zentralisierte Konfiguration in `pyproject.toml` umfasst:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: langsame Tests",
    "fast: schnelle Tests", 
    "unit: Unit-Tests",
    "integration: Integrations-Tests",
    "performance: Leistungs-Tests",
    "stress: Stress-Tests"
]

addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--durations=10",
    "--maxfail=3"
]
```

### Umgebungsvariablen

```bash
export SPLEETER_TEST_MODE="true"
export SPLEETER_LOG_LEVEL="DEBUG"
export SPLEETER_CACHE_DISABLED="true"
export COVERAGE_MIN="85"
```

### Coverage-Konfiguration

```toml
[tool.coverage.run]
source = ["backend/spleeter"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 85
show_missing = true
```

## 🔧 Makefile-Befehle

### Hauptbefehle

```bash
make help              # Vollständige Hilfe anzeigen
make test              # Basis-Tests (schnell)
make test-all          # Alle Tests (vollständig)
make coverage          # Konsolen-Coverage-Analyse
make coverage-html     # HTML-Coverage-Bericht
make lint              # Linting-Überprüfungen
make format            # Automatische Formatierung
make clean             # Temporäre Dateien aufräumen
```

### Tests nach Modul

```bash
make test-core         # Kern-Engine-Tests
make test-models       # Modellverwaltungs-Tests  
make test-processor    # Audio-Verarbeitungs-Tests
make test-cache        # Cache-System-Tests
make test-utils        # Hilfsprogramm-Tests
make test-monitoring   # Überwachungs-Tests
make test-exceptions   # Fehlerbehandlungs-Tests
```

### Tests nach Typ

```bash
make test-unit         # Unit-Tests
make test-integration  # Integrations-Tests
make test-performance  # Leistungs-Tests
make test-stress       # Stress-Tests
make test-fast         # Nur schnelle Tests
make test-slow         # Nur langsame Tests
```

### Erweiterte Hilfsprogramme

```bash
make test-parallel     # Parallele Tests
make test-report       # Tests mit HTML-Bericht
make benchmark         # Spezifische Benchmarks
make test-security     # Sicherheits-Tests
make test-regression   # Regressions-Tests
make smoke-test        # Smoke-Tests
make acceptance-test   # Akzeptanz-Tests
```

## 🐍 Python-Skript

### Skript-Verwendung

```bash
# Skript ausführbar machen
chmod +x run_tests.sh

# Vollständige Hilfe
./run_tests.sh --help

# Vollständige CI/CD-Pipeline
./run_tests.sh --ci

# Spezifische Tests
./run_tests.sh --unit
./run_tests.sh --integration
./run_tests.sh --performance
./run_tests.sh --stress
```

### Verfügbare Optionen

| Option | Beschreibung |
|--------|-------------|
| `--ci` | Vollständige CI/CD-Pipeline |
| `--unit` | Nur Unit-Tests |
| `--integration` | Nur Integrations-Tests |
| `--performance` | Nur Leistungs-Tests |
| `--stress` | Nur Stress-Tests |
| `--coverage` | Nur Coverage-Analyse |
| `--quality` | Nur Qualitäts-Checks |
| `--smoke` | Smoke-Tests (schnelle Überprüfung) |
| `--install-deps` | Abhängigkeiten installieren |
| `--cleanup` | Temporäre Dateien aufräumen |

### Skript-Konfiguration

```bash
# Umgebungsvariablen zur Anpassung
export COVERAGE_MIN=90           # Mindest-Coverage
export TIMEOUT_UNIT=45          # Unit-Test-Timeout
export TIMEOUT_INTEGRATION=180  # Integrations-Test-Timeout
export TIMEOUT_PERFORMANCE=450  # Leistungs-Test-Timeout
export TIMEOUT_STRESS=900       # Stress-Test-Timeout
```

## 📈 Code-Abdeckung

### Bericht-Generierung

```bash
# Konsolen-Bericht
make coverage

# Detaillierter HTML-Bericht
make coverage-html
# Automatisches Öffnen: coverage_html/index.html

# XML-Bericht (für CI/CD)
make coverage-xml
# Generierte Datei: coverage.xml
```

### Coverage-Schwellenwerte

- **Globale Coverage**: ≥ 85%
- **Pro-Datei-Coverage**: ≥ 80%
- **Branch-Coverage**: ≥ 75%

### Coverage-Analyse

```bash
# Differentielle Coverage
make diff-coverage

# Coverage mit Kontext
pytest --cov=../../spleeter \
       --cov-context=test \
       --cov-branch
```

## 🏭 CI/CD

### GitHub Actions

Der GitHub Actions-Workflow umfasst:

```yaml
# .github/workflows/spleeter-tests.yml
name: 🎵 Spleeter Tests CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * *'  # Tägliche Tests

jobs:
  smoke-tests:    # Schnelle Überprüfungs-Tests
  unit-tests:     # Multi-Plattform Unit-Tests
  integration-tests: # Integrations-Tests mit Services
  performance-tests: # Leistungs-Tests und Benchmarks
  stress-tests:   # Last- und Stress-Tests
  security-tests: # Sicherheits-Tests
  code-quality:   # Code-Qualitäts-Checks
  final-report:   # Konsolidierter Bericht
```

### Lokale Konfiguration

```bash
# Lokale CI/CD-Pipeline-Simulation
./run_tests.sh --ci

# Pre-Commit-Überprüfung
make pre-commit

# Vollständige Validierung
make check-all
```

### Kontinuierliche Integration

```bash
# Schnelle Tests für Entwicklung
make test-fast

# Vollständige Tests für Release
make test-all coverage-html

# Pre-Commit-Überprüfung
make pre-commit
```

## 🔍 Debugging

### Spezifisches Test-Debug

```bash
# Verbose Debug-Modus
make test-debug TEST=test_core.py::test_engine_initialization

# Debug mit pdb
pytest test_core.py::test_engine_initialization -vvv -s --pdb

# Leistungs-Profiling
make profile-tests
```

### Debug-Logs

```bash
# Detaillierte Log-Aktivierung
export SPLEETER_LOG_LEVEL="DEBUG"
pytest test_core.py -vvv -s --log-cli-level=DEBUG
```

### Watch-Modus Tests

```bash
# Watch-Modus für Entwicklung
make watch-tests

# Kontinuierliche Tests mit pytest-watch
pip install pytest-watch
ptw -- -m "not slow"
```

### Fehler-Debugging

```bash
# Tests mit erweiterten Fehlerinformationen
pytest --tb=long --capture=no

# Fehlerbericht-Generierung
pytest --html=failure_report.html --self-contained-html
```

## 📊 Metriken und Überwachung

### Gesammelte Metriken

- **Leistung**: Ausführungszeit, Durchsatz
- **Ressourcen**: CPU, Speicher, I/O
- **Qualität**: Abdeckung, Komplexität
- **Zuverlässigkeit**: Erfolgsrate, Stabilität

### Verfügbare Berichte

```bash
# Vollständiger HTML-Bericht
make test-report

# Leistungs-Metriken
make test-metrics

# JSON-Ergebnis-Export
make test-export
```

### Kontinuierliche Überwachung

```bash
# Tests mit Ressourcen-Überwachung
pytest test_performance.py::TestResourceMonitoring -v

# Speicher-Verfolgung
pytest --memray

# CPU-Profiling
pytest --profile
```

## 🔧 Wartung

### Abhängigkeits-Updates

```bash
# Versions-Überprüfung
pip list --outdated

# Sicheres Update
pip install --upgrade -r requirements/testing.txt

# Kompatibilitäts-Überprüfung
./run_tests.sh --smoke
```

### Periodische Aufräumung

```bash
# Vollständige Aufräumung
make clean

# Cache-Entfernung
rm -rf .pytest_cache __pycache__ .coverage

# Artefakt-Aufräumung
rm -rf coverage_html test_report.html
```

### Test-Optimierung

```bash
# Parallele Tests
make test-parallel

# Fixture-Optimierung
pytest --setup-show

# Dauer-Analyse
pytest --durations=0
```

## 🔐 Sicherheit

### Sicherheits-Tests

```bash
# Spezifische Sicherheits-Tests
make test-security

# Schwachstellen-Scan
bandit -r ../../spleeter/

# Abhängigkeits-Überprüfung
safety check
```

### Eingabe-Validierung

```bash
# Validierungs-Tests
pytest -k "validation or sanitize" test_utils.py

# Injection-Tests
pytest -k "security" test_exceptions.py
```

## 📝 Mitwirken

### Neue Tests hinzufügen

1. **Test-Datei erstellen** im entsprechenden Verzeichnis
2. **Fixtures verwenden** definiert in `conftest.py`
3. **Entsprechende Marker hinzufügen** (`@pytest.mark.unit`, etc.)
4. **Tests dokumentieren** mit klaren Docstrings
5. **Coverage überprüfen** mit `make coverage`

### Test-Struktur

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Tests für neue Funktionalität."""
    
    @pytest.mark.unit
    @pytest.mark.fast
    def test_basic_functionality(self, mock_engine):
        """Test der Grundfunktionalität."""
        # Arrange
        input_data = "test_input"
        expected = "expected_output"
        
        # Act
        result = mock_engine.process(input_data)
        
        # Assert
        assert result == expected
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_integration_workflow(self):
        """Vollständiger Integrations-Test."""
        # Integrations-Test mit echten Abhängigkeiten
        pass
```

### Test-Richtlinien

- **Benennung**: `test_<functionality>_<scenario>`
- **Isolation**: Jeder Test muss unabhängig sein
- **Mocking**: Mocks für externe Abhängigkeiten verwenden
- **Assertions**: Klare und spezifische Assertions
- **Dokumentation**: Erklärende Docstrings
- **Marker**: Angemessene pytest-Marker-Verwendung

### Pre-Commit-Validierung

```bash
# Vollständige Validierung vor Commit
make pre-commit

# Einzelne Überprüfungen
make format    # Code-Formatierung
make lint      # Linting-Überprüfungen
make type-check # Typ-Überprüfung
make test-fast # Schnelle Tests
```

## 🎯 Qualitätsziele

### Leistungs-Ziele

- **Unit-Tests**: < 30 Sekunden gesamt
- **Integrations-Tests**: < 2 Minuten gesamt  
- **Leistungs-Tests**: < 5 Minuten gesamt
- **Audio-Trennung**: < 5 Sekunden/Datei

### Coverage-Ziele

- **Globale Coverage**: ≥ 85%
- **Branch-Coverage**: ≥ 75%
- **Kritische Module**: ≥ 90%

### Zuverlässigkeits-Ziele

- **Erfolgsrate**: ≥ 99%
- **Stabilität**: 0 instabile Tests
- **Reproduzierbarkeit**: 100%

## 🚀 Leistung und Optimierungen

### Test-Optimierungen

```bash
# Parallele Tests
pytest -n auto

# Ergebnis-Caching
pytest --cache-clear  # Cache zurücksetzen
pytest --lf          # Nur letzte Fehlgeschlagene
pytest --ff          # Fehlgeschlagene zuerst

# Speicher-Optimierung
pytest --maxfail=1 --tb=no
```

### Leistungs-Überwachung

```bash
# Detailliertes Profiling
pytest --profile --profile-svg

# Speicher-Überwachung
pytest --memray --memray-bin-path=memory_profile.bin

# Engpass-Analyse
pytest --durations=20
```

## 📞 Support und Hilfe

### Ressourcen

- **Dokumentation**: Dieses README und Docstrings
- **Beispiele**: Bestehende Test-Dateien
- **Konfiguration**: `pyproject.toml`, `conftest.py`
- **Skripte**: `Makefile`, `run_tests.sh`

### Hilfe-Befehle

```bash
# Makefile-Hilfe
make help

# Bash-Skript-Hilfe
./run_tests.sh --help

# Pytest-Hilfe
pytest --help

# Aktive Konfiguration
make show-config
```

### Häufige Problemlösungen

**Fehlschlagende Tests**:
```bash
# Verbose Debug-Modus
pytest -vvv -s --tb=long

# Isolierte Tests
pytest test_file.py::test_function -v
```

**Langsame Leistung**:
```bash
# Profiling
make profile-tests

# Nur schnelle Tests
make test-fast
```

**Coverage-Probleme**:
```bash
# Detaillierter Bericht
make coverage-html

# Fehlende Dateien
coverage report --show-missing
```

---

## 🎉 Fazit

Diese vollständige Test-Suite gewährleistet die Qualität, Leistung und Zuverlässigkeit des Spleeter-Moduls. Verwenden Sie diese Dokumentation als Referenz für alle Ihre Test-Anforderungen!

**Frohes Testen! 🚀🎵**

---

*Entwickelt mit ❤️ vom Enterprise-Expertenteam für den Spotify AI-Agent*
