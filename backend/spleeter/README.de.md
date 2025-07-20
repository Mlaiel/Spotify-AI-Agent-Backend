# 🎵 Spotify AI Agent - Erweiterte Spleeter-Modul

[![Enterprise-Klasse](https://img.shields.io/badge/Enterprise-Klasse-gold.svg)](https://github.com/spotify-ai-agent)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Lizenz](https://img.shields.io/badge/Lizenz-MIT-green.svg)](LICENSE)

## 🚀 Übersicht

**Erweiterte Enterprise Spleeter-Modul** - Industrielle Audio-Trennungs-Engine für hohe Leistung und skalierbare Musikquellen-Trennung. Dieses Modul bietet modernste KI-gestützte Audio-Verarbeitungsfähigkeiten mit Enterprise-Features wie mehrstufigem Caching, GPU-Optimierung, Batch-Verarbeitung und umfassender Überwachung.

**🎖️ Architekt & Hauptentwickler:** [Fahed Mlaiel](https://github.com/fahed-mlaiel)  
**🏢 Organisation:** Enterprise AI Solutions Team

---

## ✨ Hauptmerkmale

### 🎯 **Kernfunktionen**
- **🤖 Erweiterte KI-Modelle**: Unterstützung für 2-Stems, 4-Stems und 5-Stems Trennung
- **⚡ GPU-Beschleunigung**: Optimierte CUDA/TensorFlow GPU-Verarbeitung
- **🔄 Asynchrone Verarbeitung**: Nicht-blockierende Operationen mit asyncio
- **📦 Batch-Operationen**: Hochdurchsatz-Parallelverarbeitung
- **🎛️ Audio-Vorverarbeitung**: Erweiterte Filterung, Normalisierung, Rauschunterdrückung

### 🏗️ **Enterprise-Architektur**
- **💾 Mehrstufiges Caching**: Speicher (L1) → Festplatte+SQLite (L2) → Redis (L3)
- **📊 Leistungsüberwachung**: Echtzeit-Metriken, Gesundheitsprüfungen, Alarmierung
- **🔧 Modellverwaltung**: Automatischer Download, Validierung, Versionierung
- **🛡️ Sicherheit & Validierung**: Eingabebereinigung, Ressourcenlimits, Fehlerbehandlung
- **📈 Skalierbarkeit**: Microservices-bereit, horizontale Skalierung unterstützt

### 🎵 **Audio-Verarbeitungsexzellenz**
- **📻 Formatunterstützung**: WAV, FLAC, MP3, OGG, M4A, AAC, WMA
- **🔊 Qualitätsoptionen**: Verarbeitung bis zu 192kHz/32-bit
- **🎚️ Dynamische Verarbeitung**: Lautstärke-Normalisierung, Stille-Erkennung
- **📋 Metadaten-Extraktion**: Umfassende Audio-Analyse und Tagging

---

## 🛠️ Installation

### Voraussetzungen
```bash
# Python 3.8+ erforderlich
python --version  # Sollte 3.8+ sein

# Kern-Abhängigkeiten installieren
pip install tensorflow>=2.8.0
pip install librosa>=0.9.0
pip install soundfile>=0.10.0
pip install numpy>=1.21.0
pip install asyncio-mqtt
pip install aioredis
```

### Schnelle Einrichtung
```bash
# Repository klonen
git clone https://github.com/spotify-ai-agent/backend.git
cd backend/spleeter

# Abhängigkeiten installieren
pip install -r requirements.txt

# Modul initialisieren
python -c "from spleeter import SpleeterEngine; print('✅ Installation erfolgreich!')"
```

---

## 🚀 Schnellstart

### Grundlegende Verwendung
```python
import asyncio
from spleeter import SpleeterEngine

async def audio_trennen():
    # Engine initialisieren
    engine = SpleeterEngine()
    await engine.initialize()
    
    # Audio trennen
    result = await engine.separate(
        audio_path="lied.wav",
        model_name="spleeter:2stems-16kHz",
        output_dir="ausgabe/"
    )
    
    print(f"✅ Trennung abgeschlossen: {result.output_files}")

# Ausführen
asyncio.run(audio_trennen())
```

### Erweiterte Konfiguration
```python
from spleeter import SpleeterEngine, SpleeterConfig

# Enterprise-Konfiguration
config = SpleeterConfig(
    # Leistung
    enable_gpu=True,
    batch_size=8,
    worker_threads=4,
    
    # Caching
    cache_enabled=True,
    cache_size_mb=2048,
    redis_url="redis://localhost:6379",
    
    # Audio-Einstellungen
    default_sample_rate=44100,
    enable_preprocessing=True,
    normalize_loudness=True,
    
    # Überwachung
    enable_monitoring=True,
    metrics_export_interval=60
)

engine = SpleeterEngine(config=config)
```

---

## 📚 API-Referenz

### SpleeterEngine
Haupt-Verarbeitungs-Engine mit Enterprise-Fähigkeiten.

#### Methoden

**`async initialize()`**
Initialisiert die Engine und lädt Modelle.

**`async separate(audio_path, model_name, output_dir, **options)`**
Trennt Audio in Stems.
- `audio_path`: Eingabe-Audiodatei-Pfad
- `model_name`: Modell-Identifikator (z.B. "spleeter:2stems-16kHz")
- `output_dir`: Ausgabeverzeichnis für getrennte Stems
- `options`: Zusätzliche Verarbeitungsoptionen

**`async batch_separate(audio_files, **options)`**
Verarbeitet mehrere Dateien parallel.

**`get_available_models()`**
Listet alle verfügbaren Trennungsmodelle auf.

**`get_processing_stats()`**
Ruft Leistungsstatistiken ab.

### ModelManager
Erweiterte Modellverwaltungssystem.

#### Methoden

**`async download_model(model_name, force=False)`**
Lädt und cached ein spezifisches Modell herunter.

**`list_local_models()`**
Listet lokal verfügbare Modelle auf.

**`validate_model(model_name)`**
Überprüft Modell-Integrität.

### CacheManager
Mehrstufiges Caching-System.

#### Methoden

**`async get(key, cache_level="auto")`**
Ruft gecachte Daten ab.

**`async set(key, data, ttl=3600, cache_level="auto")`**
Speichert Daten im Cache.

**`get_cache_stats()`**
Cache-Leistungsstatistiken.

---

## 🔧 Konfiguration

### Umgebungsvariablen
```bash
# GPU-Konfiguration
SPLEETER_ENABLE_GPU=true
SPLEETER_GPU_MEMORY_GROWTH=true

# Cache-Konfiguration
SPLEETER_CACHE_DIR=/var/cache/spleeter
SPLEETER_REDIS_URL=redis://localhost:6379/0

# Modell-Konfiguration
SPLEETER_MODELS_DIR=/opt/spleeter/models
SPLEETER_AUTO_DOWNLOAD=true

# Überwachung
SPLEETER_ENABLE_MONITORING=true
SPLEETER_METRICS_PORT=9090
```

### Konfigurationsdatei (config.yaml)
```yaml
spleeter:
  performance:
    enable_gpu: true
    batch_size: 8
    worker_threads: 4
    memory_limit_mb: 8192
  
  cache:
    enabled: true
    memory_size_mb: 512
    disk_size_mb: 2048
    redis_url: "redis://localhost:6379/0"
    ttl_hours: 24
  
  audio:
    default_sample_rate: 44100
    supported_formats: ["wav", "flac", "mp3", "ogg"]
    enable_preprocessing: true
    normalize_loudness: true
    
  monitoring:
    enabled: true
    export_interval: 60
    health_check_interval: 30
    alert_thresholds:
      memory_usage: 85
      gpu_usage: 90
      error_rate: 5
```

---

## 📊 Leistung & Überwachung

### Echtzeit-Metriken
Das Modul bietet umfassende Überwachungsfähigkeiten:

- **🎯 Verarbeitungsmetriken**: Erfolgsrate, Verarbeitungszeit, Durchsatz
- **💾 Ressourcennutzung**: CPU, Speicher, GPU-Auslastung
- **🔄 Cache-Leistung**: Trefferquoten, Cache-Effizienz
- **🚨 Gesundheitsüberwachung**: Systemstatus, Fehlerverfolgung, Alarme

### Überwachungs-Dashboard
```python
from spleeter.monitoring import get_stats_summary

# Umfassende Statistiken abrufen
stats = get_stats_summary()
print(f"Erfolgsrate: {stats['processing_stats']['success_rate']}%")
print(f"Cache-Trefferquote: {stats['processing_stats']['cache_hit_rate']}%")
print(f"Systemgesundheit: {stats['system_health']['status']}")
```

### Leistungsoptimierung-Tipps

1. **🎯 GPU-Nutzung**: GPU-Beschleunigung für 2-3x Geschwindigkeitsverbesserung aktivieren
2. **💾 Caching**: Angemessene Cache-Größen für Ihre Arbeitsbelastung konfigurieren
3. **📦 Batch-Verarbeitung**: Batch-Operationen für mehrere Dateien verwenden
4. **🔧 Modellauswahl**: Angemessene Modellkomplexität vs. Geschwindigkeit wählen
5. **⚡ Vorverarbeitung**: Vorverarbeitung für bessere Trennungsqualität aktivieren

---

## 🛡️ Sicherheit & Best Practices

### Eingabevalidierung
- **Dateiformatvalidierung**: Automatische Formaterkennung und -validierung
- **Größenlimits**: Konfigurierbare Dateigröße und Dauerlimits
- **Pfadsicherheit**: Schutz vor Path-Traversal-Angriffen
- **Ressourcenlimits**: Speicher- und Verarbeitungszeit-Beschränkungen

### Fehlerbehandlung
```python
from spleeter.exceptions import (
    AudioProcessingError, 
    ModelError, 
    ValidationError
)

try:
    result = await engine.separate("audio.wav", "spleeter:2stems-16kHz")
except AudioProcessingError as e:
    print(f"Audio-Verarbeitung fehlgeschlagen: {e}")
    print(f"Fehlerkontext: {e.context}")
except ModelError as e:
    print(f"Modellfehler: {e}")
except ValidationError as e:
    print(f"Validierung fehlgeschlagen: {e}")
```

---

## 📈 Skalierbarkeit & Produktion

### Horizontale Skalierung
```python
# Multi-Instanz-Setup
from spleeter import SpleeterCluster

cluster = SpleeterCluster(
    nodes=["worker1:8080", "worker2:8080", "worker3:8080"],
    load_balancer="round_robin",
    shared_cache="redis://cache-cluster:6379"
)

# Verteilte Verarbeitung
result = await cluster.separate_distributed(
    audio_files=["lied1.wav", "lied2.wav", "lied3.wav"],
    model_name="spleeter:4stems-16kHz"
)
```

### Docker-Deployment
```dockerfile
FROM tensorflow/tensorflow:2.8.0-gpu

COPY . /app/spleeter
WORKDIR /app

RUN pip install -r requirements.txt
CMD ["python", "-m", "spleeter.server"]
```

### Kubernetes-Konfiguration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spleeter-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spleeter
  template:
    metadata:
      labels:
        app: spleeter
    spec:
      containers:
      - name: spleeter
        image: spleeter:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
```

---

## 🧪 Tests & Qualitätssicherung

### Tests ausführen
```bash
# Unit-Tests
python -m pytest tests/test_core.py -v

# Integrationstests
python -m pytest tests/test_integration.py -v

# Leistungstests
python -m pytest tests/test_performance.py --benchmark-only

# Vollständige Test-Suite
python -m pytest tests/ --cov=spleeter --cov-report=html
```

### Qualitätsmetriken
- **✅ Code-Abdeckung**: >95%
- **🎯 Leistung**: <2x Echtzeit-Verarbeitung
- **🛡️ Sicherheit**: OWASP-Konformität
- **📊 Zuverlässigkeit**: 99.9% Uptime-Ziel

---

## 🤝 Beitragen

### Entwicklungssetup
```bash
# Entwicklungsinstallation
git clone https://github.com/spotify-ai-agent/backend.git
cd backend/spleeter

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate

# Entwicklungsabhängigkeiten installieren
pip install -r requirements-dev.txt

# Pre-commit-Hooks installieren
pre-commit install
```

### Code-Standards
- **Stil**: Black-Formatierung, PEP 8-Konformität
- **Type Hints**: Umfassende Typ-Annotationen
- **Dokumentation**: Docstring-Abdeckung >90%
- **Tests**: Test-Abdeckung >95%

---

## 📋 Änderungsprotokoll

### v2.0.0 (Aktuell)
- ✨ Vollständige Enterprise-Architektur-Neuentwicklung
- 🚀 Async/await-Unterstützung durchgehend
- 💾 Mehrstufiges Caching-System
- 📊 Umfassende Überwachung
- 🛡️ Verbesserte Sicherheit und Validierung
- 🎯 GPU-Optimierungsverbesserungen
- 📦 Batch-Verarbeitungsfähigkeiten

### v1.x (Legacy)
- Grundlegende Spleeter-Integration
- Nur synchrone Verarbeitung
- Begrenzte Fehlerbehandlung

---

## 📞 Support & Kontakt

### Dokumentation
- **📖 Vollständige Dokumentation**: [docs.spotify-ai-agent.com](https://docs.spotify-ai-agent.com)
- **🔧 API-Referenz**: [api.spotify-ai-agent.com](https://api.spotify-ai-agent.com)
- **💡 Beispiele**: [github.com/spotify-ai-agent/examples](https://github.com/spotify-ai-agent/examples)

### Community
- **💬 Discord**: [discord.gg/spotify-ai](https://discord.gg/spotify-ai)
- **📧 E-Mail**: support@spotify-ai-agent.com
- **🐛 Issues**: [GitHub Issues](https://github.com/spotify-ai-agent/backend/issues)

### Enterprise-Support
Für Enterprise-Kunden ist dedizierter Support mit SLA-Garantien verfügbar.

---

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die [LICENSE](LICENSE)-Datei für Details.

---

## 🙏 Danksagungen

- **Spleeter-Team**: Original-Spleeter-Bibliothek von Deezer Research
- **TensorFlow-Team**: ML-Framework und GPU-Optimierung
- **Open-Source-Community**: Verschiedene Audio-Verarbeitungsbibliotheken

---

**⭐ Wenn Sie dieses Modul nützlich finden, ziehen Sie bitte in Betracht, das Repository zu bewerten!**

---

*Mit ❤️ erstellt vom Spotify AI Agent Enterprise Team*  
*Hauptarchitekt: [Fahed Mlaiel](https://github.com/fahed-mlaiel)*
