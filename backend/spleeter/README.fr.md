# 🎵 Agent IA Spotify - Module Spleeter Avancé

[![Grade Entreprise](https://img.shields.io/badge/Entreprise-Grade-gold.svg)](https://github.com/spotify-ai-agent)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Licence](https://img.shields.io/badge/Licence-MIT-green.svg)](LICENSE)

## 🚀 Aperçu

**Module Spleeter Avancé Entreprise** - Moteur de séparation audio de niveau industriel conçu pour des performances élevées et une séparation de sources musicales évolutive. Ce module fournit des capacités de traitement audio alimentées par IA de pointe avec des fonctionnalités entreprise incluant le cache multi-niveaux, l'optimisation GPU, le traitement par lots et une surveillance complète.

**🎖️ Architecte & Développeur Principal :** [Fahed Mlaiel](https://github.com/fahed-mlaiel)  
**🏢 Organisation :** Équipe Enterprise AI Solutions

---

## ✨ Fonctionnalités Clés

### 🎯 **Capacités Principales**
- **🤖 Modèles IA Avancés** : Support pour séparation 2-stems, 4-stems et 5-stems
- **⚡ Accélération GPU** : Traitement GPU CUDA/TensorFlow optimisé
- **🔄 Traitement Asynchrone** : Opérations non-bloquantes avec asyncio
- **📦 Opérations par Lots** : Traitement parallèle haute performance
- **🎛️ Prétraitement Audio** : Filtrage avancé, normalisation, réduction de bruit

### 🏗️ **Architecture Entreprise**
- **💾 Cache Multi-Niveaux** : Mémoire (L1) → Disque+SQLite (L2) → Redis (L3)
- **📊 Surveillance Performance** : Métriques temps réel, vérifications santé, alertes
- **🔧 Gestion Modèles** : Téléchargement automatique, validation, versioning
- **🛡️ Sécurité & Validation** : Sanitisation des entrées, limites ressources, gestion erreurs
- **📈 Évolutivité** : Prêt microservices, support mise à l'échelle horizontale

### 🎵 **Excellence Traitement Audio**
- **📻 Support Formats** : WAV, FLAC, MP3, OGG, M4A, AAC, WMA
- **🔊 Options Qualité** : Traitement jusqu'à 192kHz/32-bit
- **🎚️ Traitement Dynamique** : Normalisation volume, détection silences
- **📋 Extraction Métadonnées** : Analyse audio complète et étiquetage

---

## 🛠️ Installation

### Prérequis
```bash
# Python 3.8+ requis
python --version  # Devrait être 3.8+

# Installer dépendances principales
pip install tensorflow>=2.8.0
pip install librosa>=0.9.0
pip install soundfile>=0.10.0
pip install numpy>=1.21.0
pip install asyncio-mqtt
pip install aioredis
```

### Configuration Rapide
```bash
# Cloner le dépôt
git clone https://github.com/spotify-ai-agent/backend.git
cd backend/spleeter

# Installer dépendances
pip install -r requirements.txt

# Initialiser le module
python -c "from spleeter import SpleeterEngine; print('✅ Installation réussie!')"
```

---

## 🚀 Démarrage Rapide

### Utilisation Basique
```python
import asyncio
from spleeter import SpleeterEngine

async def separer_audio():
    # Initialiser moteur
    engine = SpleeterEngine()
    await engine.initialize()
    
    # Séparer audio
    result = await engine.separate(
        audio_path="chanson.wav",
        model_name="spleeter:2stems-16kHz",
        output_dir="sortie/"
    )
    
    print(f"✅ Séparation terminée : {result.output_files}")

# Exécuter
asyncio.run(separer_audio())
```

### Configuration Avancée
```python
from spleeter import SpleeterEngine, SpleeterConfig

# Configuration entreprise
config = SpleeterConfig(
    # Performance
    enable_gpu=True,
    batch_size=8,
    worker_threads=4,
    
    # Cache
    cache_enabled=True,
    cache_size_mb=2048,
    redis_url="redis://localhost:6379",
    
    # Paramètres audio
    default_sample_rate=44100,
    enable_preprocessing=True,
    normalize_loudness=True,
    
    # Surveillance
    enable_monitoring=True,
    metrics_export_interval=60
)

engine = SpleeterEngine(config=config)
```

---

## 📚 Référence API

### SpleeterEngine
Moteur de traitement principal avec capacités entreprise.

#### Méthodes

**`async initialize()`**
Initialise le moteur et charge les modèles.

**`async separate(audio_path, model_name, output_dir, **options)`**
Sépare l'audio en stems.
- `audio_path` : Chemin fichier audio d'entrée
- `model_name` : Identifiant modèle (ex : "spleeter:2stems-16kHz")
- `output_dir` : Répertoire sortie pour stems séparés
- `options` : Options traitement additionnelles

**`async batch_separate(audio_files, **options)`**
Traite plusieurs fichiers en parallèle.

**`get_available_models()`**
Liste tous les modèles de séparation disponibles.

**`get_processing_stats()`**
Récupère statistiques de performance.

### ModelManager
Système de gestion avancée des modèles.

#### Méthodes

**`async download_model(model_name, force=False)`**
Télécharge et met en cache un modèle spécifique.

**`list_local_models()`**
Liste les modèles disponibles localement.

**`validate_model(model_name)`**
Vérifie l'intégrité du modèle.

### CacheManager
Système de cache multi-niveaux.

#### Méthodes

**`async get(key, cache_level="auto")`**
Récupère données en cache.

**`async set(key, data, ttl=3600, cache_level="auto")`**
Stocke données dans le cache.

**`get_cache_stats()`**
Statistiques performance du cache.

---

## 🔧 Configuration

### Variables d'Environnement
```bash
# Configuration GPU
SPLEETER_ENABLE_GPU=true
SPLEETER_GPU_MEMORY_GROWTH=true

# Configuration Cache
SPLEETER_CACHE_DIR=/var/cache/spleeter
SPLEETER_REDIS_URL=redis://localhost:6379/0

# Configuration Modèles
SPLEETER_MODELS_DIR=/opt/spleeter/models
SPLEETER_AUTO_DOWNLOAD=true

# Surveillance
SPLEETER_ENABLE_MONITORING=true
SPLEETER_METRICS_PORT=9090
```

### Fichier Configuration (config.yaml)
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

## 📊 Performance & Surveillance

### Métriques Temps Réel
Le module fournit des capacités de surveillance complètes :

- **🎯 Métriques Traitement** : Taux de succès, temps de traitement, débit
- **💾 Utilisation Ressources** : CPU, mémoire, utilisation GPU
- **🔄 Performance Cache** : Taux de réussite, efficacité cache
- **🚨 Surveillance Santé** : État système, suivi erreurs, alertes

### Tableau de Bord Surveillance
```python
from spleeter.monitoring import get_stats_summary

# Obtenir statistiques complètes
stats = get_stats_summary()
print(f"Taux de Succès : {stats['processing_stats']['success_rate']}%")
print(f"Taux Réussite Cache : {stats['processing_stats']['cache_hit_rate']}%")
print(f"Santé Système : {stats['system_health']['status']}")
```

### Conseils Optimisation Performance

1. **🎯 Utilisation GPU** : Activer accélération GPU pour amélioration vitesse 2-3x
2. **💾 Cache** : Configurer tailles cache appropriées pour votre charge
3. **📦 Traitement Lots** : Utiliser opérations par lots pour plusieurs fichiers
4. **🔧 Sélection Modèle** : Choisir complexité modèle appropriée vs vitesse
5. **⚡ Prétraitement** : Activer prétraitement pour meilleure qualité séparation

---

## 🛡️ Sécurité & Bonnes Pratiques

### Validation Entrées
- **Validation Format Fichier** : Détection et validation automatique format
- **Limites Taille** : Limites configurables taille fichier et durée
- **Sécurité Chemins** : Protection contre attaques traversée chemins
- **Limites Ressources** : Contraintes mémoire et temps traitement

### Gestion Erreurs
```python
from spleeter.exceptions import (
    AudioProcessingError, 
    ModelError, 
    ValidationError
)

try:
    result = await engine.separate("audio.wav", "spleeter:2stems-16kHz")
except AudioProcessingError as e:
    print(f"Échec traitement audio : {e}")
    print(f"Contexte erreur : {e.context}")
except ModelError as e:
    print(f"Erreur modèle : {e}")
except ValidationError as e:
    print(f"Échec validation : {e}")
```

---

## 📈 Évolutivité & Production

### Mise à l'Échelle Horizontale
```python
# Configuration multi-instances
from spleeter import SpleeterCluster

cluster = SpleeterCluster(
    nodes=["worker1:8080", "worker2:8080", "worker3:8080"],
    load_balancer="round_robin",
    shared_cache="redis://cache-cluster:6379"
)

# Traitement distribué
result = await cluster.separate_distributed(
    audio_files=["chanson1.wav", "chanson2.wav", "chanson3.wav"],
    model_name="spleeter:4stems-16kHz"
)
```

### Déploiement Docker
```dockerfile
FROM tensorflow/tensorflow:2.8.0-gpu

COPY . /app/spleeter
WORKDIR /app

RUN pip install -r requirements.txt
CMD ["python", "-m", "spleeter.server"]
```

### Configuration Kubernetes
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

## 🧪 Tests & Assurance Qualité

### Exécution Tests
```bash
# Tests unitaires
python -m pytest tests/test_core.py -v

# Tests intégration
python -m pytest tests/test_integration.py -v

# Tests performance
python -m pytest tests/test_performance.py --benchmark-only

# Suite tests complète
python -m pytest tests/ --cov=spleeter --cov-report=html
```

### Métriques Qualité
- **✅ Couverture Code** : >95%
- **🎯 Performance** : <2x traitement temps réel
- **🛡️ Sécurité** : Conformité OWASP
- **📊 Fiabilité** : Objectif uptime 99.9%

---

## 🤝 Contribution

### Configuration Développement
```bash
# Installation développement
git clone https://github.com/spotify-ai-agent/backend.git
cd backend/spleeter

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Installer dépendances développement
pip install -r requirements-dev.txt

# Installer hooks pre-commit
pre-commit install
```

### Standards Code
- **Style** : Formatage Black, conformité PEP 8
- **Annotations Type** : Annotations type complètes
- **Documentation** : Couverture docstring >90%
- **Tests** : Couverture tests >95%

---

## 📋 Journal des Modifications

### v2.0.0 (Actuel)
- ✨ Réécriture complète architecture entreprise
- 🚀 Support async/await partout
- 💾 Système cache multi-niveaux
- 📊 Surveillance complète
- 🛡️ Sécurité et validation améliorées
- 🎯 Améliorations optimisation GPU
- 📦 Capacités traitement par lots

### v1.x (Héritage)
- Intégration Spleeter basique
- Traitement synchrone uniquement
- Gestion erreurs limitée

---

## 📞 Support & Contact

### Documentation
- **📖 Documentation Complète** : [docs.spotify-ai-agent.com](https://docs.spotify-ai-agent.com)
- **🔧 Référence API** : [api.spotify-ai-agent.com](https://api.spotify-ai-agent.com)
- **💡 Exemples** : [github.com/spotify-ai-agent/examples](https://github.com/spotify-ai-agent/examples)

### Communauté
- **💬 Discord** : [discord.gg/spotify-ai](https://discord.gg/spotify-ai)
- **📧 Email** : support@spotify-ai-agent.com
- **🐛 Issues** : [GitHub Issues](https://github.com/spotify-ai-agent/backend/issues)

### Support Entreprise
Pour les clients entreprise, support dédié disponible avec garanties SLA.

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour détails.

---

## 🙏 Remerciements

- **Équipe Spleeter** : Bibliothèque Spleeter originale par Deezer Research
- **Équipe TensorFlow** : Framework ML et optimisation GPU
- **Communauté Open Source** : Diverses bibliothèques traitement audio

---

**⭐ Si vous trouvez ce module utile, pensez à étoiler le dépôt !**

---

*Construit avec ❤️ par l'Équipe Entreprise Agent IA Spotify*  
*Architecte Principal : [Fahed Mlaiel](https://github.com/fahed-mlaiel)*
