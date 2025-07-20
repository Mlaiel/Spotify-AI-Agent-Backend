# 🧪 Tests Frameworks - Guide d'Utilisation
# ==========================================

## 📁 Structure des Tests

```
backend/tests_backend/app/frameworks/
├── __init__.py                 # Configuration pytest globale
├── conftest.py                # Fixtures et utilitaires de test  
├── run_tests.sh              # Script d'exécution automatisé
├── test_core.py              # Tests Framework Orchestrator (400+ lignes)
├── test_hybrid_backend.py    # Tests Django/FastAPI hybride (600+ lignes)
├── test_ml_frameworks.py     # Tests ML/IA Spotify (800+ lignes)
├── test_security.py          # Tests sécurité enterprise (700+ lignes)
├── test_monitoring.py        # Tests monitoring/observabilité (800+ lignes)
├── test_microservices.py     # Tests architecture microservices (900+ lignes)
└── test_integration.py       # Tests intégration bout-en-bout (600+ lignes)
```

## 🚀 Exécution Rapide

### Tests par Mode
```bash
# Tests unitaires seulement
./run_tests.sh unit

# Tests d'intégration
./run_tests.sh integration

# Tests de performance
./run_tests.sh performance

# Tous les tests
./run_tests.sh all

# Tests avec couverture
./run_tests.sh coverage

# Tests rapides (développement)
./run_tests.sh fast
```

### Tests par Framework
```bash
# Framework Core Orchestrator
./run_tests.sh framework core

# Framework Hybrid Backend
./run_tests.sh framework hybrid

# Framework ML/IA
./run_tests.sh framework ml

# Framework Sécurité
./run_tests.sh framework security

# Framework Monitoring
./run_tests.sh framework monitoring

# Framework Microservices
./run_tests.sh framework microservices
```

### Tests CI/CD
```bash
# Mode CI optimisé
CI=true ./run_tests.sh ci

# Tests rapides pour développement
FAST_TESTS=true ./run_tests.sh fast

# Ignorer tests externes
SKIP_EXTERNAL_TESTS=true ./run_tests.sh all
```

## 📊 Couverture de Test

### Par Framework

| Framework | Tests | Couverture | Fonctionnalités |
|-----------|-------|------------|-----------------|
| **Core Orchestrator** | 15+ tests | >90% | Circuit breaker, health check, lifecycle |
| **Hybrid Backend** | 20+ tests | >85% | Django/FastAPI, middleware, sessions |
| **ML Frameworks** | 25+ tests | >90% | Spotify ML, MFCC, BERT, model versioning |
| **Security** | 25+ tests | >95% | JWT/OAuth2, crypto, rate limiting, audit |
| **Monitoring** | 30+ tests | >85% | Prometheus, Jaeger, alerting, health |
| **Microservices** | 35+ tests | >90% | Service mesh, discovery, load balancing |
| **Integration** | 20+ tests | >80% | End-to-end, business scenarios |

### Types de Tests

#### 🔬 Tests Unitaires
- Tests de composants isolés
- Mocking des dépendances externes
- Validation logique métier
- Tests de cas limites

#### 🔗 Tests d'Intégration
- Communication inter-frameworks
- Workflows bout-en-bout
- Scénarios business réels
- Tests de résilience

#### ⚡ Tests de Performance
- Benchmarks de latence
- Tests de charge
- Optimisation mémoire
- Scalabilité horizontale

#### 🛡️ Tests de Sécurité
- Authentification/autorisation
- Chiffrement/déchiffrement
- Rate limiting
- Audit de sécurité

## 🔧 Configuration

### Variables d'Environnement
```bash
# Mode développement
export TEST_MODE="development"
export FAST_TESTS="true"
export SKIP_EXTERNAL_TESTS="true"

# Mode CI/CD
export CI="true"
export TEST_WORKERS="4"
export COVERAGE_THRESHOLD="85"

# Services externes
export REDIS_URL="redis://localhost:6379/15"
export DATABASE_URL="sqlite:///test_frameworks.db"
```

### Markers pytest
```python
@pytest.mark.unit          # Test unitaire
@pytest.mark.integration   # Test d'intégration
@pytest.mark.performance   # Test de performance
@pytest.mark.slow          # Test lent (>1s)
@pytest.mark.external      # Nécessite services externes
@pytest.mark.security      # Test de sécurité
@pytest.mark.ml            # Test ML/IA
```

## 📈 Métriques et Rapports

### Rapports Générés
```
coverage_reports/
├── html/                   # Rapport HTML interactif
├── coverage.json          # Données JSON
└── coverage.xml           # Format XML (CI)

test_logs/
├── unit_tests.log         # Logs tests unitaires
├── integration_tests.log  # Logs intégration  
├── performance_tests.log  # Logs performance
├── benchmark_results.json # Résultats benchmarks
└── test_report.txt       # Rapport final
```

### Métriques Clés
- **Couverture globale**: >85%
- **Temps d'exécution**: <5min (fast), <15min (all)
- **Taux de succès**: >95%
- **Performance**: <100ms latence moyenne

## 🎯 Scénarios de Test Spécialisés

### 🤖 Expert ML Engineer - Tests ML
```python
# Tests modèles Spotify
test_spotify_recommendation_model()
test_audio_analysis_mfcc_extraction()
test_bert_sentiment_analysis()
test_model_versioning_mlops()
```

### 🔒 Expert Sécurité - Tests Security
```python
# Tests authentification
test_jwt_token_lifecycle()
test_oauth2_spotify_integration()
test_rate_limiting_redis()
test_encryption_fernet_rsa()
```

### 📊 Expert DevOps - Tests Monitoring
```python
# Tests observabilité
test_prometheus_metrics_collection()
test_jaeger_distributed_tracing()
test_intelligent_alerting()
test_health_monitoring_concurrent()
```

### 🏗️ Expert Architecture - Tests Microservices
```python
# Tests architecture distribuée
test_service_discovery_multi_backend()
test_load_balancing_weighted()
test_service_mesh_mtls()
test_message_broker_patterns()
```

## 🚀 Workflows Recommandés

### Développement Local
```bash
# 1. Tests rapides pendant développement
./run_tests.sh fast

# 2. Tests framework spécifique
./run_tests.sh framework [nom_framework]

# 3. Tests complets avant commit
./run_tests.sh coverage
```

### Pipeline CI/CD
```bash
# 1. Tests rapides (validation PR)
CI=true ./run_tests.sh fast

# 2. Tests complets (merge main)
CI=true ./run_tests.sh ci

# 3. Tests performance (release)
./run_tests.sh performance
```

### Debug et Troubleshooting
```bash
# Tests verbeux avec détails
./run_tests.sh unit --verbose -s

# Test spécifique avec debug
pytest test_core.py::TestFrameworkOrchestrator::test_lifecycle -vvv -s

# Tests avec profiling
./run_tests.sh performance --benchmark-json=results.json
```

## 🔍 Fixtures et Mocks Disponibles

### Fixtures Globales (conftest.py)
- `clean_frameworks`: Nettoyage entre tests
- `mock_redis`: Redis mocké
- `mock_consul`: Consul mocké  
- `mock_http_client`: Client HTTP mocké
- `sample_audio_data`: Données audio test
- `test_user_data`: Données utilisateur test

### Mocks Spécialisés
- `MockSpotifyAPI`: API Spotify mockée
- `MockPrometheusRegistry`: Registre Prometheus mocké
- `MockJaegerTracer`: Traceur Jaeger mocké
- `MockMLModel`: Modèles ML mockés

## 📚 Ressources et Documentation

### Documentation Interne
- Chaque test file contient sa documentation
- Exemples d'usage dans les docstrings
- Patterns de test recommandés

### Standards de Qualité
- Couverture minimale: 85%
- Tests atomiques et idempotents
- Nommage descriptif et cohérent
- Documentation des cas complexes

---

**Développé par l'équipe d'experts Spotify AI Agent** 🎵🤖
