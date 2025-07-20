# 🎵 Spotify AI Agent - README Tests Spleeter
# ============================================
# 
# Guide complet pour l'exécution et la maintenance
# des tests du module Spleeter enterprise.
#
# 🎖️ Développé par l'équipe d'experts enterprise

# 🎵 Tests Spleeter - Guide Complet

Bienvenue dans la suite de tests complète du module **Spleeter** de l'agent IA Spotify ! 🚀

Cette documentation vous guidera à travers l'installation, la configuration et l'exécution de tous les types de tests disponibles.

## 📋 Table des Matières

- [🎯 Vue d'Ensemble](#-vue-densemble)
- [🛠️ Installation](#️-installation)
- [🚀 Démarrage Rapide](#-démarrage-rapide)
- [📊 Types de Tests](#-types-de-tests)
- [⚙️ Configuration](#️-configuration)
- [🔧 Commandes Makefile](#-commandes-makefile)
- [🐍 Script Python](#-script-python)
- [📈 Couverture de Code](#-couverture-de-code)
- [🏭 CI/CD](#-cicd)
- [🔍 Debugging](#-debugging)
- [📝 Contribuer](#-contribuer)

## 🎯 Vue d'Ensemble

La suite de tests Spleeter comprend **11 modules de test** couvrant tous les aspects du système :

### 📁 Structure des Tests

```
tests_backend/spleeter/
├── 📋 conftest.py              # Configuration globale & fixtures
├── 🧪 test_core.py             # Tests moteur principal
├── 🤖 test_models.py           # Tests gestion modèles
├── 🎵 test_processor.py        # Tests traitement audio
├── 💾 test_cache.py            # Tests système cache
├── 🔧 test_utils.py            # Tests utilitaires
├── 📊 test_monitoring.py       # Tests monitoring
├── ⚠️ test_exceptions.py       # Tests gestion erreurs
├── 🔗 test_integration.py      # Tests d'intégration
├── ⚡ test_performance.py      # Tests performance
├── 🛠️ test_helpers.py          # Utilitaires de test
├── 📋 Makefile                 # Automatisation make
├── 🐍 run_tests.sh             # Script bash automation
├── ⚙️ pyproject.toml           # Configuration pytest
└── 📖 README.md                # Cette documentation
```

### 🎖️ Couverture de Test

- **Tests Unitaires** : Validation de chaque composant individuellement
- **Tests d'Intégration** : Validation des workflows complets
- **Tests de Performance** : Benchmarks et optimisations
- **Tests de Stress** : Validation sous charge élevée
- **Tests de Sécurité** : Validation des contrôles de sécurité
- **Tests de Régression** : Prévention des régressions

## 🛠️ Installation

### Prérequis Système

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg libsndfile1

# macOS
brew install python ffmpeg libsndfile

# Windows (avec chocolatey)
choco install python ffmpeg
```

### Installation des Dépendances Python

```bash
# Installation automatique via le script
cd /workspaces/Achiri/spotify-ai-agent/backend/tests_backend/spleeter
chmod +x run_tests.sh
./run_tests.sh --install-deps

# Ou installation manuelle
pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark
pip install black flake8 mypy isort coverage[toml]
pip install pytest-html pytest-xdist pytest-timeout
```

### Installation via Makefile

```bash
# Installation complète des dépendances
make install-deps

# Vérification de la configuration
make show-config
```

## 🚀 Démarrage Rapide

### Tests Rapides (< 1 minute)

```bash
# Via Makefile
make test-fast

# Via script bash
./run_tests.sh --smoke

# Via pytest direct
pytest -m "not slow and not performance and not stress" --timeout=30
```

### Tests Complets

```bash
# Pipeline CI/CD complet
./run_tests.sh --ci

# Tous les tests via Makefile
make test-all

# Tests avec couverture
make coverage-html
```

### Tests Spécifiques

```bash
# Tests par module
make test-core          # Moteur principal
make test-models        # Gestion modèles
make test-processor     # Traitement audio
make test-cache         # Système cache

# Tests par type
make test-unit          # Tests unitaires
make test-integration   # Tests d'intégration
make test-performance   # Tests de performance
```

## 📊 Types de Tests

### 🧪 Tests Unitaires

Validation individuelle de chaque composant :

```bash
# Exécution des tests unitaires
pytest test_core.py test_models.py test_processor.py \
       test_cache.py test_utils.py test_monitoring.py test_exceptions.py \
       -v -m "not slow"
```

**Couverture** :
- ✅ Initialisation des engines
- ✅ Séparation audio basique
- ✅ Gestion des modèles ML
- ✅ Cache multi-niveaux
- ✅ Utilitaires et validation
- ✅ Monitoring système
- ✅ Gestion d'exceptions

### 🔗 Tests d'Intégration

Validation des workflows end-to-end :

```bash
# Tests d'intégration
pytest test_integration.py -v -m "integration" --timeout=120
```

**Scénarios** :
- ✅ Séparation complète avec cache
- ✅ Traitement batch avec monitoring
- ✅ Gestion d'erreurs robuste
- ✅ Workflows concurrents

### ⚡ Tests de Performance

Benchmarks et optimisations :

```bash
# Tests de performance
pytest test_performance.py -v -m "performance" --benchmark-only
```

**Métriques** :
- ⏱️ Temps de séparation audio
- 📊 Utilisation mémoire
- 🚀 Throughput batch
- 💾 Performance cache
- 📈 Monitoring overhead

### 💪 Tests de Stress

Validation sous charge élevée :

```bash
# Tests de stress
pytest test_performance.py -v -m "stress" --timeout=600
```

**Scénarios** :
- 🔥 Charge CPU maximale
- 💧 Stress mémoire
- 🌊 Traitement concurrent
- 🎯 Stabilité longue durée

## ⚙️ Configuration

### Configuration pytest (pyproject.toml)

La configuration centralisée dans `pyproject.toml` inclut :

```toml
[tool.pytest.ini_options]
markers = [
    "slow: tests lents",
    "fast: tests rapides", 
    "unit: tests unitaires",
    "integration: tests d'intégration",
    "performance: tests de performance",
    "stress: tests de stress"
]

addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--durations=10",
    "--maxfail=3"
]
```

### Variables d'Environnement

```bash
export SPLEETER_TEST_MODE="true"
export SPLEETER_LOG_LEVEL="DEBUG"
export SPLEETER_CACHE_DISABLED="true"
export COVERAGE_MIN="85"
```

### Configuration Coverage

```toml
[tool.coverage.run]
source = ["backend/spleeter"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 85
show_missing = true
```

## 🔧 Commandes Makefile

### Commandes Principales

```bash
make help              # Affichage de l'aide complète
make test              # Tests de base (rapides)
make test-all          # Tous les tests (complet)
make coverage          # Analyse de couverture console
make coverage-html     # Rapport HTML de couverture
make lint              # Vérifications linting
make format            # Formatage automatique
make clean             # Nettoyage fichiers temporaires
```

### Tests par Module

```bash
make test-core         # Tests moteur principal
make test-models       # Tests gestion modèles  
make test-processor    # Tests traitement audio
make test-cache        # Tests système cache
make test-utils        # Tests utilitaires
make test-monitoring   # Tests monitoring
make test-exceptions   # Tests gestion erreurs
```

### Tests par Type

```bash
make test-unit         # Tests unitaires
make test-integration  # Tests d'intégration
make test-performance  # Tests de performance
make test-stress       # Tests de stress
make test-fast         # Tests rapides seulement
make test-slow         # Tests lents seulement
```

### Utilitaires Avancés

```bash
make test-parallel     # Tests en parallèle
make test-report       # Tests avec rapport HTML
make benchmark         # Benchmarks spécifiques
make test-security     # Tests de sécurité
make test-regression   # Tests de régression
make smoke-test        # Tests de smoke
make acceptance-test   # Tests d'acceptation
```

## 🐍 Script Python

### Utilisation du Script

```bash
# Rendre le script exécutable
chmod +x run_tests.sh

# Aide complète
./run_tests.sh --help

# Pipeline CI/CD complet
./run_tests.sh --ci

# Tests spécifiques
./run_tests.sh --unit
./run_tests.sh --integration
./run_tests.sh --performance
./run_tests.sh --stress
```

### Options Disponibles

| Option | Description |
|--------|-------------|
| `--ci` | Pipeline CI/CD complet |
| `--unit` | Tests unitaires uniquement |
| `--integration` | Tests d'intégration uniquement |
| `--performance` | Tests de performance uniquement |
| `--stress` | Tests de stress uniquement |
| `--coverage` | Analyse de couverture uniquement |
| `--quality` | Vérifications de qualité uniquement |
| `--smoke` | Tests de smoke (vérification rapide) |
| `--install-deps` | Installation des dépendances |
| `--cleanup` | Nettoyage des fichiers temporaires |

### Configuration du Script

```bash
# Variables d'environnement pour personnalisation
export COVERAGE_MIN=90           # Couverture minimale
export TIMEOUT_UNIT=45          # Timeout tests unitaires
export TIMEOUT_INTEGRATION=180  # Timeout tests intégration
export TIMEOUT_PERFORMANCE=450  # Timeout tests performance
export TIMEOUT_STRESS=900       # Timeout tests stress
```

## 📈 Couverture de Code

### Génération des Rapports

```bash
# Rapport console
make coverage

# Rapport HTML détaillé
make coverage-html
# Ouverture automatique: coverage_html/index.html

# Rapport XML (pour CI/CD)
make coverage-xml
# Fichier généré: coverage.xml
```

### Seuils de Couverture

- **Couverture Globale** : ≥ 85%
- **Couverture par Fichier** : ≥ 80%
- **Couverture Branches** : ≥ 75%

### Analyse de la Couverture

```bash
# Couverture différentielle
make diff-coverage

# Couverture avec contexte
pytest --cov=../../spleeter \
       --cov-context=test \
       --cov-branch
```

## 🏭 CI/CD

### GitHub Actions

Le workflow GitHub Actions inclut :

```yaml
# .github/workflows/spleeter-tests.yml
name: 🎵 Spleeter Tests CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * *'  # Tests quotidiens

jobs:
  smoke-tests:    # Tests de vérification rapide
  unit-tests:     # Tests unitaires multi-plateforme
  integration-tests: # Tests d'intégration avec services
  performance-tests: # Tests de performance et benchmarks
  stress-tests:   # Tests de charge et stress
  security-tests: # Tests de sécurité
  code-quality:   # Vérifications qualité de code
  final-report:   # Rapport consolidé
```

### Configuration Locale

```bash
# Simulation locale du pipeline CI/CD
./run_tests.sh --ci

# Vérification pre-commit
make pre-commit

# Validation complète
make check-all
```

### Intégration Continue

```bash
# Tests rapides pour développement
make test-fast

# Tests complets pour release
make test-all coverage-html

# Vérification avant commit
make pre-commit
```

## 🔍 Debugging

### Debug de Tests Spécifiques

```bash
# Mode debug verbose
make test-debug TEST=test_core.py::test_engine_initialization

# Debug avec pdb
pytest test_core.py::test_engine_initialization -vvv -s --pdb

# Profiling de performance
make profile-tests
```

### Logs de Debug

```bash
# Activation des logs détaillés
export SPLEETER_LOG_LEVEL="DEBUG"
pytest test_core.py -vvv -s --log-cli-level=DEBUG
```

### Tests en Mode Watch

```bash
# Mode watch pour développement
make watch-tests

# Tests continus avec pytest-watch
pip install pytest-watch
ptw -- -m "not slow"
```

### Debugging d'Échecs

```bash
# Tests avec informations d'échec étendues
pytest --tb=long --capture=no

# Génération de rapports d'échec
pytest --html=failure_report.html --self-contained-html
```

## 📊 Métriques et Monitoring

### Métriques Collectées

- **Performance** : Temps d'exécution, throughput
- **Ressources** : CPU, mémoire, I/O
- **Qualité** : Couverture, complexité
- **Fiabilité** : Taux de réussite, stabilité

### Rapports Disponibles

```bash
# Rapport HTML complet
make test-report

# Métriques de performance
make test-metrics

# Export JSON des résultats
make test-export
```

### Monitoring Continue

```bash
# Tests avec monitoring des ressources
pytest test_performance.py::TestResourceMonitoring -v

# Suivi de la mémoire
pytest --memray

# Profiling CPU
pytest --profile
```

## 🔧 Maintenance

### Mise à Jour des Dépendances

```bash
# Vérification des versions
pip list --outdated

# Mise à jour sécurisée
pip install --upgrade -r requirements/testing.txt

# Vérification de compatibilité
./run_tests.sh --smoke
```

### Nettoyage Périodique

```bash
# Nettoyage complet
make clean

# Suppression des caches
rm -rf .pytest_cache __pycache__ .coverage

# Nettoyage des artifacts
rm -rf coverage_html test_report.html
```

### Optimisation des Tests

```bash
# Tests en parallèle
make test-parallel

# Optimisation des fixtures
pytest --setup-show

# Analyse des durées
pytest --durations=0
```

## 🔐 Sécurité

### Tests de Sécurité

```bash
# Tests de sécurité spécifiques
make test-security

# Scan de vulnérabilités
bandit -r ../../spleeter/

# Vérification des dépendances
safety check
```

### Validation des Entrées

```bash
# Tests de validation
pytest -k "validation or sanitize" test_utils.py

# Tests d'injection
pytest -k "security" test_exceptions.py
```

## 📝 Contribuer

### Ajout de Nouveaux Tests

1. **Créer le fichier de test** dans le répertoire approprié
2. **Utiliser les fixtures** définies dans `conftest.py`
3. **Ajouter les marqueurs** appropriés (`@pytest.mark.unit`, etc.)
4. **Documenter les tests** avec des docstrings claires
5. **Vérifier la couverture** avec `make coverage`

### Structure d'un Test

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Tests pour la nouvelle fonctionnalité."""
    
    @pytest.mark.unit
    @pytest.mark.fast
    def test_basic_functionality(self, mock_engine):
        """Test de la fonctionnalité de base."""
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
        """Test d'intégration complet."""
        # Test d'intégration avec vraies dépendances
        pass
```

### Guidelines de Test

- **Nommage** : `test_<functionality>_<scenario>`
- **Isolation** : Chaque test doit être indépendant
- **Mocking** : Utiliser les mocks pour les dépendances externes
- **Assertions** : Assertions claires et spécifiques
- **Documentation** : Docstrings explicatives
- **Marqueurs** : Utilisation appropriée des marqueurs pytest

### Validation Pre-Commit

```bash
# Validation complète avant commit
make pre-commit

# Vérifications individuelles
make format    # Formatage du code
make lint      # Vérifications linting
make type-check # Vérification des types
make test-fast # Tests rapides
```

## 🎯 Objectifs de Qualité

### Cibles de Performance

- **Tests Unitaires** : < 30 secondes total
- **Tests d'Intégration** : < 2 minutes total  
- **Tests de Performance** : < 5 minutes total
- **Séparation Audio** : < 5 secondes/fichier

### Cibles de Couverture

- **Couverture Globale** : ≥ 85%
- **Couverture Branches** : ≥ 75%
- **Modules Critiques** : ≥ 90%

### Cibles de Fiabilité

- **Taux de Réussite** : ≥ 99%
- **Stabilité** : 0 tests flaky
- **Reproductibilité** : 100%

## 🚀 Performances et Optimisations

### Optimisations de Test

```bash
# Tests en parallèle
pytest -n auto

# Cache des résultats
pytest --cache-clear  # Reset cache
pytest --lf          # Last failed only
pytest --ff          # Failed first

# Optimisation mémoire
pytest --maxfail=1 --tb=no
```

### Monitoring des Performances

```bash
# Profiling détaillé
pytest --profile --profile-svg

# Monitoring mémoire
pytest --memray --memray-bin-path=memory_profile.bin

# Analyse des goulots d'étranglement
pytest --durations=20
```

## 📞 Support et Aide

### Ressources

- **Documentation** : Ce README et les docstrings
- **Exemples** : Fichiers de test existants
- **Configuration** : `pyproject.toml`, `conftest.py`
- **Scripts** : `Makefile`, `run_tests.sh`

### Commandes d'Aide

```bash
# Aide Makefile
make help

# Aide script bash
./run_tests.sh --help

# Aide pytest
pytest --help

# Configuration active
make show-config
```

### Dépannage Courant

**Tests qui échouent** :
```bash
# Debug mode verbose
pytest -vvv -s --tb=long

# Tests isolés
pytest test_file.py::test_function -v
```

**Performance lente** :
```bash
# Profiling
make profile-tests

# Tests rapides seulement
make test-fast
```

**Problèmes de couverture** :
```bash
# Rapport détaillé
make coverage-html

# Fichiers manqués
coverage report --show-missing
```

---

## 🎉 Conclusion

Cette suite de tests complète garantit la qualité, la performance et la fiabilité du module Spleeter. Utilisez cette documentation comme référence pour tous vos besoins de test !

**Bon testing ! 🚀🎵**

---

*Développé avec ❤️ par l'équipe d'experts enterprise pour l'agent IA Spotify*
