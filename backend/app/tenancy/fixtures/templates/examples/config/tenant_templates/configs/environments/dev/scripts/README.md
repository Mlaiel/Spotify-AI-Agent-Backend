# =============================================================================
# Documentation Complète des Scripts d'Automatisation - Environnement de Développement
# =============================================================================
# 
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel
# Lead Dev + AI Architect, Senior Backend Developer, ML Engineer,
# DBA & Data Engineer, Backend Security Specialist, Microservices Architect
# =============================================================================

## Vue d'Ensemble

Ce répertoire contient une suite complète de scripts d'automatisation pour l'environnement de développement du projet Spotify AI Agent. Ces scripts industrialisés permettent une gestion complète et professionnelle de l'environnement de développement.

## Architecture des Scripts

### 🚀 **setup_dev.sh**
**Script de Configuration Initiale**

Automatise la configuration complète de l'environnement de développement :

#### Fonctionnalités :
- Installation et configuration de l'environnement virtuel Python
- Installation automatique des dépendances (requirements.txt, requirements-dev.txt)
- Configuration de PostgreSQL avec création de base de données
- Configuration de Redis avec optimisations
- Installation des outils de développement (pre-commit, black, flake8)
- Configuration des variables d'environnement
- Validation de l'installation complète

#### Utilisation :
```bash
./setup_dev.sh                    # Configuration standard
./setup_dev.sh --force            # Reconfiguration forcée
./setup_dev.sh --skip-db          # Sans configuration DB
./setup_dev.sh --skip-redis       # Sans configuration Redis
./setup_dev.sh --minimal          # Installation minimale
```

#### Variables d'environnement supportées :
- `PYTHON_VERSION` : Version de Python (défaut: 3.9)
- `VENV_NAME` : Nom de l'environnement virtuel (défaut: venv)
- `SKIP_VALIDATION` : Ignore les validations (défaut: false)
- `DEBUG` : Mode debug verbose (défaut: false)

---

### 🗄️ **reset_db.sh**
**Script de Gestion de Base de Données**

Script complet pour la gestion, la réinitialisation et la maintenance de la base de données PostgreSQL :

#### Fonctionnalités :
- Sauvegarde automatique avant réinitialisation
- Réinitialisation complète avec recréation de schéma
- Exécution des migrations Alembic
- Chargement des données de développement (fixtures)
- Validation de l'intégrité post-réinitialisation
- Gestion des connexions et transactions

#### Utilisation :
```bash
./reset_db.sh                     # Réinitialisation complète
./reset_db.sh --no-backup         # Sans sauvegarde
./reset_db.sh --data-only          # Données seulement
./reset_db.sh --schema-only        # Schéma seulement
./reset_db.sh --restore backup.sql # Restauration depuis backup
```

#### Sauvegardes automatiques :
- Horodatage : `backup_YYYYMMDD_HHMMSS.sql`
- Compression gzip automatique
- Rétention configurable (défaut: 7 jours)
- Validation de l'intégrité des backups

---

### 🔧 **start_services.sh**
**Script de Démarrage des Services**

Lance tous les services nécessaires au développement avec surveillance et gestion des dépendances :

#### Fonctionnalités :
- Démarrage automatique des services externes (PostgreSQL, Redis)
- Gestion des migrations de base de données
- Lancement du serveur API FastAPI avec hot-reload
- Configuration dynamique des workers et ports
- Mode arrière-plan avec PID tracking
- Services de monitoring (Prometheus optionnel)
- Vérification de santé des services

#### Utilisation :
```bash
./start_services.sh                      # Mode interactif
./start_services.sh --background          # Mode arrière-plan
./start_services.sh --start-db --start-redis  # Avec services locaux
./start_services.sh --update-deps         # Mise à jour des dépendances
./start_services.sh --start-monitoring    # Avec monitoring
./start_services.sh --status              # Statut des services
```

#### Configuration avancée :
- Support du hot-reload pour développement
- Configuration multi-workers pour production
- Gestion des logs rotatifs
- Intégration Prometheus/Grafana
- Health checks automatiques

---

### 📊 **monitor_health.sh**
**Script de Surveillance et Monitoring**

Système complet de surveillance des services avec alerting et rapports :

#### Fonctionnalités :
- Surveillance continue des services (API, PostgreSQL, Redis)
- Collecte de métriques système (CPU, mémoire, disque)
- Tests de performance automatisés
- Génération d'alertes intelligentes
- Tableaux de bord en temps réel
- Rapports quotidiens automatiques
- Archivage et analyse des tendances

#### Utilisation :
```bash
./monitor_health.sh                  # Monitoring continu
./monitor_health.sh --once           # Vérification unique
./monitor_health.sh --dashboard       # Tableau de bord interactif
./monitor_health.sh --performance     # Tests de performance
./monitor_health.sh --status          # Statut actuel
./monitor_health.sh --interval 60     # Intervalle personnalisé
```

#### Métriques surveillées :
- **API** : Temps de réponse, disponibilité, taux d'erreur
- **PostgreSQL** : Connexions actives, requêtes, performances
- **Redis** : Utilisation mémoire, hit rate, opérations
- **Système** : CPU, RAM, espace disque, charge

#### Alerting avancé :
- Seuils configurables par métrique
- Notifications par email/Slack (configurable)
- Escalade automatique des alertes critiques
- Suppression des faux positifs

---

### 📋 **manage_logs.sh**
**Script de Gestion des Logs**

Système industriel de gestion, rotation et analyse des logs :

#### Fonctionnalités :
- Rotation automatique basée sur la taille
- Archivage avec compression gzip
- Analyse intelligente des patterns d'erreurs
- Génération de rapports d'analyse
- Nettoyage automatique des anciens logs
- Détection d'anomalies et tendances
- Export des métriques pour monitoring

#### Utilisation :
```bash
./manage_logs.sh --status            # Statut des logs
./manage_logs.sh --rotate            # Rotation des logs
./manage_logs.sh --analyze-all       # Analyses complètes
./manage_logs.sh --report            # Rapport consolidé
./manage_logs.sh --cleanup           # Nettoyage archives
./manage_logs.sh --full              # Maintenance complète
```

#### Types d'analyses :
- **Erreurs** : Patterns, fréquence, criticité
- **Performance** : Temps de réponse, throughput
- **Accès** : Trafic, endpoints populaires, codes d'erreur
- **Sécurité** : Tentatives d'intrusion, IPs suspectes

---

## Variables d'Environnement Globales

### Configuration de Base
```bash
# Environnement Python
PYTHON_VERSION=3.9
VENV_NAME=venv

# Configuration API
DEV_API_HOST=0.0.0.0
DEV_API_PORT=8000
DEV_API_WORKERS=1
DEV_HOT_RELOAD=true

# Configuration Base de Données
DEV_DB_HOST=localhost
DEV_DB_PORT=5432
DEV_DB_NAME=spotify_ai_dev
DEV_DB_USER=postgres
DEV_DB_PASSWORD=postgres

# Configuration Redis
DEV_REDIS_HOST=localhost
DEV_REDIS_PORT=6379
DEV_REDIS_PASSWORD=

# Configuration Monitoring
MONITORING_INTERVAL=30
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
ALERT_THRESHOLD_RESPONSE_TIME=5000

# Configuration Logs
LOG_RETENTION_DAYS=30
LOG_ARCHIVE_DAYS=7
LOG_MAX_SIZE_MB=100
LOG_ANALYSIS_DAYS=1

# Modes de débogage
DEBUG=true
SKIP_VALIDATION=false
```

## Workflows d'Utilisation

### 🔄 **Workflow de Démarrage Quotidien**
```bash
# 1. Configuration initiale (première fois)
./setup_dev.sh

# 2. Démarrage des services
./start_services.sh --start-db --start-redis

# 3. Monitoring en arrière-plan
./monitor_health.sh --dashboard &

# 4. Vérification du statut
./start_services.sh --status
```

### 🔄 **Workflow de Maintenance Hebdomadaire**
```bash
# 1. Gestion des logs
./manage_logs.sh --full

# 2. Réinitialisation de la DB (si nécessaire)
./reset_db.sh

# 3. Mise à jour des dépendances
./start_services.sh --update-deps

# 4. Test de santé complet
./monitor_health.sh --performance
```

### 🔄 **Workflow de Débogage**
```bash
# 1. Vérification des services
./monitor_health.sh --once

# 2. Analyse des logs d'erreurs
./manage_logs.sh --analyze-errors

# 3. Réinitialisation si nécessaire
./reset_db.sh --no-backup

# 4. Redémarrage propre
./start_services.sh --update-deps
```

## Intégrations et Extensions

### 🔌 **Intégration CI/CD**
Les scripts supportent l'intégration dans les pipelines CI/CD :

```yaml
# Exemple GitLab CI
development_setup:
  script:
    - ./scripts/setup_dev.sh --minimal
    - ./scripts/start_services.sh --background
    - ./scripts/monitor_health.sh --once
```

### 🔌 **Intégration Docker**
Support pour les environnements containerisés :

```bash
# Variables Docker
DOCKER_MODE=true
CONTAINER_NAME=spotify-ai-dev
```

### 🔌 **Intégration Kubernetes**
Configuration pour déploiement K8s :

```bash
# Variables Kubernetes
K8S_NAMESPACE=development
K8S_CONTEXT=dev-cluster
```

## Sécurité et Bonnes Pratiques

### 🔒 **Gestion des Secrets**
- Chiffrement AES-256 pour les secrets sensibles
- Rotation automatique des clés
- Audit trail des accès
- Intégration avec les gestionnaires de secrets

### 🔒 **Validation et Sécurité**
- Validation des entrées utilisateur
- Sanitisation des paramètres
- Logging sécurisé (masquage des secrets)
- Permissions restrictives sur les fichiers

### 🔒 **Monitoring de Sécurité**
- Détection d'anomalies dans les logs
- Surveillance des tentatives d'accès
- Alertes de sécurité automatiques
- Compliance avec les standards industriels

## Support et Maintenance

### 📞 **Dépannage Courant**

#### Problème : Services ne démarrent pas
```bash
# Diagnostic
./monitor_health.sh --once
./manage_logs.sh --analyze-errors

# Solution
./reset_db.sh
./start_services.sh --start-db --start-redis
```

#### Problème : Performances dégradées
```bash
# Analyse
./monitor_health.sh --performance
./manage_logs.sh --analyze-performance

# Optimisation
./manage_logs.sh --cleanup
./start_services.sh --update-deps
```

#### Problème : Logs volumineux
```bash
# Nettoyage
./manage_logs.sh --rotate --cleanup
./manage_logs.sh --analyze-all
```

### 📞 **Logs et Débogage**

Tous les scripts génèrent des logs détaillés dans :
- `$PROJECT_ROOT/logs/` - Logs applicatifs
- `$PROJECT_ROOT/logs/monitoring/` - Logs de surveillance
- `$PROJECT_ROOT/logs/reports/` - Rapports d'analyse

### 📞 **Performance et Optimisation**

Les scripts sont optimisés pour :
- Exécution rapide avec mise en cache
- Utilisation minimale des ressources
- Parallélisation des tâches longues
- Gestion intelligente des dépendances

## Roadmap et Évolutions

### 🚀 **Fonctionnalités Prévues**
- Intégration avec Kubernetes
- Support multi-environnements
- Alerting Slack/Teams
- Dashboard web interactif
- API REST pour contrôle distant

### 🚀 **Améliorations Continues**
- Performance optimizations
- Sécurité renforcée
- Monitoring avancé
- Tests automatisés des scripts

---

## Contribution et Support

**Équipe de Développement :**
- **Fahed Mlaiel** - Lead Dev + AI Architect
- **Senior Backend Developer** - Architecture FastAPI
- **ML Engineer** - Optimisations ML
- **DBA & Data Engineer** - Gestion données
- **Backend Security Specialist** - Sécurité
- **Microservices Architect** - Architecture distribuée

**Contact :** Fahed Mlaiel pour questions techniques et évolutions.

---

*Cette documentation est maintenue automatiquement et mise à jour avec chaque évolution des scripts.*
