# =============================================================================
# Documentation des Scripts d'Automatisation - Environnement de Développement (Français)
# =============================================================================
# 
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel
# Lead Dev + AI Architect, Senior Backend Developer, ML Engineer,
# DBA & Data Engineer, Backend Security Specialist, Microservices Architect
# =============================================================================

## Aperçu Général

Ce répertoire contient une suite complète de scripts d'automatisation pour l'environnement de développement du projet Spotify AI Agent. Ces scripts professionnels permettent une gestion industrialisée et efficace de l'environnement de développement.

## Scripts Disponibles

### 🚀 **setup_dev.sh** - Configuration Initiale
Script de configuration automatisée de l'environnement de développement complet.

**Fonctionnalités principales :**
- Installation automatique de l'environnement virtuel Python
- Configuration des bases de données PostgreSQL et Redis
- Installation des dépendances et outils de développement
- Validation complète de l'installation

**Utilisation typique :**
```bash
./setup_dev.sh                    # Installation standard
./setup_dev.sh --force            # Réinstallation forcée
./setup_dev.sh --minimal          # Installation minimale
```

### 🗄️ **reset_db.sh** - Gestion de Base de Données
Script complet pour la réinitialisation et maintenance de la base de données.

**Fonctionnalités principales :**
- Sauvegarde automatique avant réinitialisation
- Recréation complète du schéma
- Exécution des migrations Alembic
- Chargement des données de test

**Utilisation typique :**
```bash
./reset_db.sh                     # Réinitialisation complète
./reset_db.sh --no-backup         # Sans sauvegarde préalable
./reset_db.sh --data-only          # Données uniquement
```

### 🔧 **start_services.sh** - Démarrage des Services
Lance tous les services nécessaires au développement avec surveillance intégrée.

**Fonctionnalités principales :**
- Démarrage automatique des services (API, PostgreSQL, Redis)
- Gestion des migrations de base de données
- Mode arrière-plan avec suivi des processus
- Vérification de santé des services

**Utilisation typique :**
```bash
./start_services.sh                      # Mode interactif
./start_services.sh --background          # Mode arrière-plan
./start_services.sh --start-db --start-redis  # Avec services locaux
```

### 📊 **monitor_health.sh** - Surveillance et Monitoring
Système complet de surveillance des services avec génération de rapports.

**Fonctionnalités principales :**
- Surveillance continue des services
- Collecte de métriques système
- Tests de performance automatisés
- Génération d'alertes et rapports

**Utilisation typique :**
```bash
./monitor_health.sh                  # Monitoring continu
./monitor_health.sh --dashboard       # Tableau de bord interactif
./monitor_health.sh --performance     # Tests de performance
```

### 📋 **manage_logs.sh** - Gestion des Logs
Système industriel de gestion, rotation et analyse des logs.

**Fonctionnalités principales :**
- Rotation automatique des logs volumineux
- Archivage avec compression
- Analyse intelligente des erreurs
- Génération de rapports d'analyse

**Utilisation typique :**
```bash
./manage_logs.sh --status            # Statut des logs
./manage_logs.sh --analyze-all       # Analyses complètes
./manage_logs.sh --full              # Maintenance complète
```

## Configuration Recommandée

### Variables d'Environnement Essentielles
```bash
# Configuration API
DEV_API_HOST=0.0.0.0
DEV_API_PORT=8000
DEV_HOT_RELOAD=true

# Configuration Base de Données
DEV_DB_HOST=localhost
DEV_DB_PORT=5432
DEV_DB_NAME=spotify_ai_dev

# Configuration Monitoring
MONITORING_INTERVAL=30
DEBUG=true
```

## Workflows d'Utilisation

### 🔄 **Démarrage Quotidien**
```bash
# 1. Premier démarrage (configuration initiale)
./setup_dev.sh

# 2. Démarrage des services
./start_services.sh --start-db --start-redis

# 3. Vérification du statut
./start_services.sh --status
```

### 🔄 **Maintenance Hebdomadaire**
```bash
# 1. Gestion des logs
./manage_logs.sh --full

# 2. Mise à jour des dépendances
./start_services.sh --update-deps

# 3. Tests de performance
./monitor_health.sh --performance
```

### 🔄 **Résolution de Problèmes**
```bash
# 1. Diagnostic des services
./monitor_health.sh --once

# 2. Analyse des erreurs
./manage_logs.sh --analyze-errors

# 3. Réinitialisation si nécessaire
./reset_db.sh
```

## Surveillance et Alertes

### Métriques Surveillées
- **Services** : Disponibilité API, PostgreSQL, Redis
- **Performance** : Temps de réponse, throughput
- **Système** : CPU, mémoire, espace disque
- **Sécurité** : Tentatives d'accès, erreurs d'authentification

### Seuils d'Alerte Configurables
- CPU > 80%
- Mémoire > 85%
- Espace disque > 90%
- Temps de réponse > 5000ms

## Gestion des Logs

### Types de Logs
- **Application** : Logs métier et fonctionnels
- **Erreurs** : Erreurs système et applicatives
- **Accès** : Logs d'accès API et web
- **Performance** : Métriques de performance
- **Sécurité** : Événements de sécurité

### Rotation et Archivage
- Rotation automatique à 100MB
- Archivage après 7 jours
- Rétention des archives : 30 jours
- Compression gzip automatique

## Sécurité

### Bonnes Pratiques Implémentées
- Chiffrement des secrets sensibles
- Validation des entrées utilisateur
- Logging sécurisé avec masquage
- Permissions restrictives sur les fichiers

### Audit et Conformité
- Logging complet des opérations
- Traçabilité des modifications
- Rapports de conformité automatiques
- Intégration avec les systèmes d'audit

## Dépannage Commun

### Problèmes Fréquents et Solutions

#### Services ne démarrent pas
```bash
# Diagnostic
./monitor_health.sh --once
./manage_logs.sh --analyze-errors

# Solution
./reset_db.sh
./start_services.sh --start-db --start-redis
```

#### Performance dégradée
```bash
# Analyse
./monitor_health.sh --performance
./manage_logs.sh --analyze-performance

# Optimisation
./manage_logs.sh --cleanup
```

#### Logs volumineux
```bash
# Maintenance
./manage_logs.sh --rotate --cleanup
```

## Support Technique

### Contacts
- **Fahed Mlaiel** - Lead Dev + AI Architect
- **Équipe Backend** - Support technique quotidien

### Documentation Supplémentaire
- Logs détaillés dans `/logs/`
- Rapports d'analyse dans `/logs/reports/`
- Métriques de monitoring dans `/logs/monitoring/`

### Contribution
Les améliorations et suggestions sont encouragées. Contactez l'équipe de développement pour contribuer au projet.

---

*Cette documentation est maintenue par l'équipe dirigée par Fahed Mlaiel et mise à jour régulièrement.*
