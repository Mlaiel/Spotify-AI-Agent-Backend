# Configuration Management Suite

## Vue d'ensemble

Cette suite de gestion des configurations fournit un ensemble complet d'outils industriels pour gérer les configurations Kubernetes du projet Spotify AI Agent. Elle inclut la génération automatisée, la validation, le déploiement, la surveillance, la sécurité, et la gestion des sauvegardes avec des capacités avancées de rollback.

**Auteurs**: Fahed Mlaiel  
**Équipe**: 
- ✅ Lead Dev + Architecte IA
- ✅ Développeur Backend Senior (Python/FastAPI/Django)  
- ✅ Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- ✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- ✅ Spécialiste Sécurité Backend
- ✅ Architecte Microservices

**Version**: 2.0.0  
**Date**: July 17, 2025

## Architecture

```
scripts/
├── generate_configs.py      # Génération automatisée des configurations
├── validate_configs.py      # Validation et conformité des configurations  
├── deploy_configs.py        # Déploiement Kubernetes
├── monitor_configs.py       # Surveillance en temps réel
├── security_scanner.py      # Scanner de sécurité avancé
├── rollback_configs.py      # Gestion des rollbacks
├── backup_restore.py        # Système de sauvegarde et restauration
├── config_management.sh     # Script principal d'orchestration
└── README.md               # Cette documentation
```

## Scripts disponibles

### 1. generate_configs.py

**Objectif**: Génère automatiquement toutes les configurations nécessaires basées sur les profils d'environnement.

**Fonctionnalités**:
- Génération de ConfigMaps et Secrets
- Support multi-environnement (dev, staging, prod)
- Templates dynamiques avec variables
- Validation intégrée lors de la génération
- Mode dry-run pour les tests

**Usage**:
```bash
# Génération pour l'environnement de développement
python3 generate_configs.py --environment dev --namespace spotify-ai-agent-dev

# Mode dry-run
python3 generate_configs.py --environment prod --dry-run

# Génération avec répertoire de sortie personnalisé
python3 generate_configs.py --output-dir ./custom-configs/
```

**Options principales**:
- `--environment` : Environnement cible (dev/staging/prod)
- `--namespace` : Namespace Kubernetes
- `--output-dir` : Répertoire de sortie
- `--dry-run` : Mode simulation
- `--template-dir` : Répertoire des templates

### 2. validate_configs.py

**Objectif**: Valide la conformité, la sécurité et la cohérence des configurations.

**Fonctionnalités**:
- Validation syntaxique YAML/JSON
- Vérification des schémas Pydantic
- Contrôles de sécurité et conformité
- Validation des contraintes métier
- Rapports détaillés avec scores

**Usage**:
```bash
# Validation complète avec conformité sécurité
python3 validate_configs.py --config-dir ./configs/ --security-compliance

# Validation avec rapport détaillé
python3 validate_configs.py --detailed-report --output-format json

# Validation d'un environnement spécifique
python3 validate_configs.py --environment prod --strict-mode
```

**Options principales**:
- `--config-dir` : Répertoire des configurations
- `--security-compliance` : Vérifications de sécurité
- `--detailed-report` : Rapport détaillé
- `--output-format` : Format de sortie (json/yaml/text)
- `--strict-mode` : Mode strict

### 3. deploy_configs.py

**Objectif**: Déploie les configurations dans un cluster Kubernetes avec vérifications complètes.

**Fonctionnalités**:
- Déploiement sécurisé avec vérifications préalables
- Support du rollback automatique
- Attente et vérification des déploiements
- Gestion des permissions et namespaces
- Logs détaillés des opérations

**Usage**:
```bash
# Déploiement standard
python3 deploy_configs.py --namespace spotify-ai-agent-dev --apply

# Déploiement avec surveillance du rollout
python3 deploy_configs.py --apply --wait-for-rollout --timeout 600

# Mode dry-run pour validation
python3 deploy_configs.py --config-dir ./configs/ --dry-run
```

**Options principales**:
- `--config-dir` : Répertoire des configurations
- `--namespace` : Namespace Kubernetes cible
- `--apply` : Application réelle (requis)
- `--dry-run` : Mode simulation
- `--wait-for-rollout` : Attendre la fin du déploiement
- `--verify` : Vérifier après déploiement

### 4. monitor_configs.py

**Objectif**: Surveille l'état des configurations déployées et génère des métriques.

**Fonctionnalités**:
- Surveillance en temps réel
- Calcul de scores de santé
- Détection d'alertes automatique
- Export métriques (Prometheus, JSON, CSV)
- Historique des métriques

**Usage**:
```bash
# Surveillance continue
python3 monitor_configs.py --namespace spotify-ai-agent-dev --watch --interval 30

# Vérification unique
python3 monitor_configs.py --one-shot

# Export des métriques Prometheus
python3 monitor_configs.py --export-metrics --format prometheus --output metrics.txt
```

**Options principales**:
- `--namespace` : Namespace à surveiller
- `--watch` : Mode surveillance continue
- `--interval` : Intervalle de vérification (secondes)
- `--export-metrics` : Export des métriques
- `--format` : Format d'export (json/prometheus/csv)

### 5. security_scanner.py

**Objectif**: Scanner de sécurité avancé pour détecter vulnérabilités et non-conformités.

**Fonctionnalités**:
- Scan de sécurité multi-niveaux (configuration, secrets, RBAC, réseau)
- Détection de vulnérabilités connues (CVE)
- Vérifications de conformité (GDPR, CIS, SOC2, NIST)
- Export de rapports (JSON, SARIF, HTML, CSV)
- Scoring de risque automatique

**Usage**:
```bash
# Scan complet de sécurité
python3 security_scanner.py --namespace spotify-ai-agent-dev --full-scan

# Scan de conformité uniquement
python3 security_scanner.py --compliance-check

# Export rapport SARIF pour intégration CI/CD
python3 security_scanner.py --export-report --format sarif --output security.sarif
```

**Options principales**:
- `--full-scan` : Scan complet (tous types)
- `--scan-types` : Types spécifiques (configuration, secrets, rbac, network, compliance)
- `--compliance-check` : Vérifications conformité uniquement
- `--export-report` : Export du rapport
- `--format` : Format de sortie (json/sarif/html/csv)
- `--severity` : Filtrage par sévérité minimum

### 6. rollback_configs.py

**Objectif**: Gestion avancée des rollbacks avec analyse d'impact et stratégies intelligentes.

**Fonctionnalités**:
- Création de sauvegardes automatiques avec métadonnées
- Analyse d'impact des rollbacks
- Stratégies de rollback (incrémental, atomique, standard)
- Rollback automatique basé sur la santé du système
- Validation post-rollback

**Usage**:
```bash
# Création d'une sauvegarde
python3 rollback_configs.py --create-backup --description "Avant mise à jour"

# Liste des sauvegardes
python3 rollback_configs.py --list-backups

# Rollback vers une révision
python3 rollback_configs.py --rollback --target-revision 5 --confirm

# Rollback automatique si santé < 60%
python3 rollback_configs.py --auto-rollback --health-threshold 60
```

**Options principales**:
- `--create-backup` : Création de sauvegarde
- `--list-backups` : Liste des sauvegardes
- `--rollback` : Exécution de rollback
- `--auto-rollback` : Rollback automatique
- `--target-revision` : Révision cible
- `--health-threshold` : Seuil de santé pour auto-rollback

### 7. backup_restore.py

**Objectif**: Système enterprise de sauvegarde et restauration avec stockage cloud.

**Fonctionnalités**:
- Sauvegardes complètes et incrémentielles
- Chiffrement et compression automatiques
- Synchronisation multi-cloud (AWS S3, Azure, GCP)
- Tests de restauration automatisés
- Politiques de rétention avancées

**Usage**:
```bash
# Sauvegarde complète
python3 backup_restore.py --create-backup --description "Sauvegarde production"

# Sauvegarde incrémentielle
python3 backup_restore.py --create-incremental backup-20250717-120000

# Restauration
python3 backup_restore.py --restore --backup-id backup-20250717-120000

# Test de restauration
python3 backup_restore.py --test-restore backup-20250717-120000

# Synchronisation cloud
python3 backup_restore.py --sync-to-cloud aws
```

**Options principales**:
- `--create-backup` : Sauvegarde complète
- `--create-incremental` : Sauvegarde incrémentielle
- `--restore` : Restauration
- `--test-restore` : Test de restauration
- `--sync-to-cloud` : Synchronisation cloud
- `--verify` : Vérification d'intégrité

### 8. config_management.sh (Amélioré)

**Objectif**: Script principal orchestrant tous les outils de gestion des configurations.

**Fonctionnalités**:
- Interface unifiée pour tous les outils
- Gestion des cycles complets
- Variables d'environnement
- Vérifications de dépendances
- Logs colorés et structurés

**Usage**:
```bash
# Cycle complet
./config_management.sh full-cycle

# Commandes individuelles
./config_management.sh generate
./config_management.sh validate
./config_management.sh deploy
./config_management.sh monitor

# Nouvelles commandes avancées
./config_management.sh security-scan
./config_management.sh rollback
./config_management.sh backup-restore
./config_management.sh security-audit
./config_management.sh disaster-test

# Surveillance avancée
./config_management.sh advanced-monitor

# Vérification du statut
./config_management.sh status

# Nettoyage
./config_management.sh cleanup
```

## Nouvelles Fonctionnalités Avancées

### 🔍 Scan de Sécurité Intégré

Le système de scan de sécurité analyse l'infrastructure complète :

```bash
# Scan de sécurité rapide
./config_management.sh security-scan

# Scan complet avec tous les contrôles
./config_management.sh security-scan --full-scan

# Scan de conformité spécifique
./config_management.sh security-scan --compliance-check GDPR

# Export du rapport de sécurité
./config_management.sh security-scan --export-report security-report.html
```

**Contrôles effectués** :
- ✅ Analyse des vulnérabilités CVE
- ✅ Vérification de la conformité GDPR/SOC2/CIS
- ✅ Audit des permissions RBAC
- ✅ Contrôle de la sécurité des secrets
- ✅ Validation des politiques réseau
- ✅ Scan des configurations Pod Security Standards

### ↩️ Gestion Avancée des Rollbacks

Système intelligent de rollback avec analyse d'impact :

```bash
# Rollback vers une révision spécifique
./config_management.sh rollback --target 5

# Rollback automatique basé sur la santé
./config_management.sh rollback --auto --health-threshold 60

# Liste des points de restauration
./config_management.sh rollback --list

# Création de point de sauvegarde manuel
./config_management.sh rollback --create-backup "Avant mise à jour critique"
```

**Fonctionnalités** :
- 🎯 Analyse d'impact pré-rollback
- 🔄 Stratégies multiples (incrémental, atomique, standard)
- 📊 Monitoring de santé post-rollback
- 🔒 Validation automatique des rollbacks
- 📝 Métadonnées enrichies des sauvegardes

### 💾 Système de Sauvegarde Enterprise

Solution complète de backup avec stockage cloud :

```bash
# Sauvegarde complète
./config_management.sh backup-restore --action create

# Sauvegarde incrémentielle
./config_management.sh backup-restore --action incremental

# Restauration d'une sauvegarde
./config_management.sh backup-restore --action restore --backup-id backup-20250717-120000

# Synchronisation cloud
./config_management.sh backup-restore --action sync --provider aws

# Test de restauration
./config_management.sh backup-restore --action test --backup-id backup-20250717-120000
```

**Capacités avancées** :
- 🌐 Support multi-cloud (AWS S3, Azure Blob, Google Cloud Storage)
- 🔐 Chiffrement AES-256 automatique
- 📦 Compression gzip optimisée
- ⏰ Politiques de rétention intelligentes
- 🧪 Tests de restauration automatisés
- 📈 Métriques de performance des sauvegardes

### 🛡️ Audit de Sécurité Complet

Audit approfondi de l'infrastructure avec rapport détaillé :

```bash
# Audit de sécurité complet
./config_management.sh complete-security-audit

# Audit avec rapport détaillé
./config_management.sh complete-security-audit --detailed-report

# Audit de conformité spécifique
./config_management.sh complete-security-audit --compliance SOC2
```

**Évaluations incluses** :
- 🔒 Score de sécurité global (0-100)
- 📊 Matrice de risques détaillée
- 🎯 Recommandations prioritaires
- 📋 Checklist de conformité
- 📄 Rapport exportable (HTML/PDF/JSON)

### 🚨 Test de Récupération après Sinistre

Validation de la résilience avec scénarios réalistes :

```bash
# Test de récupération complet
./config_management.sh disaster-test

# Test avec scénario spécifique
./config_management.sh disaster-test --scenario database-failure

# Test avec validation étendue
./config_management.sh disaster-test --extended-validation
```

**Scénarios de test** :
- 💣 Simulation de panne de base de données
- 🔥 Test de corruption des configurations
- ⚠️ Simulation d'échec de déploiement
- 💾 Test de perte de données critiques
- 🌐 Simulation d'indisponibilité des services

## 📈 Monitoring et Observabilité

Le système de monitoring collecte des métriques en temps réel avec alertes intelligentes :

```bash
# Surveillance continue avec alertes
./config_management.sh monitor

# Surveillance avec durée personnalisée
MONITOR_DURATION=600 ./config_management.sh monitor

# Export des métriques Prometheus
METRICS_FORMAT=prometheus ./config_management.sh monitor

# Surveillance avec alertes personnalisées
./config_management.sh monitor --alert-threshold 80
```

### 📊 Métriques disponibles
- **Performance** : CPU/Mémoire par pod, latence API, throughput
- **Fiabilité** : Taux d'erreur, disponibilité des services, SLA
- **Sécurité** : Score de sécurité, vulnérabilités détectées, conformité
- **Infrastructure** : Santé Kubernetes, utilisation ressources, capacité
- **Business** : Métriques métier personnalisées, KPIs applicatifs

### 🔔 Système d'alertes
- Alertes temps réel via Slack/Teams/Email
- Escalade automatique selon la criticité
- Corrélation intelligente des événements
- Seuils adaptatifs basés sur l'historique

## 🎯 Exemples d'Utilisation Avancés

### 🚀 Pipeline CI/CD Complet
```bash
#!/bin/bash
# Pipeline de déploiement sécurisé et automatisé

set -e
echo "🚀 Démarrage du pipeline de déploiement - $(date)"

# 1. Validation pré-déploiement
echo "📋 Validation des configurations..."
./config_management.sh validate || exit 1

# 2. Sauvegarde de sécurité
echo "💾 Création de la sauvegarde pré-déploiement..."
./config_management.sh backup-restore --action create --description "Pre-deployment-$(date +%Y%m%d-%H%M%S)"

# 3. Scan de sécurité
echo "🔍 Scan de sécurité pré-déploiement..."
./config_management.sh security-scan --full-scan || exit 1

# 4. Déploiement avec surveillance
echo "🎯 Déploiement avec monitoring..."
./config_management.sh deploy

# 5. Monitoring post-déploiement
echo "📊 Surveillance post-déploiement..."
./config_management.sh monitor --duration 300

# 6. Validation finale
echo "✅ Validation finale du déploiement..."
./config_management.sh complete-security-audit --quick

echo "🎉 Pipeline terminé avec succès - $(date)"
```

### 🌍 Gestion Multi-Environnements
```bash
#!/bin/bash
# Déploiement coordonné sur plusieurs environnements

environments=("dev" "staging" "prod")

for env in "${environments[@]}"; do
    echo "🌟 Déploiement sur l'environnement: $env"
    
    # Configuration de l'environnement
    export ENVIRONMENT=$env
    export NAMESPACE="spotify-ai-agent-$env"
    
    # Déploiement sécurisé
    ./config_management.sh validate
    ./config_management.sh deploy
    ./config_management.sh security-scan
    
    # Test de santé
    ./config_management.sh monitor --duration 180
    
    echo "✅ Environnement $env déployé avec succès"
done

# Sauvegarde croisée
echo "💾 Sauvegarde croisée des environnements..."
./config_management.sh backup-restore --action create --cross-env
```

### 🔧 Maintenance Automatisée
```bash
#!/bin/bash
# Script de maintenance hebdomadaire automatisée

set -e
echo "🛠️ Maintenance hebdomadaire automatisée - $(date)"

# Nettoyage des anciennes ressources
echo "🧹 Nettoyage des ressources..."
./config_management.sh cleanup --older-than 30d

# Mise à jour des configurations
echo "🔄 Mise à jour des configurations..."
./config_management.sh update-configs --auto-approve

# Audit de sécurité complet
echo "🛡️ Audit de sécurité complet..."
./config_management.sh complete-security-audit --export-report

# Test de récupération
echo "🧪 Test de récupération après sinistre..."
./config_management.sh disaster-test --automated

# Optimisation des performances
echo "⚡ Optimisation des performances..."
./config_management.sh optimize --auto-tune

# Génération du rapport de maintenance
echo "📊 Génération du rapport de maintenance..."
./config_management.sh generate-maintenance-report

echo "✅ Maintenance terminée avec succès - $(date)"
```

### 🚨 Gestion d'Incident Automatisée
```bash
#!/bin/bash
# Réponse automatique aux incidents critiques

incident_type=$1
severity=$2

echo "🚨 Détection d'incident: $incident_type (Sévérité: $severity)"

case $severity in
    "critical")
        # Rollback automatique immédiat
        ./config_management.sh rollback --auto --immediate
        
        # Activation du mode dégradé
        ./config_management.sh enable-degraded-mode
        
        # Notification d'escalade
        ./config_management.sh notify --escalate --channel emergency
        ;;
    "high")
        # Analyse d'impact
        ./config_management.sh analyze-impact --incident-type "$incident_type"
        
        # Rollback conditionnel
        ./config_management.sh rollback --conditional --health-threshold 70
        ;;
    "medium"|"low")
        # Monitoring renforcé
        ./config_management.sh monitor --enhanced --duration 1800
        
        # Rapport d'incident
        ./config_management.sh generate-incident-report --type "$incident_type"
        ;;
esac

echo "✅ Réponse à l'incident terminée"
```

```bash
# Scan complet avec rapport
./config_management.sh security-scan

## 🎯 Mise en Œuvre et Bonnes Pratiques

### 📋 Checklist de Déploiement
```bash
# Validation complète avant production
✅ ./config_management.sh validate
✅ ./config_management.sh security-scan --full-scan
✅ ./config_management.sh backup-restore --action create
✅ ./config_management.sh disaster-test --quick
✅ ./config_management.sh deploy
✅ ./config_management.sh monitor --duration 600
✅ ./config_management.sh complete-security-audit
```

### 🔄 Cycle de Vie DevOps Complet
```bash
# 1. Développement
./config_management.sh validate --env dev
./config_management.sh security-scan --quick

# 2. Test d'intégration
./config_management.sh deploy --env staging
./config_management.sh monitor --duration 300

# 3. Production
./config_management.sh backup-restore --action create
./config_management.sh deploy --env prod
./config_management.sh complete-security-audit
```

### 📊 Métriques et KPIs

**Disponibilité et Performance**:
- Uptime : > 99.9%
- Latence moyenne : < 200ms
- Taux d'erreur : < 0.1%

**Sécurité**:
- Score de sécurité : > 95/100
- Vulnérabilités critiques : 0
- Conformité GDPR : 100%

**Opérations**:
- Temps de déploiement : < 10min
- Temps de rollback : < 2min
- Couverture de monitoring : 100%

## 🚀 Intégration Continue et Déploiement Continu

### GitLab CI/CD
```yaml
# .gitlab-ci.yml
stages:
  - validate
  - security
  - backup
  - deploy
  - monitor
  - audit

validate_configs:
  stage: validate
  script:
    - ./config_management.sh validate

security_scan:
  stage: security
  script:
    - ./config_management.sh security-scan --full-scan
  artifacts:
    reports:
      security: security-report.json

backup_create:
  stage: backup
  script:
    - ./config_management.sh backup-restore --action create

deploy_production:
  stage: deploy
  script:
    - ./config_management.sh deploy
  only:
    - main

post_deploy_monitor:
  stage: monitor
  script:
    - ./config_management.sh monitor --duration 600

security_audit:
  stage: audit
  script:
    - ./config_management.sh complete-security-audit
```

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy Spotify AI Agent
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Validate Configurations
        run: ./config_management.sh validate
        
      - name: Security Scan
        run: ./config_management.sh security-scan --full-scan
        
      - name: Create Backup
        run: ./config_management.sh backup-restore --action create
        
      - name: Deploy
        run: ./config_management.sh deploy
        
      - name: Monitor Health
        run: ./config_management.sh monitor --duration 300
        
      - name: Security Audit
        run: ./config_management.sh complete-security-audit
```

## 🔗 Ressources et Documentation

### 📚 Documentation Technique
- [Architecture des Microservices](./docs/architecture.md)
- [Guide de Sécurité](./docs/security-guide.md)
- [Procédures de Backup](./docs/backup-procedures.md)
- [Playbook de Récupération](./docs/disaster-recovery.md)

### 🛠️ Outils Complémentaires
- **Monitoring** : Prometheus + Grafana + AlertManager
- **Logging** : ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing** : Jaeger + OpenTelemetry
- **Security** : Falco + OPA Gatekeeper + Trivy

### 📞 Support et Escalade
- **Support L1** : équipe-support@spotify-ai-agent.com
- **Support L2** : équipe-infrastructure@spotify-ai-agent.com
- **Astreinte** : +33-1-XX-XX-XX-XX (24/7)
- **Incident Critique** : incident-critique@spotify-ai-agent.com

---

## 🏆 Conclusion

Ce système de gestion des configurations représente une solution **enterprise-grade** complète pour le déploiement, la surveillance et la maintenance d'infrastructures Kubernetes critiques. 

### ✨ Points Forts
- **🔒 Sécurité** : Scan automatisé, conformité multi-standards, chiffrement end-to-end
- **🔄 Résilience** : Rollbacks intelligents, tests de récupération, haute disponibilité  
- **📊 Observabilité** : Monitoring temps réel, alertes proactives, métriques business
- **⚡ Performance** : Déploiements rapides, optimisations automatiques, scalabilité
- **🛡️ Conformité** : GDPR, SOC2, CIS, NIST - tous les standards respectés

### 🚀 Prêt pour la Production
Tous les scripts sont **industrialisés**, **testés** et **prêts** pour un déploiement en production immédiat. L'architecture modulaire permet une adoption progressive et une personnalisation selon vos besoins spécifiques.

**🎯 Démarrage rapide** : `./config_management.sh full-cycle`

---
*Développé avec ❤️ par l'équipe Spotify AI Agent - Enterprise Engineering*
```

### Gestion Avancée des Rollbacks

```bash
# Rollback vers une révision spécifique
ROLLBACK_TARGET=5 ./config_management.sh rollback

# Liste des sauvegardes disponibles
./config_management.sh rollback
```

### Système de Sauvegarde Enterprise

```bash
# Création de sauvegarde
BACKUP_ACTION=create ./config_management.sh backup-restore

# Restauration depuis une sauvegarde
BACKUP_ACTION=restore BACKUP_ID=backup-20250717 ./config_management.sh backup-restore

# Liste des sauvegardes
./config_management.sh backup-restore
```

### Test de Disaster Recovery

```bash
# Test complet de reprise après sinistre
./config_management.sh disaster-test
```

## Variables d'environnement

```bash
# Configuration du namespace
export NAMESPACE="spotify-ai-agent-dev"

# Environnement de déploiement
export ENVIRONMENT="dev"

# Mode dry-run
export DRY_RUN="true"

# Durée de surveillance (secondes)
export MONITOR_DURATION="300"

# Format d'export des métriques
export METRICS_FORMAT="prometheus"

# Répertoire de sortie personnalisé
export OUTPUT_DIR="./custom-configs"

# Nouvelles variables pour fonctionnalités avancées
export ROLLBACK_TARGET="5"
export BACKUP_ACTION="create"
export BACKUP_ID="backup-20250717-120000"

# Configuration cloud pour sauvegardes
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"
```

## Flux de travail recommandés

### Développement quotidien

```bash
# 1. Génération et validation locale
DRY_RUN=true ./config_management.sh generate
DRY_RUN=true ./config_management.sh validate

# 2. Test en mode dry-run
DRY_RUN=true ./config_management.sh deploy

# 3. Déploiement réel si validé
./config_management.sh deploy
```

### Déploiement de production

```bash
# 1. Sauvegarde de la configuration actuelle
./config_management.sh backup

# 2. Cycle complet avec surveillance
ENVIRONMENT=prod NAMESPACE=spotify-ai-agent-prod ./config_management.sh full-cycle

# 3. Surveillance prolongée
MONITOR_DURATION=3600 ./config_management.sh monitor
```

### Surveillance opérationnelle

```bash
# Surveillance continue avec export de métriques
python3 monitor_configs.py --watch --export-metrics --format prometheus &

# Vérification de statut périodique
*/5 * * * * /path/to/config_management.sh status
```

## Intégration CI/CD

### Pipeline GitLab CI

```yaml
stages:
  - validate
  - deploy-dev
  - deploy-prod

validate-configs:
  stage: validate
  script:
    - ./scripts/config_management.sh validate
  only:
    - merge_requests

deploy-dev:
  stage: deploy-dev
  script:
    - ENVIRONMENT=dev ./scripts/config_management.sh full-cycle
  only:
    - develop

deploy-prod:
  stage: deploy-prod
  script:
    - ENVIRONMENT=prod ./scripts/config_management.sh full-cycle
  only:
    - main
  when: manual
```

### Pipeline GitHub Actions

```yaml
name: Configuration Management

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate configurations
        run: ./scripts/config_management.sh validate

  deploy:
    needs: validate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy configurations
        run: |
          ENVIRONMENT=prod ./scripts/config_management.sh full-cycle
```

## Surveillance et alerting

### Métriques Prometheus

Les métriques suivantes sont exposées:

```
# Score de santé général
spotify_ai_agent_health_score{namespace="spotify-ai-agent-dev"} 95.0

# Nombre de pods par phase
spotify_ai_agent_pods_total{namespace="spotify-ai-agent-dev",phase="Running"} 5
spotify_ai_agent_pods_total{namespace="spotify-ai-agent-dev",phase="Pending"} 0

# Répliques des déploiements
spotify_ai_agent_deployment_replicas_desired{namespace="spotify-ai-agent-dev",deployment="api"} 3
spotify_ai_agent_deployment_replicas_ready{namespace="spotify-ai-agent-dev",deployment="api"} 3
```

### Alertes recommandées

```yaml
# Score de santé faible
- alert: SpotifyAIAgentLowHealthScore
  expr: spotify_ai_agent_health_score < 50
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Score de santé faible"

# Pods non running
- alert: SpotifyAIAgentPodsNotRunning
  expr: spotify_ai_agent_pods_total{phase!="Running"} > 0
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Pods en état non-running détectés"
```

## Dépannage

### Problèmes courants

1. **Erreur de connectivité Kubernetes**
   ```bash
   # Vérification de kubectl
   kubectl cluster-info
   
   # Test des permissions
   kubectl auth can-i create configmaps --namespace spotify-ai-agent-dev
   ```

2. **Échec de validation des configurations**
   ```bash
   # Mode debug avec rapport détaillé
   python3 validate_configs.py --detailed-report --debug
   ```

3. **Déploiement en échec**
   ```bash
   # Vérification des logs
   kubectl logs -n spotify-ai-agent-dev -l app=spotify-ai-agent
   
   # Statut des ressources
   kubectl get all -n spotify-ai-agent-dev
   ```

### Logs et debug

Activer le mode debug:
```bash
export LOG_LEVEL=DEBUG
./config_management.sh [command]
```

Consulter les logs détaillés:
```bash
# Logs des pods
kubectl logs -f -n spotify-ai-agent-dev deployment/api

# Events du namespace
kubectl get events -n spotify-ai-agent-dev --sort-by='.lastTimestamp'
```

## Sécurité

### Bonnes pratiques

1. **Gestion des secrets**
   - Utilisation de Kubernetes Secrets
   - Chiffrement au repos activé
   - Rotation régulière des clés

2. **Permissions**
   - RBAC configuré avec privilèges minimum
   - ServiceAccounts dédiés
   - Audit des accès

3. **Validation**
   - Contrôles de conformité automatiques
   - Scan de vulnérabilités
   - Validation des images

### Conformité

Le script de validation vérifie automatiquement:
- Conformité GDPR/CCPA
- Standards de sécurité industriels
- Politiques d'entreprise
- Contraintes techniques

## Support et maintenance

### Maintenance régulière

```bash
# Nettoyage hebdomadaire
./config_management.sh cleanup

# Sauvegarde mensuelle
./config_management.sh backup

# Validation continue
./config_management.sh validate
```

### Mise à jour des scripts

Les scripts sont auto-documentés et incluent la gestion de versions. Vérifier régulièrement:
- Nouvelles versions des dépendances
- Mises à jour des APIs Kubernetes
- Évolution des standards de sécurité

### Support

Pour toute question ou problème:
1. Consulter cette documentation
2. Vérifier les logs détaillés
3. Utiliser le mode debug
4. Contacter l'équipe DevOps

---

**Auteur**: Fahed Mlaiel  
**Team**: Spotify AI Agent Development Team  
**Version**: 1.0.0  
**Dernière mise à jour**: 2024
