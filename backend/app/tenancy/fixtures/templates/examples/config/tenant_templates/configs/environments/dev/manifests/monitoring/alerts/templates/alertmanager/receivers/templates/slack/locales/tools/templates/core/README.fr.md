# Module Core - Système de Tenancy Avancé

**Auteur**: Fahed Mlaiel  
**Rôle**: Lead Dev & Architecte IA  
**Version**: 1.0.0  

## Aperçu

Ce module Core fournit l'infrastructure centrale pour le système de tenancy multi-tenant du Spotify AI Agent. Il intègre des fonctionnalités avancées de gestion, sécurité, monitoring et orchestration pour une solution industrielle complète.

## Architecture

### Composants Principaux

```
┌─────────────────────────────────────────────────────────────┐
│                    MODULE CORE                              │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │   Configuration │ │    Sécurité     │ │     Cache       │ │
│ │    Manager      │ │    Manager      │ │    Manager      │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │    Alertes      │ │   Templates     │ │   Métriques     │ │
│ │    Manager      │ │    Moteur       │ │   Collecteur    │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │   Validation    │ │    Workflow     │ │   Événements    │ │
│ │   Framework     │ │    Moteur       │ │   Bus           │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1. Gestionnaire de Configuration (`config.py`)
- **Fonction**: Gestion centralisée de la configuration
- **Caractéristiques**:
  - Configuration hiérarchique par environnement
  - Rechargement à chaud
  - Validation de configuration
  - Gestion des secrets
  - Configuration par tenant

### 2. Système d'Alertes (`alerts.py`)
- **Fonction**: Gestion intelligente des alertes
- **Caractéristiques**:
  - Règles d'alertes configurables
  - Multiples canaux (Email, Slack, Webhook)
  - Agrégation et déduplication
  - Escalade automatique
  - Templates d'alertes

### 3. Moteur de Templates (`templates.py`)
- **Fonction**: Rendu de templates avec localisation
- **Caractéristiques**:
  - Support Jinja2 avancé
  - Internationalisation (i18n)
  - Cache de templates
  - Templates dynamiques
  - Validation de templates

### 4. Gestionnaire de Sécurité (`security.py`)
- **Fonction**: Sécurité multi-couches
- **Caractéristiques**:
  - Chiffrement AES-256
  - Gestion des permissions
  - Journal d'audit complet
  - Politiques de sécurité
  - Contrôle d'accès granulaire

### 5. Système de Cache (`cache.py`)
- **Fonction**: Cache distribué haute performance
- **Caractéristiques**:
  - Support cluster Redis
  - Cache multi-niveaux
  - Invalidation intelligente
  - Compression automatique
  - Métriques de cache

### 6. Collecteur de Métriques (`metrics.py`)
- **Fonction**: Monitoring et observabilité
- **Caractéristiques**:
  - Métriques Prometheus
  - Agrégation en temps réel
  - Métriques système et métier
  - Tableaux de bord automatiques
  - Seuils et alertes

### 7. Framework de Validation (`validation.py`)
- **Fonction**: Validation avancée des données
- **Caractéristiques**:
  - Règles de validation flexibles
  - Validation schéma JSON/YAML
  - Validateurs personnalisés
  - Rapports d'erreurs détaillés
  - Validation asynchrone

### 8. Moteur de Workflow (`workflow.py`)
- **Fonction**: Orchestration des processus
- **Caractéristiques**:
  - Workflows configurables
  - Tâches parallèles et séquentielles
  - Gestion des erreurs et retry
  - Conditions et boucles
  - Monitoring des workflows

### 9. Bus d'Événements (`events.py`)
- **Fonction**: Architecture événementielle
- **Caractéristiques**:
  - Bus d'événements asynchrone
  - Handlers configurables
  - Priorités et filtrage
  - File de lettres mortes
  - Métriques d'événements

## Utilisation

### Initialisation du Système

```python
from core import initialize_core_system, shutdown_core_system

# Initialisation
await initialize_core_system()

# Utilisation des composants
from core import config_manager, alert_manager, template_engine

# Configuration
config = await config_manager.get_tenant_config("tenant_123")

# Alertes
await alert_manager.send_alert("system.high_cpu", {"value": 95})

# Templates
html = await template_engine.render("welcome_email", {"user": "Jean"}, locale="fr")

# Arrêt propre
await shutdown_core_system()
```

### Configuration d'un Tenant

```python
from core import tenant_validator, workflow_engine

# Validation
tenant_data = {
    "tenant_id": "acme_corp",
    "name": "ACME Corporation", 
    "email": "admin@acme.com",
    "api_quota_per_hour": 5000,
    "storage_quota_gb": 100.0,
    "features": ["audio_processing", "analytics"]
}

result = tenant_validator.validate(tenant_data)
if result.is_valid:
    # Lancement du workflow de provisioning
    workflow_id = await workflow_engine.create_workflow_from_template(
        "tenant_provisioning", 
        tenant_data["tenant_id"],
        {"tenant_config": tenant_data}
    )
    
    # Exécution du workflow
    workflow_result = await workflow_engine.execute_workflow(
        workflow_id, 
        tenant_data["tenant_id"], 
        {"tenant_config": tenant_data}
    )
```

### Gestion des Événements

```python
from core import event_bus, publish_tenant_created

# Publication d'événement
await publish_tenant_created("tenant_123", {
    "name": "Tenant Test",
    "plan": "premium"
})

# Handler personnalisé
class CustomHandler(EventHandler):
    async def handle(self, event):
        print(f"Traitement de {event.event_type} pour {event.tenant_id}")
        return True

# Enregistrement
custom_handler = CustomHandler()
event_bus.register_handler(custom_handler)
```

## Métriques et Monitoring

### Métriques Disponibles

- **Métriques Tenant**:
  - `tenant_requests_total`: Nombre de requêtes par tenant
  - `tenant_response_time_seconds`: Temps de réponse
  - `tenant_storage_usage_bytes`: Utilisation stockage
  - `tenant_api_quota_usage`: Utilisation quota API

- **Métriques Système**:
  - `system_cpu_usage_percent`: Utilisation CPU
  - `system_memory_usage_bytes`: Utilisation mémoire
  - `system_disk_usage_percent`: Utilisation disque

### Tableaux de Bord

Le module génère automatiquement des tableaux de bord Grafana pour:
- Vue d'ensemble des tenants
- Performance système
- Métriques métier
- Alertes et incidents

## Sécurité

### Fonctionnalités de Sécurité

1. **Chiffrement**:
   - AES-256 pour les données sensibles
   - Chiffrement en transit (TLS)
   - Rotation automatique des clés

2. **Contrôle d'Accès**:
   - RBAC (Role-Based Access Control)
   - Permissions granulaires
   - Isolation par tenant

3. **Audit**:
   - Journaux d'audit complets
   - Traçabilité des actions
   - Conformité automatique

4. **Politiques**:
   - Politiques de sécurité configurables
   - Validation automatique
   - Rapports de conformité

## Configuration

### Fichier de Configuration Principal

```yaml
# config/environments/dev/core.yaml
core:
  security:
    encryption_key: "${ENCRYPTION_KEY}"
    audit_enabled: true
    
  cache:
    redis_url: "redis://localhost:6379"
    default_ttl: 3600
    
  metrics:
    prometheus_enabled: true
    collection_interval: 30
    
  alerts:
    channels:
      email:
        smtp_host: "smtp.example.com"
        smtp_port: 587
      slack:
        webhook_url: "${SLACK_WEBHOOK}"
```

## Scripts d'Administration

### Scripts Disponibles

1. **Initialisation**: `scripts/init_core_system.py`
2. **Sauvegarde**: `scripts/backup_core_data.py`
3. **Migration**: `scripts/migrate_core_schema.py`
4. **Monitoring**: `scripts/health_check.py`

### Utilisation

```bash
# Initialisation
python scripts/init_core_system.py --env dev

# Vérification de santé
python scripts/health_check.py --detailed

# Sauvegarde
python scripts/backup_core_data.py --output /backup/core_$(date +%Y%m%d).tar.gz
```

## Tests et Validation

### Types de Tests

1. **Tests Unitaires**: Chaque composant individuellement
2. **Tests d'Intégration**: Interactions entre composants
3. **Tests de Performance**: Tests de charge et de stress
4. **Tests de Sécurité**: Tests de vulnérabilité et de pénétration

### Exécution des Tests

```bash
# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Tests de performance
pytest tests/performance/ --benchmark-only

# Couverture de code
pytest --cov=core tests/
```

## Déploiement

### Environnements

- **Développement**: Configuration de base
- **Staging**: Configuration proche production
- **Production**: Configuration optimisée

### Conteneurs

```dockerfile
# Dockerfile pour le module core
FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY core/ /app/core/
WORKDIR /app

CMD ["python", "-m", "core"]
```

## Support et Maintenance

### Journalisation

Les logs sont structurés en JSON pour faciliter l'analyse:

```json
{
  "timestamp": "2025-01-19T10:30:00Z",
  "level": "INFO",
  "component": "core.cache",
  "tenant_id": "tenant_123",
  "message": "Cache hit for key tenant_config",
  "latency_ms": 2.5
}
```

### Résolution de Problèmes

1. **Problèmes de Performance**: Vérifier les métriques de cache
2. **Erreurs de Configuration**: Valider avec le schéma
3. **Problèmes de Sécurité**: Consulter les journaux d'audit
4. **Échecs de Workflow**: Analyser les résultats des tâches

### Contact

**Développeur Principal**: Fahed Mlaiel  
**Email**: fahed.mlaiel@spotify-ai.com  
**Rôle**: Lead Developer & AI Architect  

---

*Ce module fait partie du projet Spotify AI Agent et suit les standards industriels pour la sécurité, la performance et la maintenabilité.*
