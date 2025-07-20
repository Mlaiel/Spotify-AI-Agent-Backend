#!/bin/bash
"""
Script d'Initialisation et Déploiement des Fixtures Slack
=========================================================

Script bash avancé pour l'initialisation complète du système de fixtures
d'alertes Slack dans l'environnement multi-tenant Spotify AI Agent.

Fonctionnalités:
- Création automatique de la base de données
- Configuration des secrets et tokens
- Déploiement des templates par défaut
- Validation des configurations
- Tests d'intégration
- Monitoring et health checks

Auteur: Fahed Mlaiel - Lead Developer Achiri
Version: 2.5.0
Usage: ./deploy_fixtures.sh [environment] [action]
"""

set -euo pipefail

# Configuration globale
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../../../../../.." && pwd)"
LOG_FILE="/tmp/slack_fixtures_deploy_$(date +%Y%m%d_%H%M%S).log"
ENVIRONMENT="${1:-dev}"
ACTION="${2:-deploy}"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration par environnement
declare -A CONFIG
CONFIG[dev_db_host]="localhost"
CONFIG[dev_db_port]="5432"
CONFIG[dev_db_name]="spotify_ai_dev"
CONFIG[dev_redis_host]="localhost"
CONFIG[dev_redis_port]="6379"
CONFIG[dev_slack_webhook]="https://hooks.slack.com/services/dev/webhook"

CONFIG[staging_db_host]="staging-db.achiri.com"
CONFIG[staging_db_port]="5432"
CONFIG[staging_db_name]="spotify_ai_staging"
CONFIG[staging_redis_host]="staging-redis.achiri.com"
CONFIG[staging_redis_port]="6379"
CONFIG[staging_slack_webhook]="https://hooks.slack.com/services/staging/webhook"

CONFIG[prod_db_host]="prod-db.achiri.com"
CONFIG[prod_db_port]="5432"
CONFIG[prod_db_name]="spotify_ai_prod"
CONFIG[prod_redis_host]="prod-redis.achiri.com"
CONFIG[prod_redis_port]="6379"
CONFIG[prod_slack_webhook]="https://hooks.slack.com/services/prod/webhook"

# Fonctions utilitaires
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        DEBUG)
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

show_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║               Spotify AI Agent - Slack Fixtures             ║
║              Déploiement Avancé Multi-Tenant                ║
║                                                              ║
║  Développé par: Fahed Mlaiel - Lead Developer Achiri        ║
║  Version: 2.5.0                                             ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_prerequisites() {
    log INFO "Vérification des prérequis..."
    
    # Vérification des outils requis
    local tools=("python3" "pip3" "psql" "redis-cli" "jq" "curl" "docker")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log ERROR "Outil manquant: $tool"
            return 1
        fi
    done
    
    # Vérification des variables d'environnement
    local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "SLACK_BOT_TOKEN")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log ERROR "Variable d'environnement manquante: $var"
            return 1
        fi
    done
    
    # Vérification de Python et des dépendances
    if ! python3 -c "import asyncpg, aioredis, jinja2, prometheus_client" 2>/dev/null; then
        log WARN "Dépendances Python manquantes, installation..."
        pip3 install -r "$SCRIPT_DIR/requirements.txt"
    fi
    
    log INFO "Prérequis validés avec succès"
}

setup_database() {
    log INFO "Configuration de la base de données PostgreSQL..."
    
    local db_host="${CONFIG[${ENVIRONMENT}_db_host]}"
    local db_port="${CONFIG[${ENVIRONMENT}_db_port]}"
    local db_name="${CONFIG[${ENVIRONMENT}_db_name]}"
    
    # Test de connexion
    if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$db_host" -p "$db_port" -U postgres -d postgres -c "\q" 2>/dev/null; then
        log ERROR "Impossible de se connecter à PostgreSQL"
        return 1
    fi
    
    # Création de la base de données si nécessaire
    if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$db_host" -p "$db_port" -U postgres -lqt | cut -d \| -f 1 | grep -qw "$db_name"; then
        log INFO "Création de la base de données: $db_name"
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$db_host" -p "$db_port" -U postgres -c "CREATE DATABASE $db_name;"
    fi
    
    # Exécution des migrations
    log INFO "Exécution des migrations..."
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$db_host" -p "$db_port" -U postgres -d "$db_name" << 'EOF'
-- Table des fixtures Slack
CREATE TABLE IF NOT EXISTS slack_fixtures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    locale VARCHAR(10) NOT NULL,
    alert_type VARCHAR(100) NOT NULL,
    template_data JSONB NOT NULL,
    metadata JSONB NOT NULL,
    version VARCHAR(50) NOT NULL,
    hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, environment, locale, alert_type)
);

-- Index pour les performances
CREATE INDEX IF NOT EXISTS idx_slack_fixtures_tenant 
ON slack_fixtures(tenant_id, environment);

CREATE INDEX IF NOT EXISTS idx_slack_fixtures_lookup
ON slack_fixtures(tenant_id, environment, locale, alert_type);

CREATE INDEX IF NOT EXISTS idx_slack_fixtures_updated
ON slack_fixtures(updated_at);

-- Table des métriques
CREATE TABLE IF NOT EXISTS slack_metrics (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les métriques
CREATE INDEX IF NOT EXISTS idx_slack_metrics_timestamp
ON slack_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_slack_metrics_tenant
ON slack_metrics(tenant_id, environment, metric_name);

-- Table des logs d'audit
CREATE TABLE IF NOT EXISTS slack_audit_log (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour l'audit
CREATE INDEX IF NOT EXISTS idx_slack_audit_timestamp
ON slack_audit_log(timestamp);

CREATE INDEX IF NOT EXISTS idx_slack_audit_tenant
ON slack_audit_log(tenant_id, action);

-- Vue pour les statistiques
CREATE OR REPLACE VIEW slack_fixture_stats AS
SELECT 
    tenant_id,
    environment,
    locale,
    COUNT(*) as fixture_count,
    MAX(updated_at) as last_updated
FROM slack_fixtures
GROUP BY tenant_id, environment, locale;

EOF
    
    log INFO "Base de données configurée avec succès"
}

setup_redis() {
    log INFO "Configuration de Redis..."
    
    local redis_host="${CONFIG[${ENVIRONMENT}_redis_host]}"
    local redis_port="${CONFIG[${ENVIRONMENT}_redis_port]}"
    
    # Test de connexion Redis
    if ! redis-cli -h "$redis_host" -p "$redis_port" -a "$REDIS_PASSWORD" ping | grep -q PONG; then
        log ERROR "Impossible de se connecter à Redis"
        return 1
    fi
    
    # Configuration Redis pour le cache
    redis-cli -h "$redis_host" -p "$redis_port" -a "$REDIS_PASSWORD" << 'EOF'
CONFIG SET maxmemory-policy allkeys-lru
CONFIG SET maxmemory 1gb
CONFIG REWRITE
EOF
    
    log INFO "Redis configuré avec succès"
}

generate_secrets() {
    log INFO "Génération des secrets..."
    
    local secrets_dir="$SCRIPT_DIR/secrets"
    mkdir -p "$secrets_dir"
    
    # Génération de la clé de chiffrement Fernet
    if [[ ! -f "$secrets_dir/encryption_key_$ENVIRONMENT" ]]; then
        python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > "$secrets_dir/encryption_key_$ENVIRONMENT"
        chmod 600 "$secrets_dir/encryption_key_$ENVIRONMENT"
        log INFO "Clé de chiffrement générée pour $ENVIRONMENT"
    fi
    
    # Génération des tokens API
    if [[ ! -f "$secrets_dir/api_token_$ENVIRONMENT" ]]; then
        python3 -c "import secrets; print(secrets.token_urlsafe(32))" > "$secrets_dir/api_token_$ENVIRONMENT"
        chmod 600 "$secrets_dir/api_token_$ENVIRONMENT"
        log INFO "Token API généré pour $ENVIRONMENT"
    fi
    
    # Génération des certificats TLS pour l'environnement de production
    if [[ "$ENVIRONMENT" == "prod" && ! -f "$secrets_dir/tls_cert.pem" ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout "$secrets_dir/tls_key.pem" -out "$secrets_dir/tls_cert.pem" -days 365 -nodes \
            -subj "/C=FR/ST=IDF/L=Paris/O=Achiri/OU=DevOps/CN=spotify-ai-agent.achiri.com"
        chmod 600 "$secrets_dir/tls_"*
        log INFO "Certificats TLS générés pour la production"
    fi
    
    log INFO "Secrets générés avec succès"
}

deploy_default_fixtures() {
    log INFO "Déploiement des fixtures par défaut..."
    
    local fixtures_script="$SCRIPT_DIR/load_default_fixtures.py"
    
    cat > "$fixtures_script" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Script de chargement des fixtures par défaut.
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Ajout du chemin du module
sys.path.append(str(Path(__file__).parent))

from manager import SlackFixtureManager, Environment, Locale
from defaults import DEFAULT_TEMPLATES, TEMPLATE_METADATA
from utils import SlackAlertValidator

async def load_fixtures():
    """Charge les fixtures par défaut."""
    
    # Configuration selon l'environnement
    environment = os.getenv('FIXTURE_ENVIRONMENT', 'dev')
    
    config = {
        'database': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'name': os.getenv('DB_NAME', f'spotify_ai_{environment}')
        },
        'redis': {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD')
        },
        'security': {
            'encryption_key': open(f'secrets/encryption_key_{environment}', 'r').read().strip()
        }
    }
    
    # Initialisation du gestionnaire
    manager = SlackFixtureManager(config)
    await manager.initialize()
    
    try:
        # Tenants par défaut
        tenants = [
            'spotify-tenant-01',
            'spotify-tenant-02', 
            'demo-tenant',
            'test-tenant'
        ]
        
        validator = SlackAlertValidator()
        loaded_count = 0
        
        # Chargement pour chaque combinaison tenant/locale/type
        for tenant_id in tenants:
            for locale, templates in DEFAULT_TEMPLATES.items():
                for alert_type, template in templates.items():
                    
                    # Validation du template
                    is_valid, errors = validator.validate_template(template)
                    if not is_valid:
                        print(f"ERREUR: Template invalide {locale}/{alert_type}: {errors}")
                        continue
                    
                    # Création de la configuration de fixture
                    from manager import FixtureConfigModel, SlackTemplateModel
                    
                    fixture_config = FixtureConfigModel(
                        metadata={
                            'tenant_id': tenant_id,
                            'environment': environment,
                            'locale': locale,
                            'alert_type': alert_type,
                            'version': TEMPLATE_METADATA['version'],
                            'tags': ['default', 'system'],
                            'priority': 5
                        },
                        template=SlackTemplateModel(**template)
                    )
                    
                    # Sauvegarde
                    fixture_id = await manager.save_fixture(
                        tenant_id=tenant_id,
                        environment=Environment(environment),
                        locale=Locale(locale),
                        alert_type=alert_type,
                        config=fixture_config
                    )
                    
                    loaded_count += 1
                    print(f"Fixture chargée: {tenant_id}/{locale}/{alert_type} -> {fixture_id}")
        
        print(f"\n✅ {loaded_count} fixtures chargées avec succès")
        
        # Test de chargement
        test_fixture = await manager.load_fixture(
            tenant_id=tenants[0],
            environment=Environment(environment),
            locale=Locale('fr'),
            alert_type='system_critical'
        )
        
        if test_fixture:
            print("✅ Test de chargement réussi")
        else:
            print("❌ Échec du test de chargement")
            
    finally:
        await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(load_fixtures())
PYTHON_EOF
    
    # Exécution du script de chargement
    chmod +x "$fixtures_script"
    
    export FIXTURE_ENVIRONMENT="$ENVIRONMENT"
    export DB_HOST="${CONFIG[${ENVIRONMENT}_db_host]}"
    export DB_PORT="${CONFIG[${ENVIRONMENT}_db_port]}"
    export DB_NAME="${CONFIG[${ENVIRONMENT}_db_name]}"
    export REDIS_HOST="${CONFIG[${ENVIRONMENT}_redis_host]}"
    export REDIS_PORT="${CONFIG[${ENVIRONMENT}_redis_port]}"
    
    if python3 "$fixtures_script"; then
        log INFO "Fixtures par défaut déployées avec succès"
    else
        log ERROR "Échec du déploiement des fixtures"
        return 1
    fi
    
    rm -f "$fixtures_script"
}

run_tests() {
    log INFO "Exécution des tests d'intégration..."
    
    local test_script="$SCRIPT_DIR/integration_tests.py"
    
    cat > "$test_script" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Tests d'intégration pour les fixtures Slack.
"""

import asyncio
import sys
import os
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from manager import SlackFixtureManager, Environment, Locale

async def run_integration_tests():
    """Exécute les tests d'intégration."""
    
    environment = os.getenv('FIXTURE_ENVIRONMENT', 'dev')
    
    config = {
        'database': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'name': os.getenv('DB_NAME', f'spotify_ai_{environment}')
        },
        'redis': {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD')
        },
        'security': {
            'encryption_key': open(f'secrets/encryption_key_{environment}', 'r').read().strip()
        }
    }
    
    manager = SlackFixtureManager(config)
    await manager.initialize()
    
    try:
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Chargement d'une fixture
        tests_total += 1
        fixture = await manager.load_fixture(
            tenant_id="spotify-tenant-01",
            environment=Environment(environment),
            locale=Locale('fr'),
            alert_type='system_critical'
        )
        
        if fixture:
            print("✅ Test 1: Chargement de fixture réussi")
            tests_passed += 1
        else:
            print("❌ Test 1: Échec du chargement de fixture")
        
        # Test 2: Rendu de template
        if fixture:
            tests_total += 1
            context = {
                'alert': {
                    'severity': 'critical',
                    'summary': 'Test alert',
                    'description': 'Test description',
                    'timestamp': 1641024000
                },
                'tenant': {
                    'name': 'Test Tenant',
                    'id': 'spotify-tenant-01'
                },
                'environment': environment
            }
            
            try:
                rendered = await manager.render_template(fixture, context)
                if rendered and 'channel' in rendered:
                    print("✅ Test 2: Rendu de template réussi")
                    tests_passed += 1
                else:
                    print("❌ Test 2: Rendu de template invalide")
            except Exception as e:
                print(f"❌ Test 2: Erreur de rendu: {e}")
        
        # Test 3: Health check
        tests_total += 1
        health = await manager.health_check()
        if health['status'] in ['healthy', 'degraded']:
            print("✅ Test 3: Health check réussi")
            tests_passed += 1
        else:
            print("❌ Test 3: Échec du health check")
        
        # Test 4: Liste des fixtures
        tests_total += 1
        fixtures_list = await manager.list_fixtures(
            tenant_id="spotify-tenant-01",
            environment=Environment(environment)
        )
        
        if len(fixtures_list) > 0:
            print(f"✅ Test 4: Liste des fixtures réussie ({len(fixtures_list)} fixtures)")
            tests_passed += 1
        else:
            print("❌ Test 4: Aucune fixture trouvée")
        
        # Résumé
        print(f"\n📊 Résumé des tests: {tests_passed}/{tests_total} réussis")
        
        if tests_passed == tests_total:
            print("🎉 Tous les tests sont passés avec succès!")
            return True
        else:
            print("⚠️  Certains tests ont échoué")
            return False
            
    finally:
        await manager.cleanup()

if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)
PYTHON_EOF
    
    chmod +x "$test_script"
    
    if python3 "$test_script"; then
        log INFO "Tests d'intégration réussis"
    else
        log ERROR "Échec des tests d'intégration"
        return 1
    fi
    
    rm -f "$test_script"
}

setup_monitoring() {
    log INFO "Configuration du monitoring..."
    
    # Configuration Prometheus
    local prometheus_config="$SCRIPT_DIR/prometheus_rules.yml"
    
    cat > "$prometheus_config" << 'EOF'
groups:
  - name: slack_fixtures
    rules:
      - alert: SlackFixtureHighErrorRate
        expr: rate(slack_alerts_failed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Taux d'erreur élevé pour les alertes Slack"
          description: "Le taux d'erreur des alertes Slack est de {{ $value }} pour le tenant {{ $labels.tenant_id }}"
      
      - alert: SlackFixtureSlowRendering
        expr: histogram_quantile(0.95, rate(slack_template_render_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Rendu de template Slack lent"
          description: "Le 95e percentile du temps de rendu est de {{ $value }}s"
      
      - alert: SlackFixtureCacheHighMissRate
        expr: rate(fixture_cache_misses_total[5m]) / (rate(fixture_cache_hits_total[5m]) + rate(fixture_cache_misses_total[5m])) > 0.5
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "Taux de miss cache élevé"
          description: "Le taux de miss du cache est de {{ $value }}"
EOF
    
    # Configuration Grafana dashboard
    local grafana_dashboard="$SCRIPT_DIR/grafana_dashboard.json"
    
    cat > "$grafana_dashboard" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Slack Fixtures - Spotify AI Agent",
    "tags": ["slack", "monitoring", "fixtures"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Alertes Slack Envoyées",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(slack_alerts_sent_total[5m]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps"
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Taux d'Erreur",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(slack_alerts_failed_total[5m])) / sum(rate(slack_alerts_sent_total[5m]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "max": 1,
            "min": 0
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      }
    ]
  }
}
EOF
    
    log INFO "Configuration du monitoring terminée"
}

cleanup() {
    log INFO "Nettoyage..."
    
    # Suppression des fichiers temporaires
    rm -f "$SCRIPT_DIR"/*.tmp
    rm -f "$SCRIPT_DIR"/integration_tests.py
    rm -f "$SCRIPT_DIR"/load_default_fixtures.py
    
    log INFO "Nettoyage terminé"
}

show_status() {
    log INFO "Statut du déploiement pour l'environnement: $ENVIRONMENT"
    
    echo -e "\n${BLUE}=== Configuration ===${NC}"
    echo "Environnement: $ENVIRONMENT"
    echo "Base de données: ${CONFIG[${ENVIRONMENT}_db_host]}:${CONFIG[${ENVIRONMENT}_db_port]}"
    echo "Redis: ${CONFIG[${ENVIRONMENT}_redis_host]}:${CONFIG[${ENVIRONMENT}_redis_port]}"
    echo "Log file: $LOG_FILE"
    
    echo -e "\n${BLUE}=== Services ===${NC}"
    
    # Test PostgreSQL
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "${CONFIG[${ENVIRONMENT}_db_host]}" -p "${CONFIG[${ENVIRONMENT}_db_port]}" -U postgres -c "\q" 2>/dev/null; then
        echo -e "PostgreSQL: ${GREEN}✓ Connecté${NC}"
    else
        echo -e "PostgreSQL: ${RED}✗ Déconnecté${NC}"
    fi
    
    # Test Redis
    if redis-cli -h "${CONFIG[${ENVIRONMENT}_redis_host]}" -p "${CONFIG[${ENVIRONMENT}_redis_port]}" -a "$REDIS_PASSWORD" ping | grep -q PONG 2>/dev/null; then
        echo -e "Redis: ${GREEN}✓ Connecté${NC}"
    else
        echo -e "Redis: ${RED}✗ Déconnecté${NC}"
    fi
    
    # Statistiques des fixtures
    local db_host="${CONFIG[${ENVIRONMENT}_db_host]}"
    local db_port="${CONFIG[${ENVIRONMENT}_db_port]}"
    local db_name="${CONFIG[${ENVIRONMENT}_db_name]}"
    
    if command -v psql >/dev/null 2>&1; then
        local fixture_count=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$db_host" -p "$db_port" -U postgres -d "$db_name" -t -c "SELECT COUNT(*) FROM slack_fixtures;" 2>/dev/null | tr -d ' ' || echo "0")
        echo -e "Fixtures déployées: ${GREEN}$fixture_count${NC}"
    fi
}

main() {
    show_banner
    
    log INFO "Démarrage du déploiement pour l'environnement: $ENVIRONMENT"
    log INFO "Action: $ACTION"
    log INFO "Logs disponibles dans: $LOG_FILE"
    
    case $ACTION in
        "deploy")
            check_prerequisites
            setup_database
            setup_redis
            generate_secrets
            deploy_default_fixtures
            setup_monitoring
            run_tests
            show_status
            log INFO "Déploiement terminé avec succès!"
            ;;
        "test")
            check_prerequisites
            run_tests
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 [environment] [action]"
            echo "Environments: dev, staging, prod"
            echo "Actions: deploy, test, status, cleanup"
            exit 1
            ;;
    esac
}

# Point d'entrée
main "$@"
