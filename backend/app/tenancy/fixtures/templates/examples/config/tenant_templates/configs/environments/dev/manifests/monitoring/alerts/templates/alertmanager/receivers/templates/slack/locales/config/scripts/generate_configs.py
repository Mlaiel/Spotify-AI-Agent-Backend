#!/usr/bin/env python3
"""
Générateur de configurations tenant avancées.

Ce script génère automatiquement des configurations tenant complètes
pour différents environnements et cas d'usage.
"""

import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TenantProfile:
    """Profil de tenant avec ses caractéristiques."""
    name: str
    tier: str
    environments: List[str]
    features: List[str]
    region: str
    compliance: List[str]
    max_users: int
    storage_gb: int


class ConfigGenerator:
    """Générateur de configurations tenant."""
    
    # Profils prédéfinis
    PROFILES = {
        'startup': TenantProfile(
            name='Startup',
            tier='basic',
            environments=['development', 'staging'],
            features=['basic_ai', 'spotify_basic'],
            region='us-east-1',
            compliance=['basic_security'],
            max_users=10,
            storage_gb=10
        ),
        'enterprise': TenantProfile(
            name='Enterprise',
            tier='enterprise',
            environments=['development', 'staging', 'production'],
            features=['advanced_ai', 'spotify_premium', 'collaboration', 'analytics'],
            region='multi-region',
            compliance=['gdpr', 'sox', 'iso27001'],
            max_users=1000,
            storage_gb=1000
        ),
        'agency': TenantProfile(
            name='Agency',
            tier='premium',
            environments=['development', 'production'],
            features=['ai_processing', 'spotify_integration', 'multi_tenant'],
            region='eu-west-1',
            compliance=['gdpr', 'ccpa'],
            max_users=100,
            storage_gb=100
        )
    }
    
    def __init__(self):
        """Initialise le générateur."""
        self.timestamp = datetime.utcnow()
    
    def generate_tenant_id(self, company_name: str) -> str:
        """
        Génère un ID tenant unique.
        
        Args:
            company_name: Nom de l'entreprise
            
        Returns:
            ID tenant formaté
        """
        # Nettoyage du nom
        clean_name = ''.join(c.lower() for c in company_name if c.isalnum())
        # Ajout d'un identifiant unique
        unique_id = str(uuid.uuid4())[:8]
        return f"spotify-ai-{clean_name}-{unique_id}"
    
    def generate_database_config(self, env: str, tenant_id: str) -> Dict[str, Any]:
        """
        Génère la configuration de base de données pour un environnement.
        
        Args:
            env: Nom de l'environnement
            tenant_id: ID du tenant
            
        Returns:
            Configuration de base de données
        """
        env_configs = {
            'development': {
                'host': 'localhost',
                'port': 5432,
                'ssl_mode': 'prefer',
                'connection_pool': {
                    'min_connections': 2,
                    'max_connections': 10,
                    'timeout': 30
                }
            },
            'staging': {
                'host': f'{tenant_id}-staging-db.company.com',
                'port': 5432,
                'ssl_mode': 'require',
                'connection_pool': {
                    'min_connections': 5,
                    'max_connections': 20,
                    'timeout': 30
                }
            },
            'production': {
                'host': f'{tenant_id}-prod-db.company.com',
                'port': 5432,
                'ssl_mode': 'require',
                'connection_pool': {
                    'min_connections': 10,
                    'max_connections': 50,
                    'timeout': 30
                }
            }
        }
        
        base_config = env_configs.get(env, env_configs['development'])
        base_config.update({
            'name': f'spotify_ai_{env}',
            'schema': tenant_id.replace('-', '_')
        })
        
        return base_config
    
    def generate_cache_config(self, env: str, tenant_id: str) -> Dict[str, Any]:
        """
        Génère la configuration de cache Redis.
        
        Args:
            env: Nom de l'environnement
            tenant_id: ID du tenant
            
        Returns:
            Configuration de cache
        """
        return {
            'redis': {
                'host': f'redis-{env}.company.com' if env != 'development' else 'localhost',
                'port': 6379,
                'db': {'development': 0, 'staging': 1, 'production': 2}.get(env, 0),
                'prefix': f'{tenant_id}:',
                'ttl': 3600
            }
        }
    
    def generate_storage_config(self, env: str, tenant_id: str, profile: TenantProfile) -> Dict[str, Any]:
        """
        Génère la configuration de stockage.
        
        Args:
            env: Nom de l'environnement
            tenant_id: ID du tenant
            profile: Profil du tenant
            
        Returns:
            Configuration de stockage
        """
        bucket_name = f'{tenant_id}-{env}-storage'
        
        return {
            'type': 's3',
            'bucket': bucket_name,
            'prefix': f'{tenant_id}/',
            'encryption': True
        }
    
    def generate_monitoring_config(self, env: str, profile: TenantProfile) -> Dict[str, Any]:
        """
        Génère la configuration de monitoring.
        
        Args:
            env: Nom de l'environnement
            profile: Profil du tenant
            
        Returns:
            Configuration de monitoring
        """
        log_levels = {
            'development': 'DEBUG',
            'staging': 'INFO',
            'production': 'WARNING'
        }
        
        return {
            'enabled': True,
            'metrics': {
                'prometheus': {
                    'endpoint': f'http://prometheus-{env}:9090',
                    'labels': {
                        'environment': env,
                        'tier': profile.tier
                    }
                }
            },
            'logging': {
                'level': log_levels.get(env, 'INFO'),
                'structured': True,
                'retention_days': 30 if env == 'development' else 90
            }
        }
    
    def generate_features_config(self, profile: TenantProfile) -> Dict[str, Any]:
        """
        Génère la configuration des fonctionnalités.
        
        Args:
            profile: Profil du tenant
            
        Returns:
            Configuration des fonctionnalités
        """
        features = {}
        
        # Fonctionnalités IA
        if any('ai' in feature for feature in profile.features):
            ai_models = {
                'basic': ['gpt-3.5-turbo'],
                'premium': ['gpt-3.5-turbo', 'gpt-4'],
                'enterprise': ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet']
            }
            
            rate_limits = {
                'basic': {'requests_per_minute': 60, 'requests_per_hour': 1000},
                'premium': {'requests_per_minute': 300, 'requests_per_hour': 10000},
                'enterprise': {'requests_per_minute': 1000, 'requests_per_hour': 50000}
            }
            
            features['ai_processing'] = {
                'enabled': True,
                'models': ai_models.get(profile.tier, ai_models['basic']),
                'rate_limits': rate_limits.get(profile.tier, rate_limits['basic'])
            }
        
        # Intégration Spotify
        if any('spotify' in feature for feature in profile.features):
            scopes = {
                'basic': ['user-read-private', 'user-read-email'],
                'premium': ['user-read-private', 'user-read-email', 'playlist-read-private', 'user-top-read'],
                'enterprise': [
                    'user-read-private', 'user-read-email', 'playlist-read-private',
                    'user-top-read', 'user-read-recently-played', 'playlist-modify-private'
                ]
            }
            
            features['spotify_integration'] = {
                'enabled': True,
                'api_version': 'v1',
                'scopes': scopes.get(profile.tier, scopes['basic'])
            }
        
        # Collaboration
        if 'collaboration' in profile.features:
            features['collaboration'] = {
                'enabled': True,
                'max_users': profile.max_users,
                'real_time': profile.tier in ['premium', 'enterprise']
            }
        
        return features
    
    def generate_security_config(self, profile: TenantProfile) -> Dict[str, Any]:
        """
        Génère la configuration de sécurité.
        
        Args:
            profile: Profil du tenant
            
        Returns:
            Configuration de sécurité
        """
        # Authentification
        auth_config = {
            'provider': 'oauth2' if profile.tier == 'enterprise' else 'jwt',
            'mfa_required': profile.tier in ['premium', 'enterprise'],
            'session_timeout': 3600 if profile.tier == 'basic' else 7200
        }
        
        # Chiffrement
        encryption_config = {
            'at_rest': True,
            'in_transit': True,
            'algorithm': 'AES-256'
        }
        
        # Confidentialité des données
        data_privacy = {
            'gdpr_compliant': 'gdpr' in profile.compliance,
            'data_retention_days': 365 if profile.tier == 'basic' else 1095,
            'anonymization': True
        }
        
        return {
            'authentication': auth_config,
            'encryption': encryption_config,
            'data_privacy': data_privacy
        }
    
    def generate_billing_config(self, profile: TenantProfile) -> Dict[str, Any]:
        """
        Génère la configuration de facturation.
        
        Args:
            profile: Profil du tenant
            
        Returns:
            Configuration de facturation
        """
        plans = {
            'basic': 'free',
            'standard': 'basic',
            'premium': 'pro',
            'enterprise': 'enterprise'
        }
        
        usage_limits = {
            'basic': {
                'api_calls': 10000,
                'storage_gb': profile.storage_gb,
                'ai_minutes': 60
            },
            'standard': {
                'api_calls': 100000,
                'storage_gb': profile.storage_gb,
                'ai_minutes': 500
            },
            'premium': {
                'api_calls': 500000,
                'storage_gb': profile.storage_gb,
                'ai_minutes': 2000
            },
            'enterprise': {
                'api_calls': -1,  # Illimité
                'storage_gb': profile.storage_gb,
                'ai_minutes': -1  # Illimité
            }
        }
        
        return {
            'plan': plans.get(profile.tier, 'free'),
            'billing_cycle': 'yearly' if profile.tier == 'enterprise' else 'monthly',
            'usage_limits': usage_limits.get(profile.tier, usage_limits['basic'])
        }
    
    def generate_tenant_config(
        self,
        company_name: str,
        profile_name: str,
        contact_email: str,
        contact_name: str,
        custom_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Génère une configuration tenant complète.
        
        Args:
            company_name: Nom de l'entreprise
            profile_name: Nom du profil à utiliser
            contact_email: Email de contact
            contact_name: Nom du contact
            custom_config: Configuration personnalisée additionnelle
            
        Returns:
            Configuration tenant complète
        """
        profile = self.PROFILES.get(profile_name)
        if not profile:
            raise ValueError(f"Profil inconnu: {profile_name}")
        
        tenant_id = self.generate_tenant_id(company_name)
        
        # Métadonnées
        metadata = {
            'name': f"{company_name} - Spotify AI Agent",
            'description': f"Configuration {profile.name} pour {company_name}",
            'created_at': self.timestamp.isoformat(),
            'updated_at': self.timestamp.isoformat(),
            'owner': {
                'user_id': f"user_{uuid.uuid4().hex[:8]}",
                'email': contact_email,
                'name': contact_name
            },
            'tags': [profile.tier, profile.region] + profile.compliance
        }
        
        # Environnements
        environments = {}
        for env in profile.environments:
            environments[env] = {
                'enabled': True,
                'database': self.generate_database_config(env, tenant_id),
                'cache': self.generate_cache_config(env, tenant_id),
                'storage': self.generate_storage_config(env, tenant_id, profile),
                'monitoring': self.generate_monitoring_config(env, profile)
            }
        
        # Configuration complète
        config = {
            'tenant_id': tenant_id,
            'metadata': metadata,
            'environments': environments,
            'features': self.generate_features_config(profile),
            'security': self.generate_security_config(profile),
            'billing': self.generate_billing_config(profile)
        }
        
        # Fusion avec la configuration personnalisée
        if custom_config:
            config = self._deep_merge(config, custom_config)
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Fusionne récursivement deux dictionnaires.
        
        Args:
            base: Dictionnaire de base
            override: Dictionnaire de surcharge
            
        Returns:
            Dictionnaire fusionné
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(
        self,
        config: Dict[str, Any],
        output_dir: str = '.',
        format_type: str = 'json'
    ) -> str:
        """
        Sauvegarde la configuration dans un fichier.
        
        Args:
            config: Configuration à sauvegarder
            output_dir: Répertoire de sortie
            format_type: Format de fichier ('json' ou 'yaml')
            
        Returns:
            Chemin du fichier créé
        """
        tenant_id = config['tenant_id']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{tenant_id}_{timestamp}.{format_type}"
        filepath = Path(output_dir) / filename
        
        # Création du répertoire si nécessaire
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde
        with open(filepath, 'w', encoding='utf-8') as f:
            if format_type == 'json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif format_type == 'yaml':
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Format non supporté: {format_type}")
        
        logger.info(f"Configuration sauvegardée: {filepath}")
        return str(filepath)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Générateur de configurations tenant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Profils disponibles:
{chr(10).join(f"  {name}: {profile.name} - {profile.tier}" for name, profile in ConfigGenerator.PROFILES.items())}

Exemples d'utilisation:
  python generate_configs.py --company "MyStartup" --profile startup --email admin@mystartup.com --name "John Doe"
  python generate_configs.py --company "BigCorp" --profile enterprise --email it@bigcorp.com --name "IT Admin" --format yaml
        """
    )
    
    parser.add_argument(
        '--company',
        type=str,
        required=True,
        help='Nom de l\'entreprise'
    )
    
    parser.add_argument(
        '--profile',
        type=str,
        choices=list(ConfigGenerator.PROFILES.keys()),
        required=True,
        help='Profil de configuration à utiliser'
    )
    
    parser.add_argument(
        '--email',
        type=str,
        required=True,
        help='Email du contact principal'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Nom du contact principal'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='generated_configs',
        help='Répertoire de sortie (défaut: generated_configs)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'yaml'],
        default='json',
        help='Format de fichier de sortie (défaut: json)'
    )
    
    parser.add_argument(
        '--custom-config',
        type=str,
        help='Fichier JSON de configuration personnalisée à fusionner'
    )
    
    args = parser.parse_args()
    
    # Chargement de la configuration personnalisée
    custom_config = None
    if args.custom_config:
        with open(args.custom_config, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
    
    # Génération
    generator = ConfigGenerator()
    
    try:
        config = generator.generate_tenant_config(
            company_name=args.company,
            profile_name=args.profile,
            contact_email=args.email,
            contact_name=args.name,
            custom_config=custom_config
        )
        
        filepath = generator.save_config(
            config=config,
            output_dir=args.output_dir,
            format_type=args.format
        )
        
        print(f"\n✅ Configuration générée avec succès!")
        print(f"📁 Fichier: {filepath}")
        print(f"🏢 Entreprise: {args.company}")
        print(f"📊 Profil: {args.profile}")
        print(f"🆔 Tenant ID: {config['tenant_id']}")
        print(f"🌍 Environnements: {', '.join(config['environments'].keys())}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
