"""
Feature Flags Management System
==============================

Système de gestion des feature flags pour le contrôle granulaire des fonctionnalités
dans l'application Spotify AI Agent. Permet l'activation/désactivation dynamique
des fonctionnalités sans redéploiement.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

class FeatureStatus(Enum):
    """Statuts des feature flags."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    BETA = "beta"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"

class TargetAudience(Enum):
    """Audiences cibles pour les feature flags."""
    ALL_USERS = "all_users"
    BETA_USERS = "beta_users"
    PREMIUM_USERS = "premium_users"
    ADMIN_USERS = "admin_users"
    INTERNAL_USERS = "internal_users"
    PERCENTAGE_USERS = "percentage_users"

@dataclass
class FeatureFlag:
    """Définition d'un feature flag."""
    key: str
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.DISABLED
    target_audience: TargetAudience = TargetAudience.ALL_USERS
    percentage: float = 0.0  # Pour les rollouts progressifs (0-100)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class FeatureFlagManager:
    """Gestionnaire des feature flags."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.flags = self._initialize_default_flags()
    
    def _initialize_default_flags(self) -> Dict[str, FeatureFlag]:
        """Initialise les feature flags par défaut."""
        default_flags = [
            # Core Features
            FeatureFlag(
                key="core.api_v2",
                name="API Version 2",
                description="Active la nouvelle version de l'API avec des fonctionnalités améliorées",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="core.rate_limiting",
                name="Rate Limiting",
                description="Active la limitation de débit pour les API",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="core.caching",
                name="Response Caching",
                description="Active la mise en cache des réponses API",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            
            # AI & Machine Learning Features
            FeatureFlag(
                key="ai.music_recommendation",
                name="AI Music Recommendation",
                description="Système de recommandation musicale basé sur l'IA",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="ai.sentiment_analysis",
                name="Sentiment Analysis",
                description="Analyse de sentiment des commentaires et reviews",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.BETA_USERS,
                percentage=50.0
            ),
            FeatureFlag(
                key="ai.playlist_generation",
                name="AI Playlist Generation",
                description="Génération automatique de playlists par IA",
                status=FeatureStatus.EXPERIMENTAL,
                target_audience=TargetAudience.INTERNAL_USERS,
                percentage=25.0
            ),
            FeatureFlag(
                key="ai.music_classification",
                name="Music Classification",
                description="Classification automatique des genres musicaux",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="ai.voice_commands",
                name="Voice Commands",
                description="Contrôle vocal de l'application",
                status=FeatureStatus.EXPERIMENTAL,
                target_audience=TargetAudience.BETA_USERS,
                percentage=10.0
            ),
            
            # Audio Processing Features
            FeatureFlag(
                key="audio.spleeter_separation",
                name="Spleeter Audio Separation",
                description="Séparation audio en pistes individuelles avec Spleeter",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.PREMIUM_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="audio.real_time_processing",
                name="Real-time Audio Processing",
                description="Traitement audio en temps réel",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.PREMIUM_USERS,
                percentage=30.0
            ),
            FeatureFlag(
                key="audio.noise_reduction",
                name="Noise Reduction",
                description="Réduction de bruit automatique",
                status=FeatureStatus.EXPERIMENTAL,
                target_audience=TargetAudience.BETA_USERS,
                percentage=20.0
            ),
            FeatureFlag(
                key="audio.format_conversion",
                name="Audio Format Conversion",
                description="Conversion entre différents formats audio",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            
            # Analytics Features
            FeatureFlag(
                key="analytics.user_behavior",
                name="User Behavior Analytics",
                description="Analyse du comportement utilisateur",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="analytics.music_trends",
                name="Music Trends Analytics",
                description="Analyse des tendances musicales",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.PREMIUM_USERS,
                percentage=75.0
            ),
            FeatureFlag(
                key="analytics.real_time_dashboard",
                name="Real-time Dashboard",
                description="Tableau de bord en temps réel",
                status=FeatureStatus.EXPERIMENTAL,
                target_audience=TargetAudience.ADMIN_USERS,
                percentage=100.0
            ),
            
            # Collaboration Features
            FeatureFlag(
                key="collaboration.playlist_sharing",
                name="Playlist Sharing",
                description="Partage de playlists entre utilisateurs",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="collaboration.real_time_editing",
                name="Real-time Collaborative Editing",
                description="Édition collaborative en temps réel",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.PREMIUM_USERS,
                percentage=60.0
            ),
            FeatureFlag(
                key="collaboration.comments_system",
                name="Comments System",
                description="Système de commentaires sur les playlists",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="collaboration.live_sessions",
                name="Live Music Sessions",
                description="Sessions d'écoute collaborative en direct",
                status=FeatureStatus.EXPERIMENTAL,
                target_audience=TargetAudience.BETA_USERS,
                percentage=15.0
            ),
            
            # Security Features
            FeatureFlag(
                key="security.two_factor_auth",
                name="Two-Factor Authentication",
                description="Authentification à deux facteurs",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.PREMIUM_USERS,
                percentage=80.0
            ),
            FeatureFlag(
                key="security.oauth_providers",
                name="OAuth Providers",
                description="Authentification via fournisseurs OAuth",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="security.api_key_auth",
                name="API Key Authentication",
                description="Authentification par clé API",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="security.advanced_encryption",
                name="Advanced Encryption",
                description="Chiffrement avancé des données sensibles",
                status=FeatureStatus.EXPERIMENTAL,
                target_audience=TargetAudience.PREMIUM_USERS,
                percentage=50.0
            ),
            
            # Performance Features
            FeatureFlag(
                key="performance.auto_scaling",
                name="Auto Scaling",
                description="Mise à l'échelle automatique des ressources",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="performance.lazy_loading",
                name="Lazy Loading",
                description="Chargement paresseux des ressources",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="performance.cdn_optimization",
                name="CDN Optimization",
                description="Optimisation via CDN",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.ALL_USERS,
                percentage=90.0
            ),
            
            # Monitoring Features
            FeatureFlag(
                key="monitoring.detailed_metrics",
                name="Detailed Metrics",
                description="Métriques détaillées de performance",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="monitoring.error_tracking",
                name="Advanced Error Tracking",
                description="Suivi avancé des erreurs",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="monitoring.user_analytics",
                name="User Analytics",
                description="Analyses utilisateur détaillées",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.ADMIN_USERS,
                percentage=100.0
            ),
            
            # Business Features
            FeatureFlag(
                key="business.subscription_model",
                name="Subscription Model",
                description="Modèle d'abonnement premium",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="business.payment_processing",
                name="Payment Processing",
                description="Traitement des paiements",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="business.referral_system",
                name="Referral System",
                description="Système de parrainage",
                status=FeatureStatus.EXPERIMENTAL,
                target_audience=TargetAudience.BETA_USERS,
                percentage=40.0
            ),
            
            # UI/UX Features
            FeatureFlag(
                key="ui.dark_mode",
                name="Dark Mode",
                description="Mode sombre de l'interface",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="ui.mobile_responsive",
                name="Mobile Responsive Design",
                description="Design responsive pour mobile",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            ),
            FeatureFlag(
                key="ui.accessibility_features",
                name="Accessibility Features",
                description="Fonctionnalités d'accessibilité",
                status=FeatureStatus.BETA,
                target_audience=TargetAudience.ALL_USERS,
                percentage=85.0
            ),
            FeatureFlag(
                key="ui.advanced_search",
                name="Advanced Search",
                description="Recherche avancée avec filtres",
                status=FeatureStatus.ENABLED,
                target_audience=TargetAudience.ALL_USERS,
                percentage=100.0
            )
        ]
        
        return {flag.key: flag for flag in default_flags}
    
    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Récupère un feature flag par sa clé."""
        return self.flags.get(key)
    
    def is_enabled(self, key: str, user_context: Optional[Dict[str, Any]] = None) -> bool:
        """Vérifie si un feature flag est activé pour un contexte utilisateur donné."""
        flag = self.get_flag(key)
        if not flag:
            return False
        
        # Vérifier le statut
        if flag.status == FeatureStatus.DISABLED:
            return False
        
        # Vérifier les dates
        now = datetime.now()
        if flag.start_date and now < flag.start_date:
            return False
        if flag.end_date and now > flag.end_date:
            return False
        
        # Vérifier les dépendances
        for dependency in flag.dependencies:
            if not self.is_enabled(dependency, user_context):
                return False
        
        # Vérifier l'audience cible
        if not self._check_target_audience(flag, user_context):
            return False
        
        # Vérifier le pourcentage de rollout
        if flag.percentage < 100.0:
            return self._check_percentage_rollout(flag, user_context)
        
        return flag.status in [FeatureStatus.ENABLED, FeatureStatus.BETA, FeatureStatus.EXPERIMENTAL]
    
    def _check_target_audience(self, flag: FeatureFlag, user_context: Optional[Dict[str, Any]]) -> bool:
        """Vérifie si l'utilisateur fait partie de l'audience cible."""
        if not user_context:
            return flag.target_audience == TargetAudience.ALL_USERS
        
        user_type = user_context.get("user_type", "regular")
        
        audience_checks = {
            TargetAudience.ALL_USERS: True,
            TargetAudience.BETA_USERS: user_context.get("is_beta_user", False),
            TargetAudience.PREMIUM_USERS: user_context.get("is_premium_user", False),
            TargetAudience.ADMIN_USERS: user_type == "admin",
            TargetAudience.INTERNAL_USERS: user_type == "internal",
            TargetAudience.PERCENTAGE_USERS: True  # Géré par le pourcentage
        }
        
        return audience_checks.get(flag.target_audience, False)
    
    def _check_percentage_rollout(self, flag: FeatureFlag, user_context: Optional[Dict[str, Any]]) -> bool:
        """Vérifie le rollout en pourcentage."""
        if not user_context or "user_id" not in user_context:
            return False
        
        # Utilise le hash de l'ID utilisateur + clé du flag pour la cohérence
        user_id = str(user_context["user_id"])
        hash_input = f"{flag.key}:{user_id}"
        hash_value = hash(hash_input) % 100
        
        return hash_value < flag.percentage
    
    def set_flag_status(self, key: str, status: FeatureStatus) -> bool:
        """Modifie le statut d'un feature flag."""
        if key not in self.flags:
            return False
        
        self.flags[key].status = status
        self.flags[key].updated_at = datetime.now()
        return True
    
    def set_flag_percentage(self, key: str, percentage: float) -> bool:
        """Modifie le pourcentage de rollout d'un feature flag."""
        if key not in self.flags or not (0 <= percentage <= 100):
            return False
        
        self.flags[key].percentage = percentage
        self.flags[key].updated_at = datetime.now()
        return True
    
    def add_flag(self, flag: FeatureFlag) -> bool:
        """Ajoute un nouveau feature flag."""
        if flag.key in self.flags:
            return False
        
        self.flags[flag.key] = flag
        return True
    
    def remove_flag(self, key: str) -> bool:
        """Supprime un feature flag."""
        if key not in self.flags:
            return False
        
        del self.flags[key]
        return True
    
    def list_flags(self, status_filter: Optional[FeatureStatus] = None) -> List[FeatureFlag]:
        """Liste les feature flags avec filtre optionnel."""
        flags = list(self.flags.values())
        
        if status_filter:
            flags = [flag for flag in flags if flag.status == status_filter]
        
        return sorted(flags, key=lambda f: f.key)
    
    def export_to_config(self) -> Dict[str, str]:
        """Exporte les feature flags vers un format de configuration."""
        config = {}
        
        for key, flag in self.flags.items():
            # Convertit en format de variable d'environnement
            env_key = f"FEATURE_{key.upper().replace('.', '_')}_ENABLED"
            config[env_key] = str(self.is_enabled(key)).lower()
            
            # Ajoute le pourcentage si applicable
            if flag.percentage < 100.0:
                percentage_key = f"FEATURE_{key.upper().replace('.', '_')}_PERCENTAGE"
                config[percentage_key] = str(flag.percentage)
        
        return config
    
    def import_from_config(self, config: Dict[str, str]) -> None:
        """Importe les feature flags depuis une configuration."""
        for env_key, value in config.items():
            if env_key.startswith("FEATURE_") and env_key.endswith("_ENABLED"):
                # Extrait la clé du feature flag
                flag_key = env_key[8:-8].lower().replace("_", ".")
                
                if flag_key in self.flags:
                    status = FeatureStatus.ENABLED if value.lower() == "true" else FeatureStatus.DISABLED
                    self.set_flag_status(flag_key, status)
    
    def get_flags_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des feature flags."""
        total = len(self.flags)
        enabled = len([f for f in self.flags.values() if f.status == FeatureStatus.ENABLED])
        beta = len([f for f in self.flags.values() if f.status == FeatureStatus.BETA])
        experimental = len([f for f in self.flags.values() if f.status == FeatureStatus.EXPERIMENTAL])
        disabled = len([f for f in self.flags.values() if f.status == FeatureStatus.DISABLED])
        
        return {
            "total_flags": total,
            "enabled": enabled,
            "beta": beta,
            "experimental": experimental,
            "disabled": disabled,
            "environment": self.environment,
            "last_updated": max([f.updated_at for f in self.flags.values()]) if self.flags else None
        }

# Décorateur pour vérifier les feature flags
def require_feature_flag(flag_key: str):
    """Décorateur pour vérifier qu'un feature flag est activé."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Cette implémentation devrait être adaptée selon le framework utilisé
            # Par exemple, avec FastAPI, on pourrait accéder à la requête pour obtenir le contexte utilisateur
            manager = FeatureFlagManager()
            
            if not manager.is_enabled(flag_key):
                raise ValueError(f"Feature '{flag_key}' is not enabled")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Exportation des classes
__all__ = [
    'FeatureFlag',
    'FeatureStatus',
    'TargetAudience',
    'FeatureFlagManager',
    'require_feature_flag'
]
