"""
Énumérations et types constants - Spotify AI Agent
Définitions centralisées des constantes et énumérations avec logique métier avancée
"""

from enum import Enum, IntEnum, Flag, auto
from typing import Dict, List, Set, Optional, Tuple
import colorsys
from datetime import datetime, timedelta


class SmartEnum(Enum):
    """Énumération de base avec fonctionnalités avancées"""
    
    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        """Retourne les choix pour les formulaires"""
        return [(item.value, item.name.replace('_', ' ').title()) for item in cls]
    
    @classmethod
    def values(cls) -> List[str]:
        """Retourne toutes les valeurs"""
        return [item.value for item in cls]
    
    @classmethod
    def names(cls) -> List[str]:
        """Retourne tous les noms"""
        return [item.name for item in cls]
    
    @classmethod
    def from_string(cls, value: str, default=None):
        """Crée une instance depuis une chaîne (insensible à la casse)"""
        value_lower = value.lower() if value else ""
        for item in cls:
            if item.value.lower() == value_lower or item.name.lower() == value_lower:
                return item
        return default
    
    def __str__(self) -> str:
        return self.value


class AlertLevel(SmartEnum):
    """Niveaux d'alerte avec priorités et métadonnées"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    DEBUG = "debug"
    
    @property
    def priority(self) -> int:
        """Priorité numérique (plus haut = plus important)"""
        priorities = {
            self.CRITICAL: 100,
            self.HIGH: 80,
            self.MEDIUM: 60,
            self.LOW: 40,
            self.INFO: 20,
            self.DEBUG: 10
        }
        return priorities[self]
    
    @property
    def color_code(self) -> str:
        """Code couleur hexadécimal"""
        colors = {
            self.CRITICAL: "#FF0000",  # Rouge vif
            self.HIGH: "#FF6600",      # Orange
            self.MEDIUM: "#FFD700",    # Or
            self.LOW: "#00BFFF",       # Bleu ciel
            self.INFO: "#32CD32",      # Vert citron
            self.DEBUG: "#808080"      # Gris
        }
        return colors[self]
    
    @property
    def emoji(self) -> str:
        """Emoji associé"""
        emojis = {
            self.CRITICAL: "🔴",
            self.HIGH: "🟠",
            self.MEDIUM: "🟡",
            self.LOW: "🔵",
            self.INFO: "🟢",
            self.DEBUG: "⚪"
        }
        return emojis[self]
    
    @property
    def text_color(self) -> str:
        """Couleur de texte appropriée (noir/blanc)"""
        # Calcul de la luminance pour déterminer le contraste
        hex_color = self.color_code.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return "#FFFFFF" if luminance < 0.5 else "#000000"
    
    @property
    def response_time_sla_minutes(self) -> int:
        """SLA de temps de réponse en minutes"""
        sla_times = {
            self.CRITICAL: 5,
            self.HIGH: 15,
            self.MEDIUM: 60,
            self.LOW: 240,
            self.INFO: 480,
            self.DEBUG: 1440
        }
        return sla_times[self]
    
    @property
    def escalation_threshold_minutes(self) -> int:
        """Seuil d'escalade en minutes"""
        return self.response_time_sla_minutes * 2
    
    def can_escalate_to(self, target_level: 'AlertLevel') -> bool:
        """Vérifie si peut escalader vers un niveau cible"""
        return target_level.priority > self.priority
    
    @classmethod
    def from_priority(cls, priority: int) -> 'AlertLevel':
        """Retourne le niveau basé sur la priorité numérique"""
        for level in cls:
            if level.priority >= priority:
                return level
        return cls.DEBUG


class AlertStatus(SmartEnum):
    """Statuts d'alerte avec machine à états"""
    PENDING = "pending"
    PROCESSING = "processing"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    
    @property
    def is_active(self) -> bool:
        """Indique si l'alerte est dans un état actif"""
        active_statuses = {
            self.PENDING, self.PROCESSING, self.ACTIVE, 
            self.ACKNOWLEDGED, self.ASSIGNED, self.IN_PROGRESS, self.ESCALATED
        }
        return self in active_statuses
    
    @property
    def is_final(self) -> bool:
        """Indique si le statut est final (non modifiable)"""
        final_statuses = {
            self.RESOLVED, self.CLOSED, self.FAILED, 
            self.EXPIRED, self.CANCELLED
        }
        return self in final_statuses
    
    @property
    def requires_action(self) -> bool:
        """Indique si le statut requiert une action"""
        action_required = {
            self.PENDING, self.ACTIVE, self.ESCALATED, self.FAILED
        }
        return self in action_required
    
    @property
    def color_code(self) -> str:
        """Couleur du statut"""
        colors = {
            self.PENDING: "#FFA500",      # Orange
            self.PROCESSING: "#1E90FF",   # Bleu
            self.ACTIVE: "#FF4500",       # Rouge-orange
            self.ACKNOWLEDGED: "#32CD32", # Vert
            self.ASSIGNED: "#9370DB",     # Violet
            self.IN_PROGRESS: "#FF69B4",  # Rose
            self.RESOLVED: "#008000",     # Vert foncé
            self.CLOSED: "#696969",       # Gris foncé
            self.SUPPRESSED: "#C0C0C0",   # Argent
            self.ESCALATED: "#DC143C",    # Rouge crimson
            self.FAILED: "#8B0000",       # Rouge foncé
            self.EXPIRED: "#708090",      # Gris ardoise
            self.CANCELLED: "#778899"     # Gris-bleu
        }
        return colors[self]
    
    def can_transition_to(self, new_status: 'AlertStatus') -> bool:
        """Vérifie si la transition est valide selon la machine à états"""
        valid_transitions = {
            self.PENDING: {self.PROCESSING, self.ACTIVE, self.SUPPRESSED, self.CANCELLED},
            self.PROCESSING: {self.ACTIVE, self.FAILED, self.SUPPRESSED, self.CANCELLED},
            self.ACTIVE: {self.ACKNOWLEDGED, self.ESCALATED, self.RESOLVED, self.SUPPRESSED, self.EXPIRED},
            self.ACKNOWLEDGED: {self.ASSIGNED, self.IN_PROGRESS, self.RESOLVED, self.ESCALATED},
            self.ASSIGNED: {self.IN_PROGRESS, self.ACKNOWLEDGED, self.ESCALATED, self.RESOLVED},
            self.IN_PROGRESS: {self.RESOLVED, self.ESCALATED, self.ACKNOWLEDGED},
            self.ESCALATED: {self.ACKNOWLEDGED, self.ASSIGNED, self.IN_PROGRESS, self.RESOLVED},
            self.SUPPRESSED: {self.PENDING, self.ACTIVE, self.RESOLVED},
            self.RESOLVED: {self.CLOSED, self.ACTIVE},  # Réouverture possible
            self.CLOSED: set(),  # État final
            self.FAILED: {self.PENDING, self.PROCESSING},
            self.EXPIRED: {self.PENDING, self.ACTIVE},
            self.CANCELLED: set()  # État final
        }
        return new_status in valid_transitions.get(self, set())
    
    def get_valid_transitions(self) -> Set['AlertStatus']:
        """Retourne les transitions valides depuis ce statut"""
        transitions = {
            self.PENDING: {self.PROCESSING, self.ACTIVE, self.SUPPRESSED, self.CANCELLED},
            self.PROCESSING: {self.ACTIVE, self.FAILED, self.SUPPRESSED, self.CANCELLED},
            self.ACTIVE: {self.ACKNOWLEDGED, self.ESCALATED, self.RESOLVED, self.SUPPRESSED, self.EXPIRED},
            self.ACKNOWLEDGED: {self.ASSIGNED, self.IN_PROGRESS, self.RESOLVED, self.ESCALATED},
            self.ASSIGNED: {self.IN_PROGRESS, self.ACKNOWLEDGED, self.ESCALATED, self.RESOLVED},
            self.IN_PROGRESS: {self.RESOLVED, self.ESCALATED, self.ACKNOWLEDGED},
            self.ESCALATED: {self.ACKNOWLEDGED, self.ASSIGNED, self.IN_PROGRESS, self.RESOLVED},
            self.SUPPRESSED: {self.PENDING, self.ACTIVE, self.RESOLVED},
            self.RESOLVED: {self.CLOSED, self.ACTIVE},
            self.CLOSED: set(),
            self.FAILED: {self.PENDING, self.PROCESSING},
            self.EXPIRED: {self.PENDING, self.ACTIVE},
            self.CANCELLED: set()
        }
        return transitions.get(self, set())


class WarningCategory(SmartEnum):
    """Catégories d'avertissement avec descriptions détaillées"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    CAPACITY = "capacity"
    COMPLIANCE = "compliance"
    QUALITY = "quality"
    BUSINESS = "business"
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATA = "data"
    
    @property
    def description(self) -> str:
        """Description détaillée de la catégorie"""
        descriptions = {
            self.PERFORMANCE: "Problèmes de performance système ou application",
            self.SECURITY: "Incidents et vulnérabilités de sécurité",
            self.AVAILABILITY: "Problèmes de disponibilité des services",
            self.CAPACITY: "Problèmes de capacité et utilisation des ressources",
            self.COMPLIANCE: "Non-conformité aux réglementations et politiques",
            self.QUALITY: "Problèmes de qualité des données ou services",
            self.BUSINESS: "Impact sur les processus métier",
            self.INFRASTRUCTURE: "Problèmes d'infrastructure technique",
            self.APPLICATION: "Erreurs et problèmes applicatifs",
            self.DATA: "Problèmes d'intégrité et qualité des données"
        }
        return descriptions[self]
    
    @property
    def icon(self) -> str:
        """Icône associée à la catégorie"""
        icons = {
            self.PERFORMANCE: "⚡",
            self.SECURITY: "🔒",
            self.AVAILABILITY: "🌐",
            self.CAPACITY: "📊",
            self.COMPLIANCE: "📋",
            self.QUALITY: "✅",
            self.BUSINESS: "💼",
            self.INFRASTRUCTURE: "🏗️",
            self.APPLICATION: "🔧",
            self.DATA: "💾"
        }
        return icons[self]
    
    @property
    def default_priority(self) -> int:
        """Priorité par défaut pour cette catégorie"""
        priorities = {
            self.SECURITY: 90,
            self.AVAILABILITY: 85,
            self.BUSINESS: 80,
            self.PERFORMANCE: 70,
            self.COMPLIANCE: 65,
            self.CAPACITY: 60,
            self.DATA: 55,
            self.QUALITY: 50,
            self.INFRASTRUCTURE: 45,
            self.APPLICATION: 40
        }
        return priorities[self]


class Priority(SmartEnum):
    """Niveaux de priorité avec système de classification"""
    P0 = "p0"  # Critique - Service down
    P1 = "p1"  # Élevé - Impact majeur
    P2 = "p2"  # Moyen - Impact modéré
    P3 = "p3"  # Bas - Impact mineur
    P4 = "p4"  # Info - Pas d'impact
    P5 = "p5"  # Minimal - Informatif
    
    @property
    def numeric_value(self) -> int:
        """Valeur numérique de la priorité"""
        return int(self.value[1])
    
    @property
    def sla_hours(self) -> int:
        """SLA en heures pour cette priorité"""
        sla_map = {
            self.P0: 1,    # 1 heure
            self.P1: 4,    # 4 heures
            self.P2: 24,   # 1 jour
            self.P3: 72,   # 3 jours
            self.P4: 168,  # 1 semaine
            self.P5: 720   # 1 mois
        }
        return sla_map[self]
    
    @property
    def escalation_threshold_hours(self) -> int:
        """Seuil d'escalade en heures"""
        return max(1, self.sla_hours // 2)
    
    @property
    def display_name(self) -> str:
        """Nom d'affichage"""
        names = {
            self.P0: "Critique",
            self.P1: "Élevé",
            self.P2: "Moyen",
            self.P3: "Bas",
            self.P4: "Info",
            self.P5: "Minimal"
        }
        return names[self]
    
    def is_higher_than(self, other: 'Priority') -> bool:
        """Vérifie si cette priorité est plus élevée qu'une autre"""
        return self.numeric_value < other.numeric_value  # P0 > P1 > P2 etc.


class Environment(SmartEnum):
    """Environnements avec configuration spécifique"""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TEST = "test"
    QA = "qa"
    LOCAL = "local"
    SANDBOX = "sandbox"
    
    @property
    def is_production_like(self) -> bool:
        """Indique si l'environnement est similaire à la production"""
        return self in {self.PRODUCTION, self.STAGING}
    
    @property
    def monitoring_level(self) -> str:
        """Niveau de monitoring pour cet environnement"""
        levels = {
            self.PRODUCTION: "comprehensive",
            self.STAGING: "moderate",
            self.QA: "moderate",
            self.TEST: "basic",
            self.DEVELOPMENT: "basic",
            self.LOCAL: "minimal",
            self.SANDBOX: "minimal"
        }
        return levels[self]
    
    @property
    def alert_retention_days(self) -> int:
        """Rétention des alertes en jours"""
        retention = {
            self.PRODUCTION: 365,
            self.STAGING: 90,
            self.QA: 60,
            self.TEST: 30,
            self.DEVELOPMENT: 7,
            self.LOCAL: 1,
            self.SANDBOX: 1
        }
        return retention[self]


class NotificationChannel(SmartEnum):
    """Canaux de notification avec configuration"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"
    CALL = "call"
    PAGER = "pager"
    DESKTOP = "desktop"
    MOBILE = "mobile"
    
    @property
    def is_real_time(self) -> bool:
        """Indique si le canal est temps réel"""
        real_time_channels = {
            self.PUSH, self.CALL, self.PAGER, self.DESKTOP, self.SMS
        }
        return self in real_time_channels
    
    @property
    def supports_rich_content(self) -> bool:
        """Indique si le canal supporte le contenu riche"""
        rich_channels = {
            self.EMAIL, self.SLACK, self.TEAMS, self.WEBHOOK, self.DESKTOP
        }
        return self in rich_channels
    
    @property
    def max_message_length(self) -> int:
        """Longueur maximale du message"""
        lengths = {
            self.EMAIL: 50000,
            self.SLACK: 40000,
            self.TEAMS: 28000,
            self.WEBHOOK: 100000,
            self.SMS: 160,
            self.PUSH: 512,
            self.CALL: 0,  # Pas de message texte
            self.PAGER: 256,
            self.DESKTOP: 1024,
            self.MOBILE: 1024
        }
        return lengths[self]
    
    @property
    def delivery_reliability(self) -> float:
        """Fiabilité de livraison estimée (0-1)"""
        reliability = {
            self.EMAIL: 0.99,
            self.SLACK: 0.98,
            self.TEAMS: 0.97,
            self.WEBHOOK: 0.95,
            self.SMS: 0.95,
            self.PUSH: 0.90,
            self.CALL: 0.85,
            self.PAGER: 0.99,
            self.DESKTOP: 0.80,
            self.MOBILE: 0.85
        }
        return reliability[self]


class EscalationLevel(SmartEnum):
    """Niveaux d'escalade hiérarchique"""
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    TEAM_LEAD = "team_lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    VP = "vp"
    EXECUTIVE = "executive"
    CEO = "ceo"
    
    @property
    def hierarchy_level(self) -> int:
        """Niveau hiérarchique (plus haut = plus important)"""
        levels = {
            self.L1: 1,
            self.L2: 2,
            self.L3: 3,
            self.TEAM_LEAD: 4,
            self.MANAGER: 5,
            self.DIRECTOR: 6,
            self.VP: 7,
            self.EXECUTIVE: 8,
            self.CEO: 9
        }
        return levels[self]
    
    @property
    def escalation_delay_minutes(self) -> int:
        """Délai avant escalade automatique"""
        delays = {
            self.L1: 15,
            self.L2: 30,
            self.L3: 60,
            self.TEAM_LEAD: 120,
            self.MANAGER: 240,
            self.DIRECTOR: 480,
            self.VP: 720,
            self.EXECUTIVE: 1440,
            self.CEO: 2880
        }
        return delays[self]
    
    def get_next_level(self) -> Optional['EscalationLevel']:
        """Retourne le niveau d'escalade suivant"""
        levels = list(EscalationLevel)
        try:
            current_index = levels.index(self)
            if current_index < len(levels) - 1:
                return levels[current_index + 1]
        except (ValueError, IndexError):
            pass
        return None


class CorrelationMethod(SmartEnum):
    """Méthodes de corrélation d'alertes"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    PATTERN = "pattern"
    ML_CLUSTERING = "ml_clustering"
    GRAPH_ANALYSIS = "graph_analysis"
    STATISTICAL = "statistical"
    RULE_BASED = "rule_based"
    
    @property
    def complexity(self) -> str:
        """Complexité de la méthode"""
        complexity_map = {
            self.TEMPORAL: "low",
            self.SPATIAL: "low",
            self.RULE_BASED: "medium",
            self.STATISTICAL: "medium",
            self.PATTERN: "medium",
            self.CAUSAL: "high",
            self.ML_CLUSTERING: "high",
            self.GRAPH_ANALYSIS: "high"
        }
        return complexity_map[self]
    
    @property
    def requires_ml(self) -> bool:
        """Indique si la méthode nécessite du ML"""
        ml_methods = {
            self.ML_CLUSTERING, self.PATTERN, self.STATISTICAL
        }
        return self in ml_methods
    
    @property
    def accuracy_score(self) -> float:
        """Score de précision estimé (0-1)"""
        scores = {
            self.TEMPORAL: 0.7,
            self.SPATIAL: 0.6,
            self.RULE_BASED: 0.8,
            self.STATISTICAL: 0.75,
            self.PATTERN: 0.85,
            self.CAUSAL: 0.9,
            self.ML_CLUSTERING: 0.88,
            self.GRAPH_ANALYSIS: 0.92
        }
        return scores[self]


class WorkflowStatus(SmartEnum):
    """États de workflow d'automatisation"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"
    
    @property
    def is_terminal(self) -> bool:
        """Indique si l'état est terminal"""
        terminal_states = {
            self.COMPLETED, self.FAILED, self.CANCELLED, self.TIMEOUT
        }
        return self in terminal_states
    
    @property
    def can_be_retried(self) -> bool:
        """Indique si le workflow peut être relancé"""
        retryable_states = {
            self.FAILED, self.TIMEOUT, self.CANCELLED
        }
        return self in retryable_states


class IncidentStatus(SmartEnum):
    """États d'incident avec cycle de vie complet"""
    OPEN = "open"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"
    POST_MORTEM = "post_mortem"
    
    @property
    def phase(self) -> str:
        """Phase du cycle de vie"""
        phases = {
            self.OPEN: "detection",
            self.TRIAGED: "analysis",
            self.INVESTIGATING: "analysis",
            self.IDENTIFIED: "response",
            self.MONITORING: "response",
            self.RESOLVED: "recovery",
            self.CLOSED: "recovery",
            self.POST_MORTEM: "learning"
        }
        return phases[self]


class ModelFramework(SmartEnum):
    """Frameworks de machine learning"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    SPARK_ML = "spark_ml"
    CUSTOM = "custom"
    
    @property
    def supports_gpu(self) -> bool:
        """Indique si le framework supporte le GPU"""
        gpu_frameworks = {
            self.TENSORFLOW, self.PYTORCH, self.XGBOOST, 
            self.LIGHTGBM, self.ONNX, self.SPARK_ML
        }
        return self in gpu_frameworks
    
    @property
    def is_distributed(self) -> bool:
        """Indique si le framework supporte la distribution"""
        distributed_frameworks = {
            self.TENSORFLOW, self.PYTORCH, self.SPARK_ML, self.XGBOOST
        }
        return self in distributed_frameworks


class SecurityLevel(SmartEnum):
    """Niveaux de sécurité avec contrôles associés"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"
    TOP_SECRET = "top_secret"
    
    @property
    def security_score(self) -> int:
        """Score de sécurité (0-100)"""
        scores = {
            self.NONE: 0,
            self.BASIC: 20,
            self.STANDARD: 50,
            self.HIGH: 75,
            self.CRITICAL: 90,
            self.TOP_SECRET: 100
        }
        return scores[self]
    
    @property
    def required_controls(self) -> List[str]:
        """Contrôles de sécurité requis"""
        controls = {
            self.NONE: [],
            self.BASIC: ["authentication"],
            self.STANDARD: ["authentication", "authorization", "logging"],
            self.HIGH: ["authentication", "authorization", "logging", "encryption", "monitoring"],
            self.CRITICAL: ["authentication", "authorization", "logging", "encryption", "monitoring", "mfa", "audit"],
            self.TOP_SECRET: ["authentication", "authorization", "logging", "encryption", "monitoring", "mfa", "audit", "isolation", "compliance"]
        }
        return controls[self]


# Énumérations de flags pour combinaisons
class PermissionFlag(Flag):
    """Flags de permissions combinables"""
    NONE = 0
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    ADMIN = auto()
    
    # Combinaisons courantes
    READ_WRITE = READ | WRITE
    ALL_EXCEPT_DELETE = READ | WRITE | ADMIN
    ALL = READ | WRITE | DELETE | ADMIN


class FeatureFlag(Flag):
    """Flags de fonctionnalités"""
    NONE = 0
    ALERTS = auto()
    NOTIFICATIONS = auto()
    WORKFLOWS = auto()
    ANALYTICS = auto()
    ML_FEATURES = auto()
    REPORTING = auto()
    INTEGRATIONS = auto()
    
    # Combinaisons par niveau
    BASIC = ALERTS | NOTIFICATIONS
    STANDARD = BASIC | WORKFLOWS | ANALYTICS
    PREMIUM = STANDARD | ML_FEATURES | REPORTING | INTEGRATIONS
    ENTERPRISE = PREMIUM


# Classes utilitaires pour les énumérations
class EnumRegistry:
    """Registre central des énumérations"""
    _enums = {}
    
    @classmethod
    def register(cls, enum_class):
        """Enregistre une énumération"""
        cls._enums[enum_class.__name__] = enum_class
    
    @classmethod
    def get_enum(cls, name: str):
        """Récupère une énumération par nom"""
        return cls._enums.get(name)
    
    @classmethod
    def list_enums(cls) -> List[str]:
        """Liste toutes les énumérations enregistrées"""
        return list(cls._enums.keys())


# Auto-enregistrement des énumérations
for name, obj in globals().items():
    if (isinstance(obj, type) and 
        issubclass(obj, (Enum, SmartEnum)) and 
        obj not in (Enum, SmartEnum, Flag, IntEnum)):
        EnumRegistry.register(obj)


__all__ = [
    # Énumérations principales
    'SmartEnum', 'AlertLevel', 'AlertStatus', 'WarningCategory', 'Priority', 
    'Environment', 'NotificationChannel', 'EscalationLevel', 'CorrelationMethod',
    'WorkflowStatus', 'IncidentStatus', 'ModelFramework', 'SecurityLevel',
    
    # Flags
    'PermissionFlag', 'FeatureFlag',
    
    # Utilitaires
    'EnumRegistry'
]
    """Catégories de warnings avec métadonnées"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    ML_MODEL = "ml_model"
    API = "api"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"
    USER_BEHAVIOR = "user_behavior"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    COMPLIANCE = "compliance"
    BUSINESS_LOGIC = "business_logic"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    BACKUP = "backup"
    DISASTER_RECOVERY = "disaster_recovery"
    CAPACITY = "capacity"
    COST_OPTIMIZATION = "cost_optimization"
    USER_EXPERIENCE = "user_experience"
    UNKNOWN = "unknown"
    
    @property
    def description(self) -> str:
        """Description de la catégorie"""
        descriptions = {
            self.PERFORMANCE: "Problèmes de performance et latence",
            self.SECURITY: "Incidents de sécurité et vulnérabilités",
            self.ML_MODEL: "Problèmes liés aux modèles ML/IA",
            self.API: "Erreurs et problèmes d'API",
            self.DATABASE: "Problèmes de base de données",
            self.INFRASTRUCTURE: "Problèmes d'infrastructure",
            self.USER_BEHAVIOR: "Comportements utilisateur anormaux",
            self.DATA_QUALITY: "Problèmes de qualité des données",
            self.SYSTEM_HEALTH: "Santé générale du système",
            self.AUTHENTICATION: "Problèmes d'authentification",
            self.AUTHORIZATION: "Problèmes d'autorisation",
            self.COMPLIANCE: "Violations de conformité",
            self.BUSINESS_LOGIC: "Erreurs de logique métier",
            self.INTEGRATION: "Problèmes d'intégration",
            self.MONITORING: "Problèmes de monitoring",
            self.BACKUP: "Problèmes de sauvegarde",
            self.DISASTER_RECOVERY: "Problèmes de reprise d'activité",
            self.CAPACITY: "Problèmes de capacité",
            self.COST_OPTIMIZATION: "Optimisation des coûts",
            self.USER_EXPERIENCE: "Problèmes d'expérience utilisateur",
            self.UNKNOWN: "Catégorie non déterminée"
        }
        return descriptions.get(self, "Description non disponible")
    
    @property
    def icon(self) -> str:
        """Icône associée à la catégorie"""
        icons = {
            self.PERFORMANCE: "⚡",
            self.SECURITY: "🔒",
            self.ML_MODEL: "🤖",
            self.API: "🔌",
            self.DATABASE: "🗃️",
            self.INFRASTRUCTURE: "🏗️",
            self.USER_BEHAVIOR: "👤",
            self.DATA_QUALITY: "📊",
            self.SYSTEM_HEALTH: "❤️",
            self.AUTHENTICATION: "🔑",
            self.AUTHORIZATION: "🛡️",
            self.COMPLIANCE: "📋",
            self.BUSINESS_LOGIC: "💼",
            self.INTEGRATION: "🔗",
            self.MONITORING: "👁️",
            self.BACKUP: "💾",
            self.DISASTER_RECOVERY: "🚨",
            self.CAPACITY: "📈",
            self.COST_OPTIMIZATION: "💰",
            self.USER_EXPERIENCE: "😊",
            self.UNKNOWN: "❓"
        }
        return icons.get(self, "❓")


class NotificationChannel(str, Enum):
    """Canaux de notification avec configuration"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    PUSH_NOTIFICATION = "push_notification"
    IN_APP = "in_app"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    JIRA = "jira"
    
    @property
    def supports_rich_formatting(self) -> bool:
        """Indique si le canal supporte le formatage riche"""
        rich_channels = {
            self.SLACK, self.TEAMS, self.DISCORD, 
            self.EMAIL, self.IN_APP
        }
        return self in rich_channels
    
    @property
    def is_real_time(self) -> bool:
        """Indique si le canal est temps réel"""
        realtime_channels = {
            self.SLACK, self.TEAMS, self.DISCORD, 
            self.TELEGRAM, self.PUSH_NOTIFICATION, 
            self.IN_APP, self.SMS
        }
        return self in realtime_channels
    
    @property
    def max_message_length(self) -> int:
        """Longueur maximale du message"""
        limits = {
            self.SLACK: 4000,
            self.EMAIL: 50000,
            self.SMS: 160,
            self.WEBHOOK: 10000,
            self.TEAMS: 4000,
            self.DISCORD: 2000,
            self.TELEGRAM: 4096,
            self.PUSH_NOTIFICATION: 100,
            self.IN_APP: 1000,
            self.PAGERDUTY: 1024,
            self.OPSGENIE: 15000,
            self.JIRA: 32767
        }
        return limits.get(self, 1000)


class TenantStatus(str, Enum):
    """Statuts des tenants"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"
    
    @property
    def allows_operations(self) -> bool:
        """Indique si les opérations sont autorisées"""
        return self == self.ACTIVE


class SecurityLevel(str, Enum):
    """Niveaux de sécurité"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"
    
    @property
    def numeric_level(self) -> int:
        """Niveau numérique de sécurité"""
        levels = {
            self.PUBLIC: 1,
            self.INTERNAL: 2,
            self.CONFIDENTIAL: 3,
            self.RESTRICTED: 4,
            self.SECRET: 5
        }
        return levels[self]


class Environment(str, Enum):
    """Environnements d'exécution"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @property
    def short_name(self) -> str:
        """Nom court de l'environnement"""
        short_names = {
            self.DEVELOPMENT: "dev",
            self.TESTING: "test",
            self.STAGING: "stage",
            self.PRODUCTION: "prod"
        }
        return short_names[self]
    
    @property
    def is_production_like(self) -> bool:
        """Indique si l'environnement est proche de la production"""
        return self in {self.STAGING, self.PRODUCTION}


class MetricType(str, Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    
    @property
    def aggregatable(self) -> bool:
        """Indique si la métrique peut être agrégée"""
        return self in {self.COUNTER, self.HISTOGRAM, self.SUMMARY}


class TimeUnit(str, Enum):
    """Unités de temps"""
    NANOSECONDS = "ns"
    MICROSECONDS = "μs"
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "m"
    HOURS = "h"
    DAYS = "d"
    
    @property
    def to_seconds_multiplier(self) -> float:
        """Multiplicateur pour conversion en secondes"""
        multipliers = {
            self.NANOSECONDS: 1e-9,
            self.MICROSECONDS: 1e-6,
            self.MILLISECONDS: 1e-3,
            self.SECONDS: 1.0,
            self.MINUTES: 60.0,
            self.HOURS: 3600.0,
            self.DAYS: 86400.0
        }
        return multipliers[self]


class DataSize(str, Enum):
    """Unités de taille de données"""
    BYTES = "B"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    GIGABYTES = "GB"
    TERABYTES = "TB"
    PETABYTES = "PB"
    
    @property
    def to_bytes_multiplier(self) -> int:
        """Multiplicateur pour conversion en bytes"""
        multipliers = {
            self.BYTES: 1,
            self.KILOBYTES: 1024,
            self.MEGABYTES: 1024**2,
            self.GIGABYTES: 1024**3,
            self.TERABYTES: 1024**4,
            self.PETABYTES: 1024**5
        }
        return multipliers[self]


class ProcessingStatus(str, Enum):
    """Statuts de traitement"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    
    @property
    def is_terminal(self) -> bool:
        """Indique si le statut est terminal"""
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


class Priority(IntEnum):
    """Niveaux de priorité"""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6
    
    @property
    def emoji(self) -> str:
        """Emoji associé à la priorité"""
        emojis = {
            self.LOWEST: "⬇️",
            self.LOW: "🔽",
            self.NORMAL: "➡️",
            self.HIGH: "🔼",
            self.HIGHEST: "⬆️",
            self.CRITICAL: "🚨"
        }
        return emojis.get(self, "➡️")


# Constantes système
SYSTEM_CONSTANTS = {
    "MAX_ALERT_MESSAGE_LENGTH": 2000,
    "MAX_METADATA_SIZE_BYTES": 10240,  # 10KB
    "DEFAULT_PAGE_SIZE": 50,
    "MAX_PAGE_SIZE": 1000,
    "DEFAULT_CACHE_TTL_SECONDS": 300,  # 5 minutes
    "MAX_CACHE_TTL_SECONDS": 86400,    # 24 heures
    "DEFAULT_TIMEOUT_SECONDS": 30,
    "MAX_RETRIES": 3,
    "RATE_LIMIT_REQUESTS_PER_MINUTE": 1000,
    "MAX_CONCURRENT_PROCESSING": 100
}

# Patterns de validation
VALIDATION_PATTERNS = {
    "TENANT_ID": r"^[a-zA-Z0-9_-]{1,255}$",
    "CORRELATION_ID": r"^[a-zA-Z0-9_-]{1,128}$",
    "EMAIL": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "PHONE": r"^\+?1?[0-9]{10,15}$",
    "URL": r"^https?://[^\s/$.?#].[^\s]*$",
    "VERSION": r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$",
    "HEX_COLOR": r"^#[0-9A-Fa-f]{6}$",
    "TIMEZONE": r"^[A-Z][a-z]+/[A-Z][a-z_]+$"
}

# Messages d'erreur par défaut
DEFAULT_ERROR_MESSAGES = {
    "REQUIRED_FIELD": "Ce champ est requis",
    "INVALID_FORMAT": "Format invalide",
    "VALUE_TOO_LONG": "Valeur trop longue",
    "VALUE_TOO_SHORT": "Valeur trop courte",
    "INVALID_CHOICE": "Choix invalide",
    "INVALID_TYPE": "Type invalide",
    "INVALID_RANGE": "Valeur hors limites",
    "DUPLICATE_VALUE": "Valeur déjà existante",
    "FOREIGN_KEY_ERROR": "Référence invalide",
    "PERMISSION_DENIED": "Permission refusée"
}
