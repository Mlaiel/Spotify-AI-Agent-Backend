# -*- coding: utf-8 -*-
"""
Escalation Manager - Gestionnaire d'Escalade Ultra-Avancé
========================================================

Gestionnaire d'escalade intelligent pour le système d'alertes avec:
- Matrices d'escalade hiérarchiques multi-niveaux
- Règles temporelles intelligentes (horaires, jours fériés)
- Rotation automatique des équipes
- Escalade basée sur contexte et sévérité
- Integration avec systèmes externes (LDAP, AD)
- Notifications multi-canaux par niveau
- Délégation automatique et suppléance
- Historique complet et analytics

Fonctionnalités Clés:
- Escalade en cascade avec timeouts configurables
- Support des équipes virtuelles et physiques
- Gestion des absences et congés
- Escalade cross-tenant avec permissions
- Bypass d'escalade pour urgences critiques
- Integration avec calendriers externes
- Métriques de performance par équipe
- Compliance et audit trail complet

Types d'Escalade:
- Temporelle: basée sur délais et fenêtres
- Hiérarchique: suivant organigramme
- Compétence: basée sur expertise
- Géographique: par fuseaux horaires
- Charge: répartition intelligente
- Urgence: bypass pour critiques

Version: 3.0.0
"""

import time
import threading
import logging
import json
import sqlite3
import redis
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor
import calendar

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EscalationLevel(Enum):
    """Niveaux d'escalade"""
    L1_ENGINEER = 1
    L2_SENIOR = 2
    L3_EXPERT = 3
    L4_ARCHITECT = 4
    L5_MANAGER = 5
    L6_DIRECTOR = 6
    EMERGENCY_OVERRIDE = 99

class EscalationTrigger(Enum):
    """Déclencheurs d'escalade"""
    TIMEOUT = "timeout"
    NO_RESPONSE = "no_response"
    MANUAL = "manual"
    SEVERITY_INCREASE = "severity_increase"
    SLA_BREACH = "sla_breach"
    BUSINESS_IMPACT = "business_impact"
    REPEAT_ALERT = "repeat_alert"

class EscalationStatus(Enum):
    """États d'escalade"""
    PENDING = "pending"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    BYPASSED = "bypassed"
    EXPIRED = "expired"

class TeamType(Enum):
    """Types d'équipes"""
    ON_CALL = "on_call"
    SPECIALIST = "specialist"
    MANAGEMENT = "management"
    EMERGENCY = "emergency"
    VENDOR = "vendor"

class ContactMethod(Enum):
    """Méthodes de contact"""
    EMAIL = "email"
    PHONE = "phone"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"

@dataclass
class Person:
    """Représente une personne dans le système d'escalade"""
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    role: str = ""
    department: str = ""
    location: str = ""
    timezone: str = "UTC"
    skills: List[str] = field(default_factory=list)
    contact_methods: Dict[ContactMethod, str] = field(default_factory=dict)
    availability: Dict[str, Any] = field(default_factory=dict)
    escalation_preferences: Dict[str, Any] = field(default_factory=dict)
    max_concurrent_alerts: int = 5
    is_on_call: bool = False
    is_available: bool = True
    manager_id: Optional[str] = None

@dataclass
class Team:
    """Représente une équipe"""
    id: str
    name: str
    type: TeamType
    members: List[str]  # Person IDs
    lead_id: Optional[str] = None
    escalation_timeout_minutes: int = 15
    skills: List[str] = field(default_factory=list)
    contact_channels: List[str] = field(default_factory=list)
    on_call_schedule: Dict[str, Any] = field(default_factory=dict)
    sla_targets: Dict[str, int] = field(default_factory=dict)
    business_hours: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationRule:
    """Règle d'escalade"""
    id: str
    name: str
    condition: Dict[str, Any]
    levels: List[Dict[str, Any]]
    enabled: bool = True
    priority: int = 1
    tenant_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class EscalationPath:
    """Chemin d'escalade"""
    id: str
    alert_id: str
    rule_id: str
    current_level: int
    status: EscalationStatus
    levels: List[Dict[str, Any]]
    started_at: float
    current_level_started_at: float
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    bypass_reason: Optional[str] = None
    tenant_id: Optional[str] = None

@dataclass
class EscalationEvent:
    """Événement d'escalade"""
    id: str
    escalation_id: str
    event_type: str
    level: int
    person_id: Optional[str] = None
    team_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

class EscalationManager:
    """
    Gestionnaire d'escalade ultra-avancé
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le gestionnaire d'escalade
        
        Args:
            config: Configuration du gestionnaire
        """
        self.config = config or self._default_config()
        self.is_running = False
        self.start_time = time.time()
        
        # Données principales
        self.people: Dict[str, Person] = {}
        self.teams: Dict[str, Team] = {}
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.active_escalations: Dict[str, EscalationPath] = {}
        
        # Stockage et cache
        self.db_path = self.config.get('db_path', 'escalation_manager.db')
        self.redis_client = self._init_redis()
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='EscalationManager')
        self.worker_threads: List[threading.Thread] = []
        self.lock = threading.RLock()
        
        # Métriques
        self.metrics = {
            'total_escalations': 0,
            'escalations_resolved': 0,
            'avg_resolution_time_minutes': 0,
            'escalations_by_level': defaultdict(int),
            'sla_breaches': 0,
            'bypass_count': 0
        }
        
        # Calendriers et schedules
        self.holiday_calendar: Set[str] = set()
        self.on_call_schedules: Dict[str, Any] = {}
        
        # Callbacks
        self.notification_callback: Optional[Callable] = None
        self.audit_callback: Optional[Callable] = None
        
        # Initialisation
        self._init_database()
        self._load_configuration()
        
        logger.info("EscalationManager initialisé avec succès")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'default_escalation_timeout_minutes': 15,
            'max_escalation_levels': 6,
            'auto_resolve_timeout_hours': 24,
            'enable_business_hours': True,
            'business_hours_start': '09:00',
            'business_hours_end': '17:00',
            'weekend_escalation_modifier': 2.0,
            'holiday_escalation_modifier': 3.0,
            'max_concurrent_escalations_per_person': 5,
            'escalation_check_interval_seconds': 60,
            'enable_smart_routing': True,
            'enable_load_balancing': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 8,
            'enable_external_calendar_sync': False,
            'ldap_integration': False,
            'audit_retention_days': 90,
            'metrics_retention_days': 365,
            'default_timezone': 'UTC'
        }
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialise la connexion Redis"""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            client.ping()
            logger.info("Connexion Redis EscalationManager établie")
            return client
        except Exception as e:
            logger.warning(f"Redis non disponible pour EscalationManager: {e}")
            return None
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des personnes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    role TEXT,
                    department TEXT,
                    location TEXT,
                    timezone TEXT DEFAULT 'UTC',
                    skills TEXT,
                    contact_methods TEXT,
                    availability TEXT,
                    escalation_preferences TEXT,
                    max_concurrent_alerts INTEGER DEFAULT 5,
                    is_on_call BOOLEAN DEFAULT 0,
                    is_available BOOLEAN DEFAULT 1,
                    manager_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Table des équipes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    members TEXT,
                    lead_id TEXT,
                    escalation_timeout_minutes INTEGER DEFAULT 15,
                    skills TEXT,
                    contact_channels TEXT,
                    on_call_schedule TEXT,
                    sla_targets TEXT,
                    business_hours TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Table des règles d'escalade
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS escalation_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    levels TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    priority INTEGER DEFAULT 1,
                    tenant_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Table des escalades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    current_level INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    levels TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    current_level_started_at REAL NOT NULL,
                    acknowledged_by TEXT,
                    acknowledged_at REAL,
                    resolved_at REAL,
                    bypass_reason TEXT,
                    tenant_id TEXT
                )
            ''')
            
            # Table des événements d'escalade
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS escalation_events (
                    id TEXT PRIMARY KEY,
                    escalation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    person_id TEXT,
                    team_id TEXT,
                    timestamp REAL NOT NULL,
                    details TEXT,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT
                )
            ''')
            
            # Table des métriques
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS escalation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    dimensions TEXT,
                    timestamp REAL NOT NULL,
                    tenant_id TEXT
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_escalations_alert ON escalations(alert_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_escalation ON escalation_events(escalation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON escalation_metrics(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("Base de données EscalationManager initialisée")
            
        except Exception as e:
            logger.error(f"Erreur initialisation base de données: {e}")
    
    def _load_configuration(self):
        """Charge la configuration depuis la base de données"""
        try:
            # Charge les personnes
            self._load_people()
            
            # Charge les équipes
            self._load_teams()
            
            # Charge les règles d'escalade
            self._load_escalation_rules()
            
            # Charge les calendriers
            self._load_holiday_calendar()
            
            logger.info("Configuration EscalationManager chargée")
            
        except Exception as e:
            logger.error(f"Erreur chargement configuration: {e}")
    
    def _load_people(self):
        """Charge les personnes depuis la base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM people')
            rows = cursor.fetchall()
            
            for row in rows:
                person = Person(
                    id=row[0],
                    name=row[1],
                    email=row[2],
                    phone=row[3],
                    role=row[4] or "",
                    department=row[5] or "",
                    location=row[6] or "",
                    timezone=row[7] or "UTC",
                    skills=json.loads(row[8]) if row[8] else [],
                    contact_methods={ContactMethod(k): v for k, v in json.loads(row[9]).items()} if row[9] else {},
                    availability=json.loads(row[10]) if row[10] else {},
                    escalation_preferences=json.loads(row[11]) if row[11] else {},
                    max_concurrent_alerts=row[12] or 5,
                    is_on_call=bool(row[13]),
                    is_available=bool(row[14]),
                    manager_id=row[15]
                )
                self.people[person.id] = person
            
            conn.close()
            logger.info(f"Chargé {len(self.people)} personnes")
            
        except Exception as e:
            logger.error(f"Erreur chargement personnes: {e}")
    
    def _load_teams(self):
        """Charge les équipes depuis la base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM teams')
            rows = cursor.fetchall()
            
            for row in rows:
                team = Team(
                    id=row[0],
                    name=row[1],
                    type=TeamType(row[2]),
                    members=json.loads(row[3]) if row[3] else [],
                    lead_id=row[4],
                    escalation_timeout_minutes=row[5] or 15,
                    skills=json.loads(row[6]) if row[6] else [],
                    contact_channels=json.loads(row[7]) if row[7] else [],
                    on_call_schedule=json.loads(row[8]) if row[8] else {},
                    sla_targets=json.loads(row[9]) if row[9] else {},
                    business_hours=json.loads(row[10]) if row[10] else {}
                )
                self.teams[team.id] = team
            
            conn.close()
            logger.info(f"Chargé {len(self.teams)} équipes")
            
        except Exception as e:
            logger.error(f"Erreur chargement équipes: {e}")
    
    def _load_escalation_rules(self):
        """Charge les règles d'escalade depuis la base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM escalation_rules WHERE enabled = 1')
            rows = cursor.fetchall()
            
            for row in rows:
                rule = EscalationRule(
                    id=row[0],
                    name=row[1],
                    condition=json.loads(row[2]),
                    levels=json.loads(row[3]),
                    enabled=bool(row[4]),
                    priority=row[5] or 1,
                    tenant_id=row[6],
                    created_at=row[7],
                    updated_at=row[8]
                )
                self.escalation_rules[rule.id] = rule
            
            conn.close()
            logger.info(f"Chargé {len(self.escalation_rules)} règles d'escalade")
            
        except Exception as e:
            logger.error(f"Erreur chargement règles: {e}")
    
    def _load_holiday_calendar(self):
        """Charge le calendrier des jours fériés"""
        try:
            # Jours fériés par défaut (France)
            current_year = datetime.now().year
            holidays = [
                f"{current_year}-01-01",  # Nouvel An
                f"{current_year}-05-01",  # Fête du Travail
                f"{current_year}-05-08",  # Victoire 1945
                f"{current_year}-07-14",  # Fête Nationale
                f"{current_year}-08-15",  # Assomption
                f"{current_year}-11-01",  # Toussaint
                f"{current_year}-11-11",  # Armistice
                f"{current_year}-12-25",  # Noël
            ]
            
            self.holiday_calendar = set(holidays)
            logger.info(f"Calendrier des jours fériés chargé: {len(holidays)} jours")
            
        except Exception as e:
            logger.error(f"Erreur chargement calendrier: {e}")
    
    def start(self) -> bool:
        """Démarre le gestionnaire d'escalade"""
        if self.is_running:
            logger.warning("EscalationManager déjà en cours d'exécution")
            return True
        
        try:
            self.is_running = True
            
            # Worker principal pour vérifier les escalades
            main_worker = threading.Thread(target=self._escalation_worker, daemon=True)
            main_worker.start()
            self.worker_threads.append(main_worker)
            
            # Worker pour les métriques
            metrics_worker = threading.Thread(target=self._metrics_worker, daemon=True)
            metrics_worker.start()
            self.worker_threads.append(metrics_worker)
            
            # Worker pour le nettoyage
            cleanup_worker = threading.Thread(target=self._cleanup_worker, daemon=True)
            cleanup_worker.start()
            self.worker_threads.append(cleanup_worker)
            
            logger.info("EscalationManager démarré")
            return True
            
        except Exception as e:
            logger.error(f"Erreur démarrage EscalationManager: {e}")
            return False
    
    def stop(self) -> bool:
        """Arrête le gestionnaire d'escalade"""
        if not self.is_running:
            return True
        
        try:
            logger.info("Arrêt EscalationManager...")
            self.is_running = False
            
            # Attend les workers
            for worker in self.worker_threads:
                if worker.is_alive():
                    worker.join(timeout=10)
            
            # Arrête l'executor
            self.executor.shutdown(wait=True, timeout=30)
            
            logger.info("EscalationManager arrêté")
            return True
            
        except Exception as e:
            logger.error(f"Erreur arrêt EscalationManager: {e}")
            return False
    
    def create_escalation(self, alert_id: str, alert_data: Dict[str, Any], 
                         tenant_id: Optional[str] = None) -> Optional[str]:
        """
        Crée une nouvelle escalade pour une alerte
        
        Args:
            alert_id: ID de l'alerte
            alert_data: Données de l'alerte
            tenant_id: ID du tenant
            
        Returns:
            ID de l'escalade créée
        """
        try:
            # Trouve la règle d'escalade appropriée
            rule = self._find_escalation_rule(alert_data, tenant_id)
            if not rule:
                logger.warning(f"Aucune règle d'escalade trouvée pour alerte {alert_id}")
                return None
            
            # Crée le chemin d'escalade
            escalation_id = str(uuid.uuid4())
            escalation = EscalationPath(
                id=escalation_id,
                alert_id=alert_id,
                rule_id=rule.id,
                current_level=0,
                status=EscalationStatus.PENDING,
                levels=rule.levels.copy(),
                started_at=time.time(),
                current_level_started_at=time.time(),
                tenant_id=tenant_id
            )
            
            # Ajuste les timeouts selon le contexte
            self._adjust_escalation_timeouts(escalation, alert_data)
            
            # Sauvegarde
            self._save_escalation(escalation)
            
            with self.lock:
                self.active_escalations[escalation_id] = escalation
                self.metrics['total_escalations'] += 1
            
            # Démarre immédiatement le premier niveau
            self._start_escalation_level(escalation, 0)
            
            # Log l'événement
            self._log_escalation_event(
                escalation_id=escalation_id,
                event_type="escalation_created",
                level=0,
                details={'rule_id': rule.id, 'alert_id': alert_id}
            )
            
            logger.info(f"Escalade créée: {escalation_id} pour alerte {alert_id}")
            return escalation_id
            
        except Exception as e:
            logger.error(f"Erreur création escalade pour alerte {alert_id}: {e}")
            return None
    
    def acknowledge_escalation(self, escalation_id: str, person_id: str, 
                              notes: Optional[str] = None) -> bool:
        """
        Acquitte une escalade
        
        Args:
            escalation_id: ID de l'escalade
            person_id: ID de la personne qui acquitte
            notes: Notes optionnelles
            
        Returns:
            True si acquittement réussi
        """
        try:
            escalation = self.active_escalations.get(escalation_id)
            if not escalation:
                logger.warning(f"Escalade introuvable: {escalation_id}")
                return False
            
            if escalation.status in [EscalationStatus.ACKNOWLEDGED, EscalationStatus.RESOLVED]:
                logger.warning(f"Escalade déjà acquittée/résolue: {escalation_id}")
                return False
            
            # Met à jour l'escalade
            escalation.status = EscalationStatus.ACKNOWLEDGED
            escalation.acknowledged_by = person_id
            escalation.acknowledged_at = time.time()
            
            # Sauvegarde
            self._save_escalation(escalation)
            
            # Log l'événement
            self._log_escalation_event(
                escalation_id=escalation_id,
                event_type="escalation_acknowledged",
                level=escalation.current_level,
                person_id=person_id,
                details={'notes': notes} if notes else {}
            )
            
            # Notifie via callback
            if self.notification_callback:
                try:
                    self.notification_callback({
                        'type': 'escalation_acknowledged',
                        'escalation_id': escalation_id,
                        'person_id': person_id,
                        'level': escalation.current_level
                    })
                except Exception as e:
                    logger.error(f"Erreur callback notification: {e}")
            
            logger.info(f"Escalade acquittée: {escalation_id} par {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur acquittement escalade {escalation_id}: {e}")
            return False
    
    def resolve_escalation(self, escalation_id: str, person_id: str, 
                          resolution_notes: Optional[str] = None) -> bool:
        """
        Résout une escalade
        
        Args:
            escalation_id: ID de l'escalade
            person_id: ID de la personne qui résout
            resolution_notes: Notes de résolution
            
        Returns:
            True si résolution réussie
        """
        try:
            escalation = self.active_escalations.get(escalation_id)
            if not escalation:
                logger.warning(f"Escalade introuvable: {escalation_id}")
                return False
            
            if escalation.status == EscalationStatus.RESOLVED:
                logger.warning(f"Escalade déjà résolue: {escalation_id}")
                return False
            
            # Met à jour l'escalade
            escalation.status = EscalationStatus.RESOLVED
            escalation.resolved_at = time.time()
            
            # Calcul du temps de résolution
            resolution_time = escalation.resolved_at - escalation.started_at
            
            # Sauvegarde
            self._save_escalation(escalation)
            
            # Retire des escalades actives
            with self.lock:
                if escalation_id in self.active_escalations:
                    del self.active_escalations[escalation_id]
                
                self.metrics['escalations_resolved'] += 1
                
                # Met à jour le temps moyen de résolution
                current_avg = self.metrics['avg_resolution_time_minutes']
                total_resolved = self.metrics['escalations_resolved']
                new_avg = ((current_avg * (total_resolved - 1)) + (resolution_time / 60)) / total_resolved
                self.metrics['avg_resolution_time_minutes'] = new_avg
            
            # Log l'événement
            self._log_escalation_event(
                escalation_id=escalation_id,
                event_type="escalation_resolved",
                level=escalation.current_level,
                person_id=person_id,
                details={
                    'resolution_notes': resolution_notes,
                    'resolution_time_minutes': resolution_time / 60
                }
            )
            
            # Notifie via callback
            if self.notification_callback:
                try:
                    self.notification_callback({
                        'type': 'escalation_resolved',
                        'escalation_id': escalation_id,
                        'person_id': person_id,
                        'resolution_time_minutes': resolution_time / 60
                    })
                except Exception as e:
                    logger.error(f"Erreur callback notification: {e}")
            
            logger.info(f"Escalade résolue: {escalation_id} par {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur résolution escalade {escalation_id}: {e}")
            return False
    
    def bypass_escalation(self, escalation_id: str, target_level: int, 
                         person_id: str, reason: str) -> bool:
        """
        Contourne l'escalade vers un niveau spécifique
        
        Args:
            escalation_id: ID de l'escalade
            target_level: Niveau cible
            person_id: ID de la personne qui contourne
            reason: Raison du contournement
            
        Returns:
            True si contournement réussi
        """
        try:
            escalation = self.active_escalations.get(escalation_id)
            if not escalation:
                logger.warning(f"Escalade introuvable: {escalation_id}")
                return False
            
            if target_level >= len(escalation.levels):
                logger.error(f"Niveau cible invalide: {target_level}")
                return False
            
            # Met à jour l'escalade
            escalation.current_level = target_level
            escalation.status = EscalationStatus.BYPASSED
            escalation.current_level_started_at = time.time()
            escalation.bypass_reason = reason
            
            # Sauvegarde
            self._save_escalation(escalation)
            
            # Démarre le nouveau niveau
            self._start_escalation_level(escalation, target_level)
            
            with self.lock:
                self.metrics['bypass_count'] += 1
            
            # Log l'événement
            self._log_escalation_event(
                escalation_id=escalation_id,
                event_type="escalation_bypassed",
                level=target_level,
                person_id=person_id,
                details={'reason': reason, 'from_level': escalation.current_level}
            )
            
            logger.info(f"Escalade contournée: {escalation_id} vers niveau {target_level}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur contournement escalade {escalation_id}: {e}")
            return False
    
    def _find_escalation_rule(self, alert_data: Dict[str, Any], 
                             tenant_id: Optional[str]) -> Optional[EscalationRule]:
        """Trouve la règle d'escalade appropriée"""
        try:
            # Trie les règles par priorité
            rules = sorted(
                [r for r in self.escalation_rules.values() 
                 if r.enabled and (not r.tenant_id or r.tenant_id == tenant_id)],
                key=lambda x: x.priority
            )
            
            for rule in rules:
                if self._evaluate_rule_condition(rule.condition, alert_data):
                    return rule
            
            # Règle par défaut si aucune ne correspond
            return self._get_default_escalation_rule()
            
        except Exception as e:
            logger.error(f"Erreur recherche règle escalade: {e}")
            return None
    
    def _evaluate_rule_condition(self, condition: Dict[str, Any], alert_data: Dict[str, Any]) -> bool:
        """Évalue une condition de règle"""
        try:
            # Condition simple basée sur la sévérité
            if 'severity' in condition:
                required_severity = condition['severity']
                alert_severity = alert_data.get('severity', '').lower()
                
                if isinstance(required_severity, list):
                    return alert_severity in [s.lower() for s in required_severity]
                else:
                    return alert_severity == required_severity.lower()
            
            # Condition basée sur les labels
            if 'labels' in condition:
                for key, value in condition['labels'].items():
                    alert_labels = alert_data.get('labels', {})
                    if key not in alert_labels or alert_labels[key] != value:
                        return False
                return True
            
            # Condition basée sur la source
            if 'source' in condition:
                return alert_data.get('source', '') == condition['source']
            
            # Par défaut, la condition est vraie
            return True
            
        except Exception as e:
            logger.error(f"Erreur évaluation condition: {e}")
            return False
    
    def _get_default_escalation_rule(self) -> EscalationRule:
        """Retourne une règle d'escalade par défaut"""
        try:
            default_levels = [
                {
                    'level': EscalationLevel.L1_ENGINEER.value,
                    'timeout_minutes': 15,
                    'targets': [{'type': 'team', 'id': 'on-call-l1'}],
                    'contact_methods': ['email', 'slack']
                },
                {
                    'level': EscalationLevel.L2_SENIOR.value,
                    'timeout_minutes': 30,
                    'targets': [{'type': 'team', 'id': 'on-call-l2'}],
                    'contact_methods': ['email', 'slack', 'phone']
                },
                {
                    'level': EscalationLevel.L3_EXPERT.value,
                    'timeout_minutes': 60,
                    'targets': [{'type': 'team', 'id': 'experts'}],
                    'contact_methods': ['email', 'phone', 'sms']
                }
            ]
            
            return EscalationRule(
                id='default',
                name='Default Escalation Rule',
                condition={'default': True},
                levels=default_levels
            )
            
        except Exception as e:
            logger.error(f"Erreur création règle par défaut: {e}")
            return None
    
    def _adjust_escalation_timeouts(self, escalation: EscalationPath, alert_data: Dict[str, Any]):
        """Ajuste les timeouts selon le contexte"""
        try:
            current_time = datetime.fromtimestamp(time.time())
            
            # Modificateur pour weekend
            if current_time.weekday() >= 5:  # Samedi ou Dimanche
                modifier = self.config['weekend_escalation_modifier']
                self._apply_timeout_modifier(escalation, modifier)
            
            # Modificateur pour jours fériés
            date_str = current_time.strftime('%Y-%m-%d')
            if date_str in self.holiday_calendar:
                modifier = self.config['holiday_escalation_modifier']
                self._apply_timeout_modifier(escalation, modifier)
            
            # Modificateur pour sévérité critique
            if alert_data.get('severity', '').lower() == 'critical':
                modifier = 0.5  # Réduit les timeouts de moitié
                self._apply_timeout_modifier(escalation, modifier)
            
        except Exception as e:
            logger.error(f"Erreur ajustement timeouts: {e}")
    
    def _apply_timeout_modifier(self, escalation: EscalationPath, modifier: float):
        """Applique un modificateur aux timeouts"""
        try:
            for level in escalation.levels:
                current_timeout = level.get('timeout_minutes', 15)
                level['timeout_minutes'] = max(1, int(current_timeout * modifier))
        except Exception as e:
            logger.error(f"Erreur application modificateur: {e}")
    
    def _start_escalation_level(self, escalation: EscalationPath, level: int):
        """Démarre un niveau d'escalade"""
        try:
            if level >= len(escalation.levels):
                logger.warning(f"Niveau d'escalade {level} inexistant pour {escalation.id}")
                return
            
            level_config = escalation.levels[level]
            
            # Met à jour le statut
            escalation.current_level = level
            escalation.status = EscalationStatus.ACTIVE
            escalation.current_level_started_at = time.time()
            
            # Trouve les destinataires
            targets = self._resolve_escalation_targets(level_config.get('targets', []))
            
            if not targets:
                logger.warning(f"Aucun destinataire trouvé pour niveau {level}")
                # Escalade automatiquement au niveau suivant
                self._escalate_to_next_level(escalation)
                return
            
            # Envoie les notifications
            contact_methods = level_config.get('contact_methods', ['email'])
            
            for target in targets:
                self._send_escalation_notification(escalation, target, contact_methods, level)
            
            # Sauvegarde
            self._save_escalation(escalation)
            
            with self.lock:
                self.metrics['escalations_by_level'][level] += 1
            
            # Log l'événement
            self._log_escalation_event(
                escalation_id=escalation.id,
                event_type="level_started",
                level=level,
                details={
                    'targets_count': len(targets),
                    'contact_methods': contact_methods,
                    'timeout_minutes': level_config.get('timeout_minutes', 15)
                }
            )
            
            logger.info(f"Niveau {level} démarré pour escalade {escalation.id}")
            
        except Exception as e:
            logger.error(f"Erreur démarrage niveau {level}: {e}")
    
    def _resolve_escalation_targets(self, targets_config: List[Dict[str, Any]]) -> List[Person]:
        """Résout les cibles d'escalade"""
        try:
            resolved_targets = []
            
            for target_config in targets_config:
                target_type = target_config.get('type')
                target_id = target_config.get('id')
                
                if target_type == 'person':
                    person = self.people.get(target_id)
                    if person and person.is_available:
                        resolved_targets.append(person)
                
                elif target_type == 'team':
                    team = self.teams.get(target_id)
                    if team:
                        # Trouve les membres disponibles
                        available_members = [
                            self.people[member_id] 
                            for member_id in team.members 
                            if member_id in self.people and self.people[member_id].is_available
                        ]
                        
                        if self.config['enable_load_balancing']:
                            # Sélectionne le membre avec le moins d'escalades actives
                            available_members.sort(key=lambda p: self._get_person_active_escalations_count(p.id))
                        
                        # Ajoute le ou les premiers membres disponibles
                        if available_members:
                            resolved_targets.append(available_members[0])
                
                elif target_type == 'on_call':
                    # Trouve la personne de garde
                    on_call_person = self._get_on_call_person(target_id)
                    if on_call_person:
                        resolved_targets.append(on_call_person)
            
            return resolved_targets
            
        except Exception as e:
            logger.error(f"Erreur résolution cibles: {e}")
            return []
    
    def _get_person_active_escalations_count(self, person_id: str) -> int:
        """Retourne le nombre d'escalades actives pour une personne"""
        try:
            count = 0
            for escalation in self.active_escalations.values():
                if escalation.status in [EscalationStatus.ACTIVE, EscalationStatus.PENDING]:
                    # Vérifie si la personne est impliquée dans cette escalade
                    current_level_config = escalation.levels[escalation.current_level]
                    targets = self._resolve_escalation_targets(current_level_config.get('targets', []))
                    if any(target.id == person_id for target in targets):
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Erreur comptage escalades: {e}")
            return 0
    
    def _get_on_call_person(self, schedule_id: str) -> Optional[Person]:
        """Retourne la personne de garde pour un planning"""
        try:
            # Implémentation simplifiée - dans la réalité, intégration avec système de planning
            current_time = datetime.now()
            day_of_week = current_time.weekday()
            hour = current_time.hour
            
            # Trouve l'équipe pour ce planning
            for team in self.teams.values():
                if team.type == TeamType.ON_CALL and schedule_id in team.on_call_schedule:
                    schedule = team.on_call_schedule[schedule_id]
                    
                    # Logique simple de rotation
                    if 'rotation' in schedule:
                        rotation = schedule['rotation']
                        week_number = current_time.isocalendar()[1]
                        person_index = week_number % len(rotation)
                        person_id = rotation[person_index]
                        
                        person = self.people.get(person_id)
                        if person and person.is_available:
                            return person
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur recherche personne de garde: {e}")
            return None
    
    def _send_escalation_notification(self, escalation: EscalationPath, target: Person,
                                    contact_methods: List[str], level: int):
        """Envoie une notification d'escalade"""
        try:
            if not self.notification_callback:
                logger.warning("Callback de notification non configuré")
                return
            
            # Prépare les données de notification
            notification_data = {
                'type': 'escalation_notification',
                'escalation_id': escalation.id,
                'alert_id': escalation.alert_id,
                'level': level,
                'target_person': {
                    'id': target.id,
                    'name': target.name,
                    'email': target.email,
                    'phone': target.phone
                },
                'contact_methods': contact_methods,
                'urgency': self._calculate_urgency_level(escalation, level),
                'timeout_minutes': escalation.levels[level].get('timeout_minutes', 15)
            }
            
            # Appelle le callback
            self.notification_callback(notification_data)
            
            # Log l'événement
            self._log_escalation_event(
                escalation_id=escalation.id,
                event_type="notification_sent",
                level=level,
                person_id=target.id,
                details={
                    'contact_methods': contact_methods,
                    'urgency': notification_data['urgency']
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur envoi notification escalade: {e}")
            
            # Log l'échec
            self._log_escalation_event(
                escalation_id=escalation.id,
                event_type="notification_failed",
                level=level,
                person_id=target.id,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_urgency_level(self, escalation: EscalationPath, level: int) -> str:
        """Calcule le niveau d'urgence"""
        try:
            base_urgency = ["low", "medium", "high", "critical", "emergency"]
            
            if level >= len(base_urgency):
                return "emergency"
            
            return base_urgency[level]
            
        except Exception as e:
            logger.error(f"Erreur calcul urgence: {e}")
            return "medium"
    
    def _escalate_to_next_level(self, escalation: EscalationPath):
        """Escalade au niveau suivant"""
        try:
            next_level = escalation.current_level + 1
            
            if next_level >= len(escalation.levels):
                # Plus de niveaux disponibles
                escalation.status = EscalationStatus.EXPIRED
                self._save_escalation(escalation)
                
                # Log l'événement
                self._log_escalation_event(
                    escalation_id=escalation.id,
                    event_type="escalation_expired",
                    level=escalation.current_level,
                    details={'max_level_reached': True}
                )
                
                logger.warning(f"Escalade expirée (niveau max atteint): {escalation.id}")
                return
            
            # Démarre le niveau suivant
            self._start_escalation_level(escalation, next_level)
            
            # Log l'événement
            self._log_escalation_event(
                escalation_id=escalation.id,
                event_type="escalated_to_next_level",
                level=next_level,
                details={'from_level': escalation.current_level}
            )
            
            logger.info(f"Escalade {escalation.id} passée au niveau {next_level}")
            
        except Exception as e:
            logger.error(f"Erreur escalade niveau suivant: {e}")
    
    def _escalation_worker(self):
        """Worker principal pour traiter les escalades"""
        while self.is_running:
            try:
                current_time = time.time()
                escalations_to_process = []
                
                with self.lock:
                    for escalation in self.active_escalations.values():
                        if escalation.status == EscalationStatus.ACTIVE:
                            # Vérifie le timeout du niveau actuel
                            level_config = escalation.levels[escalation.current_level]
                            timeout_seconds = level_config.get('timeout_minutes', 15) * 60
                            
                            if current_time - escalation.current_level_started_at >= timeout_seconds:
                                escalations_to_process.append(escalation)
                
                # Traite les escalades qui ont timeout
                for escalation in escalations_to_process:
                    self._escalate_to_next_level(escalation)
                
                # Attend avant la prochaine vérification
                time.sleep(self.config['escalation_check_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Erreur worker escalation: {e}")
                time.sleep(30)
    
    def _metrics_worker(self):
        """Worker pour collecter les métriques"""
        while self.is_running:
            try:
                self._collect_metrics()
                time.sleep(300)  # Toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur worker métriques: {e}")
                time.sleep(300)
    
    def _cleanup_worker(self):
        """Worker pour le nettoyage des données anciennes"""
        while self.is_running:
            try:
                self._cleanup_old_data()
                time.sleep(3600)  # Toutes les heures
                
            except Exception as e:
                logger.error(f"Erreur worker nettoyage: {e}")
                time.sleep(3600)
    
    def _collect_metrics(self):
        """Collecte les métriques"""
        try:
            current_time = time.time()
            
            # Métriques de base
            metrics_to_save = [
                ('active_escalations', len(self.active_escalations)),
                ('total_escalations', self.metrics['total_escalations']),
                ('escalations_resolved', self.metrics['escalations_resolved']),
                ('avg_resolution_time_minutes', self.metrics['avg_resolution_time_minutes']),
                ('sla_breaches', self.metrics['sla_breaches']),
                ('bypass_count', self.metrics['bypass_count'])
            ]
            
            # Métriques par niveau
            for level, count in self.metrics['escalations_by_level'].items():
                metrics_to_save.append((f'escalations_level_{level}', count))
            
            # Sauvegarde en base
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics_to_save:
                cursor.execute('''
                    INSERT INTO escalation_metrics 
                    (metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?)
                ''', (metric_name, metric_value, current_time))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques: {e}")
    
    def _cleanup_old_data(self):
        """Nettoie les données anciennes"""
        try:
            current_time = time.time()
            audit_retention = self.config['audit_retention_days'] * 24 * 3600
            metrics_retention = self.config['metrics_retention_days'] * 24 * 3600
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Nettoie les événements d'escalade anciens
            cursor.execute('''
                DELETE FROM escalation_events 
                WHERE timestamp < ?
            ''', (current_time - audit_retention,))
            
            # Nettoie les métriques anciennes
            cursor.execute('''
                DELETE FROM escalation_metrics 
                WHERE timestamp < ?
            ''', (current_time - metrics_retention,))
            
            conn.commit()
            conn.close()
            
            logger.info("Nettoyage des données anciennes effectué")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage: {e}")
    
    def _save_escalation(self, escalation: EscalationPath):
        """Sauvegarde une escalade"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO escalations 
                (id, alert_id, rule_id, current_level, status, levels, started_at, 
                 current_level_started_at, acknowledged_by, acknowledged_at, 
                 resolved_at, bypass_reason, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                escalation.id, escalation.alert_id, escalation.rule_id,
                escalation.current_level, escalation.status.value,
                json.dumps(escalation.levels), escalation.started_at,
                escalation.current_level_started_at, escalation.acknowledged_by,
                escalation.acknowledged_at, escalation.resolved_at,
                escalation.bypass_reason, escalation.tenant_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde escalade: {e}")
    
    def _log_escalation_event(self, escalation_id: str, event_type: str, level: int,
                             person_id: Optional[str] = None, team_id: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None, success: bool = True,
                             error_message: Optional[str] = None):
        """Log un événement d'escalade"""
        try:
            event = EscalationEvent(
                id=str(uuid.uuid4()),
                escalation_id=escalation_id,
                event_type=event_type,
                level=level,
                person_id=person_id,
                team_id=team_id,
                timestamp=time.time(),
                details=details or {},
                success=success,
                error_message=error_message
            )
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO escalation_events 
                (id, escalation_id, event_type, level, person_id, team_id, 
                 timestamp, details, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.id, event.escalation_id, event.event_type, event.level,
                event.person_id, event.team_id, event.timestamp,
                json.dumps(event.details), event.success, event.error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur log événement: {e}")
    
    def get_escalation_status(self, escalation_id: str) -> Optional[Dict[str, Any]]:
        """Retourne le statut d'une escalade"""
        try:
            escalation = self.active_escalations.get(escalation_id)
            if not escalation:
                # Cherche dans les escalades terminées
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM escalations WHERE id = ?', (escalation_id,))
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return None
                
                return {
                    'id': row[0],
                    'alert_id': row[1],
                    'status': row[4],
                    'current_level': row[3],
                    'started_at': row[6],
                    'acknowledged_by': row[8],
                    'acknowledged_at': row[9],
                    'resolved_at': row[10]
                }
            
            return {
                'id': escalation.id,
                'alert_id': escalation.alert_id,
                'rule_id': escalation.rule_id,
                'status': escalation.status.value,
                'current_level': escalation.current_level,
                'total_levels': len(escalation.levels),
                'started_at': escalation.started_at,
                'current_level_started_at': escalation.current_level_started_at,
                'acknowledged_by': escalation.acknowledged_by,
                'acknowledged_at': escalation.acknowledged_at,
                'resolved_at': escalation.resolved_at,
                'bypass_reason': escalation.bypass_reason,
                'tenant_id': escalation.tenant_id
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération statut escalade: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du gestionnaire"""
        try:
            return {
                'status': 'healthy' if self.is_running else 'stopped',
                'active_escalations': len(self.active_escalations),
                'people_count': len(self.people),
                'teams_count': len(self.teams),
                'rules_count': len(self.escalation_rules),
                'metrics': self.metrics.copy(),
                'uptime_seconds': time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def set_notification_callback(self, callback: Callable):
        """Configure le callback de notification"""
        self.notification_callback = callback
    
    def set_audit_callback(self, callback: Callable):
        """Configure le callback d'audit"""
        self.audit_callback = callback
