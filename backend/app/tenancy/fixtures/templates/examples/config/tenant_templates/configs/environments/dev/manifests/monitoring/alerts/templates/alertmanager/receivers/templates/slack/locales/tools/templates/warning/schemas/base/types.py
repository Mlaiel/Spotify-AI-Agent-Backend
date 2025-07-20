"""
Types avancés et modèles Pydantic - Spotify AI Agent
Définitions de types sophistiqués avec validation métier complète
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Generic, Callable, ForwardRef
from uuid import UUID, uuid4
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import re
import ipaddress
from pathlib import Path
import mimetypes

from pydantic import BaseModel, Field, validator, root_validator, computed_field, ConfigDict
from pydantic.types import (
    EmailStr, HttpUrl, IPvAnyAddress, StrictStr, PositiveInt, NonNegativeInt,
    StrictFloat, StrictBool, constr, conint, confloat, conlist, conset
)
from pydantic.networks import AnyUrl, PostgresDsn, RedisDsn, MongoDsn
from pydantic.color import Color


# Types personnalisés avancés
class TenantId(StrictStr):
    """Type personnalisé pour l'identifiant de tenant"""
    min_length = 3
    max_length = 50
    regex = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$')


class ResourceId(StrictStr):
    """Type pour les identifiants de ressources"""
    min_length = 1
    max_length = 255
    regex = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')


class AlertName(constr(min_length=1, max_length=255, strip_whitespace=True)):
    """Type pour les noms d'alerte avec validation"""
    pass


class MetricName(constr(min_length=1, max_length=100, regex=r'^[a-zA-Z][a-zA-Z0-9_]*$')):
    """Type pour les noms de métriques"""
    pass


class SqlQuery(constr(min_length=1, max_length=10000)):
    """Type pour les requêtes SQL avec validation basique"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError('string required')
        
        v = v.strip()
        if not v:
            raise ValueError('SQL query cannot be empty')
        
        # Validation basique contre les injections
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'xp_', 'sp_'
        ]
        
        upper_query = v.upper()
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {upper_query} ':
                raise ValueError(f'Dangerous SQL keyword detected: {keyword}')
        
        return v


class JsonString(str):
    """Type pour les chaînes JSON validées"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            try:
                import json
                json.loads(v)  # Valide que c'est du JSON
                return cls(v)
            except json.JSONDecodeError:
                raise ValueError('Invalid JSON string')
        raise TypeError('string required')


class Base64String(str):
    """Type pour les chaînes Base64 validées"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            try:
                import base64
                base64.b64decode(v, validate=True)
                return cls(v)
            except Exception:
                raise ValueError('Invalid Base64 string')
        raise TypeError('string required')


class HexColor(str):
    """Type pour les couleurs hexadécimales"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            pattern = re.compile(r'^#(?:[0-9a-fA-F]{3}){1,2}$')
            if pattern.match(v):
                return cls(v.upper())
            raise ValueError('Invalid hex color format')
        raise TypeError('string required')


class PhoneNumber(str):
    """Type pour les numéros de téléphone"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            # Pattern pour numéros internationaux
            pattern = re.compile(r'^\+?[1-9]\d{1,14}$')
            clean_number = re.sub(r'[^\d+]', '', v)
            if pattern.match(clean_number):
                return cls(clean_number)
            raise ValueError('Invalid phone number format')
        raise TypeError('string required')


class CronExpression(str):
    """Type pour les expressions cron"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            # Validation basique d'expression cron (5 ou 6 champs)
            parts = v.strip().split()
            if len(parts) in [5, 6]:
                return cls(v)
            raise ValueError('Invalid cron expression format')
        raise TypeError('string required')


# Modèles de base avancés
class Coordinates(BaseModel):
    """Coordonnées géographiques"""
    latitude: confloat(ge=-90, le=90) = Field(..., description="Latitude en degrés")
    longitude: confloat(ge=-180, le=180) = Field(..., description="Longitude en degrés")
    altitude: Optional[float] = Field(None, description="Altitude en mètres")
    
    @computed_field
    @property
    def decimal_degrees(self) -> str:
        """Format décimal des coordonnées"""
        return f"{self.latitude}, {self.longitude}"
    
    def distance_to(self, other: 'Coordinates') -> float:
        """Calcule la distance vers d'autres coordonnées (en km)"""
        import math
        
        # Formule de Haversine
        R = 6371  # Rayon de la Terre en km
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


class TimeRange(BaseModel):
    """Plage temporelle avec validation"""
    start_time: datetime = Field(..., description="Heure de début")
    end_time: datetime = Field(..., description="Heure de fin")
    timezone: str = Field("UTC", description="Fuseau horaire")
    
    @root_validator
    def validate_time_range(cls, values):
        """Valide que end_time > start_time"""
        start = values.get('start_time')
        end = values.get('end_time')
        
        if start and end and end <= start:
            raise ValueError('end_time must be after start_time')
        
        return values
    
    @computed_field
    @property
    def duration(self) -> timedelta:
        """Durée de la plage temporelle"""
        return self.end_time - self.start_time
    
    @computed_field
    @property
    def duration_seconds(self) -> float:
        """Durée en secondes"""
        return self.duration.total_seconds()
    
    def contains(self, timestamp: datetime) -> bool:
        """Vérifie si un timestamp est dans la plage"""
        return self.start_time <= timestamp <= self.end_time
    
    def overlaps_with(self, other: 'TimeRange') -> bool:
        """Vérifie si cette plage chevauche avec une autre"""
        return (
            self.start_time <= other.end_time and 
            self.end_time >= other.start_time
        )


class Threshold(BaseModel):
    """Seuil avec opérateurs de comparaison"""
    value: Union[int, float, Decimal] = Field(..., description="Valeur du seuil")
    operator: str = Field(..., regex=r'^(eq|ne|gt|gte|lt|lte|in|nin)$', description="Opérateur de comparaison")
    unit: Optional[str] = Field(None, max_length=20, description="Unité de mesure")
    
    def evaluate(self, test_value: Union[int, float, Decimal]) -> bool:
        """Évalue si une valeur respecte le seuil"""
        if self.operator == 'eq':
            return test_value == self.value
        elif self.operator == 'ne':
            return test_value != self.value
        elif self.operator == 'gt':
            return test_value > self.value
        elif self.operator == 'gte':
            return test_value >= self.value
        elif self.operator == 'lt':
            return test_value < self.value
        elif self.operator == 'lte':
            return test_value <= self.value
        else:
            return False
    
    @computed_field
    @property
    def description(self) -> str:
        """Description textuelle du seuil"""
        op_text = {
            'eq': 'égal à', 'ne': 'différent de', 'gt': 'supérieur à',
            'gte': 'supérieur ou égal à', 'lt': 'inférieur à', 'lte': 'inférieur ou égal à'
        }
        unit_text = f" {self.unit}" if self.unit else ""
        return f"{op_text.get(self.operator, self.operator)} {self.value}{unit_text}"


class Contact(BaseModel):
    """Contact avec informations complètes"""
    name: constr(min_length=1, max_length=255) = Field(..., description="Nom du contact")
    email: Optional[EmailStr] = Field(None, description="Adresse email")
    phone: Optional[PhoneNumber] = Field(None, description="Numéro de téléphone")
    slack_user: Optional[constr(regex=r'^@[a-zA-Z0-9._-]+$')] = Field(None, description="Utilisateur Slack")
    teams_user: Optional[str] = Field(None, description="Utilisateur Teams")
    role: Optional[str] = Field(None, max_length=100, description="Rôle du contact")
    department: Optional[str] = Field(None, max_length=100, description="Département")
    timezone: str = Field("UTC", description="Fuseau horaire du contact")
    on_call: bool = Field(False, description="Indique si le contact est de garde")
    availability: Optional['ContactAvailability'] = Field(None, description="Disponibilité du contact")
    
    @validator('name')
    def validate_name(cls, v):
        """Valide le nom du contact"""
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @computed_field
    @property
    def primary_contact_method(self) -> Optional[str]:
        """Méthode de contact primaire"""
        if self.email:
            return f"email:{self.email}"
        elif self.phone:
            return f"phone:{self.phone}"
        elif self.slack_user:
            return f"slack:{self.slack_user}"
        return None
    
    def is_available_at(self, timestamp: datetime) -> bool:
        """Vérifie si le contact est disponible à un moment donné"""
        if not self.availability:
            return True
        return self.availability.is_available_at(timestamp)


class ContactAvailability(BaseModel):
    """Disponibilité d'un contact"""
    working_hours: TimeRange = Field(..., description="Heures de travail")
    working_days: Set[int] = Field(
        default={1, 2, 3, 4, 5},  # Lundi à vendredi
        description="Jours de travail (1=Lundi, 7=Dimanche)"
    )
    emergency_contact: bool = Field(False, description="Contact d'urgence (toujours disponible)")
    out_of_office: Optional[TimeRange] = Field(None, description="Période d'absence")
    
    @validator('working_days')
    def validate_working_days(cls, v):
        """Valide les jours de travail"""
        if not isinstance(v, set):
            v = set(v) if hasattr(v, '__iter__') else set()
        return {day for day in v if isinstance(day, int) and 1 <= day <= 7}
    
    def is_available_at(self, timestamp: datetime) -> bool:
        """Vérifie la disponibilité à un moment donné"""
        if self.emergency_contact:
            return True
        
        # Vérifier si en congé
        if self.out_of_office and self.out_of_office.contains(timestamp):
            return False
        
        # Vérifier le jour de la semaine
        weekday = timestamp.isoweekday()  # 1=Lundi, 7=Dimanche
        if weekday not in self.working_days:
            return False
        
        # Vérifier les heures de travail
        time_only = timestamp.time()
        working_start = self.working_hours.start_time.time()
        working_end = self.working_hours.end_time.time()
        
        return working_start <= time_only <= working_end


class FileInfo(BaseModel):
    """Informations sur un fichier"""
    name: constr(min_length=1, max_length=255) = Field(..., description="Nom du fichier")
    path: Optional[str] = Field(None, description="Chemin du fichier")
    size_bytes: NonNegativeInt = Field(0, description="Taille en octets")
    mime_type: Optional[str] = Field(None, description="Type MIME")
    checksum: Optional[str] = Field(None, description="Checksum du fichier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: Optional[datetime] = Field(None, description="Date de modification")
    
    @validator('name')
    def validate_filename(cls, v):
        """Valide le nom de fichier"""
        # Caractères interdits dans les noms de fichier
        forbidden_chars = '<>:"/\\|?*'
        if any(char in v for char in forbidden_chars):
            raise ValueError(f'Filename contains forbidden characters: {forbidden_chars}')
        return v
    
    @validator('mime_type', pre=True, always=True)
    def auto_detect_mime_type(cls, v, values):
        """Détection automatique du type MIME"""
        if v:
            return v
        
        filename = values.get('name')
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            return mime_type
        
        return None
    
    @computed_field
    @property
    def size_human(self) -> str:
        """Taille formatée pour l'affichage"""
        size = self.size_bytes
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    @computed_field
    @property
    def extension(self) -> str:
        """Extension du fichier"""
        return Path(self.name).suffix.lower() if self.name else ""
    
    @computed_field
    @property
    def is_image(self) -> bool:
        """Indique si c'est un fichier image"""
        return self.mime_type is not None and self.mime_type.startswith('image/')
    
    @computed_field
    @property
    def is_text(self) -> bool:
        """Indique si c'est un fichier texte"""
        return self.mime_type is not None and (
            self.mime_type.startswith('text/') or 
            self.mime_type in ['application/json', 'application/xml', 'application/yaml']
        )


class NetworkInfo(BaseModel):
    """Informations réseau"""
    ip_address: IPvAnyAddress = Field(..., description="Adresse IP")
    port: conint(ge=1, le=65535) = Field(..., description="Port")
    protocol: str = Field("tcp", regex=r'^(tcp|udp|http|https)$', description="Protocole")
    hostname: Optional[str] = Field(None, description="Nom d'hôte")
    
    @computed_field
    @property
    def endpoint(self) -> str:
        """Point de terminaison complet"""
        if self.protocol in ['http', 'https']:
            return f"{self.protocol}://{self.ip_address}:{self.port}"
        return f"{self.ip_address}:{self.port}"
    
    @computed_field
    @property
    def is_private(self) -> bool:
        """Indique si l'IP est privée"""
        try:
            ip = ipaddress.ip_address(str(self.ip_address))
            return ip.is_private
        except ValueError:
            return False
    
    @computed_field
    @property
    def is_localhost(self) -> bool:
        """Indique si c'est localhost"""
        try:
            ip = ipaddress.ip_address(str(self.ip_address))
            return ip.is_loopback
        except ValueError:
            return False


class ResourceQuota(BaseModel):
    """Quota de ressources avec limites flexibles"""
    cpu_cores: Optional[confloat(gt=0)] = Field(None, description="Nombre de cœurs CPU")
    memory_mb: Optional[conint(gt=0)] = Field(None, description="Mémoire en MB")
    disk_gb: Optional[conint(gt=0)] = Field(None, description="Disque en GB")
    network_mbps: Optional[conint(gt=0)] = Field(None, description="Bande passante en Mbps")
    requests_per_minute: Optional[conint(gt=0)] = Field(None, description="Requêtes par minute")
    concurrent_connections: Optional[conint(gt=0)] = Field(None, description="Connexions simultanées")
    
    def check_limit(self, resource: str, current_usage: Union[int, float]) -> bool:
        """Vérifie si l'utilisation respecte les limites"""
        limit = getattr(self, resource, None)
        if limit is None:
            return True  # Pas de limite définie
        return current_usage <= limit
    
    @computed_field
    @property
    def total_memory_gb(self) -> Optional[float]:
        """Mémoire totale en GB"""
        return self.memory_mb / 1024 if self.memory_mb else None


class MonetaryAmount(BaseModel):
    """Montant monétaire avec devise"""
    amount: Decimal = Field(..., description="Montant", decimal_places=2)
    currency: constr(min_length=3, max_length=3, to_upper=True) = Field("USD", description="Code devise ISO")
    
    @validator('amount')
    def validate_amount(cls, v):
        """Valide le montant monétaire"""
        if v < 0:
            raise ValueError('Amount cannot be negative')
        # Arrondir à 2 décimales
        return v.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    @computed_field
    @property
    def formatted(self) -> str:
        """Montant formaté avec devise"""
        return f"{self.amount} {self.currency}"
    
    def convert_to(self, target_currency: str, exchange_rate: Decimal) -> 'MonetaryAmount':
        """Convertit vers une autre devise"""
        converted_amount = self.amount * exchange_rate
        return MonetaryAmount(amount=converted_amount, currency=target_currency.upper())


# Forward references pour les relations circulaires
ContactAvailability.model_rebuild()
Contact.model_rebuild()
TimeRange.model_rebuild()


__all__ = [
    # Types personnalisés
    'TenantId', 'ResourceId', 'AlertName', 'MetricName', 'SqlQuery', 'JsonString',
    'Base64String', 'HexColor', 'PhoneNumber', 'CronExpression',
    
    # Modèles avancés
    'Coordinates', 'TimeRange', 'Threshold', 'Contact', 'ContactAvailability',
    'FileInfo', 'NetworkInfo', 'ResourceQuota', 'MonetaryAmount'
]
