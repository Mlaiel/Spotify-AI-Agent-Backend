# Module de Validation - Spotify AI Agent

## Vue d'ensemble

Le module de validation fournit des capacités de validation et de sanitisation complètes pour le système de schémas du Spotify AI Agent. Il inclut des validateurs personnalisés, des règles de sécurité, et des optimisations de performance pour assurer l'intégrité et la sécurité des données à travers tous les composants du système.

## Fonctionnalités Principales

### 🔒 Règles de Validation Avancées
- **Validation Tenant ID**: Correspondance de motifs, vérification des mots réservés, limites de longueur
- **Validation Messages d'Alerte**: Sanitisation du contenu, détection de code malveillant
- **Validation ID de Corrélation**: Vérification de format et cohérence
- **Validation Métadonnées**: Limites de taille, vérification de structure JSON
- **Validation Tags**: Règles clé-valeur, conventions de nommage
- **Validation Destinataires**: Validation de format spécifique par canal (email, Slack, SMS)

### 🛡️ Sanitisation de Sécurité
- **Sanitisation HTML**: Suppression des tags et attributs dangereux
- **Protection Injection SQL**: Détection et prévention de mots-clés
- **Prévention XSS**: Suppression de scripts et gestionnaires d'événements
- **Normalisation d'Entrée**: Gestion des espaces blancs et caractères de contrôle

### ⚙️ Validateurs de Configuration
- **Validation Templates JSON**: Vérification syntaxe Jinja2
- **Validation Expressions Cron**: Vérification format planification
- **Validation URL**: Restrictions de protocole et domaine
- **Validation Plages Temporelles**: Vérifications de cohérence logique

### 🎵 Validateurs Domaine Spotify
- **IDs Spotify**: Validation des identifiants tracks, artistes, albums, playlists
- **URIs Spotify**: Validation des URI complets Spotify
- **Caractéristiques Audio**: Validation des features audio avec plages valides
- **Genres Musicaux**: Validation des noms de genres et filtrage de contenu

### 🤖 Validateurs ML/Audio
- **Formats Audio**: Validation des spécifications (sample rate, channels, bit depth)
- **Configuration Modèles ML**: Validation spécifique par type de modèle
- **Métriques Performance**: Validation des métriques de latence et débit
- **Utilisation Ressources**: Validation des métriques système

### 🔐 Règles de Sécurité
- **Force Mots de Passe**: Critères de complexité, patterns interdits
- **Clés API**: Validation de format et longueur
- **Tokens JWT**: Vérification de structure basique
- **Clés de Chiffrement**: Validation Fernet et AES
- **Adresses IP**: Validation avec restrictions domaines privés
- **En-têtes Sécurité**: Validation des en-têtes de sécurité recommandés

### ⚡ Optimisations Performance
- **Cache de Validation**: Cache LRU avec TTL pour validations fréquentes
- **Traitement par Batch**: Validation parallèle pour gros volumes
- **Validation Adaptative**: Ajustement automatique selon la charge
- **Profiling**: Métriques de performance détaillées

## Architecture

```
validation/
├── __init__.py              # Classes principales et décorateurs
├── custom_validators.py     # Validateurs spécifiques au domaine
├── security_rules.py       # Règles de validation sécurisée
├── configuration.py        # Schémas de configuration système
├── performance.py          # Validateurs optimisés performance
├── README.md              # Documentation principale
├── README.fr.md          # Documentation française
└── README.de.md          # Documentation allemande
```

## Exemples d'Utilisation

### Validation de Base

```python
from pydantic import BaseModel
from schemas.validation import (
    validate_tenant_id_field, 
    validate_alert_message_field,
    validate_metadata_field
)

class AlertSchema(BaseModel):
    tenant_id: str
    message: str
    metadata: Dict[str, Any]
    
    # Application des validateurs
    _validate_tenant_id = validate_tenant_id_field()
    _validate_message = validate_alert_message_field()
    _validate_metadata = validate_metadata_field()
```

### Règles de Validation Personnalisées

```python
from schemas.validation import ValidationRules

# Validation tenant ID
try:
    clean_tenant_id = ValidationRules.validate_tenant_id("mon-tenant-2024")
except ValueError as e:
    print(f"Erreur de validation: {e}")

# Validation message d'alerte
clean_message = ValidationRules.validate_alert_message(
    "Alerte système: Utilisation CPU élevée détectée"
)

# Validation destinataires par canal
email_recipients = ValidationRules.validate_recipients_list(
    ["admin@example.com", "ops@company.com"],
    NotificationChannel.EMAIL
)
```

### Sanitisation de Données

```python
from schemas.validation import DataSanitizer

# Sanitisation contenu HTML
clean_html = DataSanitizer.sanitize_html(
    "<p>Contenu sûr</p><script>alert('xss')</script>"
)
# Résultat: "<p>Contenu sûr</p>"

# Normalisation espaces blancs
clean_text = DataSanitizer.normalize_whitespace(
    "  Multiples   espaces\t\net\r\nretours\n\nà\n\nla\n\nligne  "
)
# Résultat: "Multiples espaces et retours à la ligne"

# Troncature texte long
short_text = DataSanitizer.truncate_text(
    "Très long contenu de texte ici...", 
    max_length=20, 
    suffix="..."
)
```

### Validateurs Spotify

```python
from schemas.validation.custom_validators import SpotifyDomainValidators

# Validation ID track Spotify
track_id = SpotifyDomainValidators.validate_spotify_id(
    "4iV5W9uYEdYUVa79Axb7Rh", "track"
)

# Validation URI Spotify
uri = SpotifyDomainValidators.validate_spotify_uri(
    "spotify:track:4iV5W9uYEdYUVa79Axb7Rh"
)

# Validation caractéristiques audio
features = SpotifyDomainValidators.validate_audio_features({
    "danceability": 0.8,
    "energy": 0.7,
    "valence": 0.6,
    "tempo": 120.0
})
```

### Validation de Sécurité

```python
from schemas.validation.security_rules import SecurityValidationRules

# Validation force mot de passe
password = SecurityValidationRules.validate_password_strength(
    "MonMotDePasseSecurise123!"
)

# Validation clé API
api_key = SecurityValidationRules.validate_api_key(
    "sk_live_51234567890abcdef1234567890abcdef"
)

# Validation adresse IP
ip = SecurityValidationRules.validate_ip_address(
    "192.168.1.100", allow_private=True
)

# Génération token sécurisé
secure_token = SecurityValidationRules.generate_secure_token(32)
```

### Configuration Système

```python
from schemas.validation.configuration import EnvironmentConfig, DatabaseConfig

# Configuration base de données
db_config = DatabaseConfig(
    type="postgresql",
    host="localhost",
    port=5432,
    database="spotify_ai_agent",
    username="app_user",
    password="secure_password_123",
    min_connections=5,
    max_connections=20
)

# Configuration environnement complète
env_config = EnvironmentConfig(
    environment="production",
    version="1.0.0",
    tenant_id="main",
    database=db_config,
    # ... autres configurations
)
```

### Optimisations Performance

```python
from schemas.validation.performance import (
    OptimizedValidationRules,
    BatchValidationProcessor,
    adaptive_validator
)

# Validation optimisée tenant ID
tenant_id = OptimizedValidationRules.validate_tenant_id_fast("mon-tenant")

# Validation par batch
processor = BatchValidationProcessor()
results = await processor.validate_batch(
    tenant_ids_list,
    OptimizedValidationRules.validate_tenant_id_fast
)

# Validation adaptative
result = adaptive_validator.validate_adaptive(
    data, ValidationRules.validate_tenant_id
)
```

## Patterns de Validation

Le module inclut des patterns prédéfinis pour les scénarios de validation courants:

- **TENANT_ID**: `^[a-z0-9_-]+$`
- **CORRELATION_ID**: `^[a-zA-Z0-9_-]+$`
- **EMAIL**: Pattern conforme RFC 5322
- **PHONE**: Support format international
- **URL**: HTTP/HTTPS avec restrictions domaines
- **VERSION**: Versioning sémantique
- **HEX_COLOR**: Couleurs hexadécimales 6 chiffres

## Considérations Performance

### Patterns Compilés
Tous les patterns regex sont pré-compilés pour une performance optimale dans les scénarios haute fréquence.

### Cache
Les résultats de validation sont mis en cache lorsque approprié pour éviter le retraitement redondant.

### Validation par Batch
Support pour valider plusieurs éléments en une seule opération:

```python
# Validation multiple destinataires en une fois
recipients = ["user1@example.com", "user2@example.com", "user3@example.com"]
validated = ValidationRules.validate_recipients_list(recipients, NotificationChannel.EMAIL)
```

## Gestion d'Erreurs

### Erreurs de Validation
Toutes les erreurs de validation fournissent des messages détaillés pour le débogage:

```python
try:
    ValidationRules.validate_tenant_id("TENANT-INVALIDE!")
except ValueError as e:
    # Erreur: "Tenant ID must contain only letters, numbers, hyphens and underscores"
    handle_validation_error(e)
```

### Violations de Sécurité
Les échecs de validation liés à la sécurité sont loggés et peuvent déclencher des alertes:

```python
# Ceci lèvera une erreur et loggera un événement sécuritaire
ValidationRules.validate_alert_message("<script>code_malveillant()</script>")
```

## Intégration avec Pydantic

### Validateurs Personnalisés
Utilisez les décorateurs fournis pour une intégration transparente avec Pydantic:

```python
from pydantic import BaseModel
from schemas.validation import validate_tenant_id_field, validate_time_range_fields

class TenantAlert(BaseModel):
    tenant_id: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    
    _validate_tenant = validate_tenant_id_field()
    _validate_time_range = validate_time_range_fields('start_time', 'end_time')
```

### Validateurs Root
Pour la logique de validation inter-champs:

```python
from pydantic import root_validator
from schemas.validation import ValidationRules

class ComplexSchema(BaseModel):
    field1: str
    field2: str
    
    @root_validator
    def validate_consistency(cls, values):
        # Logique de validation inter-champs personnalisée
        if values.get('field1') and values.get('field2'):
            # Assurer la cohérence entre les champs
            pass
        return values
```

## Configuration

### Variables d'Environnement
- `VALIDATION_STRICT_MODE`: Activer/désactiver la validation stricte
- `MAX_VALIDATION_CACHE_SIZE`: Limite de taille du cache de validation
- `VALIDATION_LOG_LEVEL`: Niveau de logging pour les événements de validation

### Personnalisation
Étendre les règles de validation pour des cas d'usage spécifiques:

```python
from schemas.validation import ValidationRules

class CustomValidationRules(ValidationRules):
    @classmethod
    def validate_custom_field(cls, value: str) -> str:
        # Logique de validation personnalisée
        return super().validate_tenant_id(value)
```

## Bonnes Pratiques Sécuritaires

1. **Sanitisation d'Entrée**: Toujours sanitiser l'entrée utilisateur avant traitement
2. **Validation Whitelist**: Utiliser la validation positive (patterns autorisés) plutôt que les blacklists
3. **Limites de Longueur**: Appliquer des limites raisonnables sur tous les champs chaîne
4. **Filtrage Contenu**: Vérifier les patterns malveillants dans le contenu utilisateur
5. **Restrictions URL**: Valider et restreindre les URLs webhook pour prévenir SSRF
6. **Sécurité Template**: Valider la syntaxe des templates pour prévenir les attaques par injection

## Monitoring et Métriques

Le module de validation inclut des capacités de monitoring intégrées:

```python
from schemas.validation.performance import get_performance_stats

# Obtenir les statistiques de performance
stats = get_performance_stats()
print(f"Cache hit rate: {stats['cache_stats']}")
print(f"Validation timing: {stats['profiler_stats']}")
```

## Contribution

Lors de l'ajout de nouvelles règles de validation:

1. Suivre la structure de patterns existante
2. Inclure des messages d'erreur complets
3. Ajouter des benchmarks de performance pour les validateurs complexes
4. Documenter les implications sécuritaires
5. Fournir des exemples d'utilisation

---

Ce module a été développé dans le cadre du projet Spotify AI Agent, en se concentrant sur la validation de données de niveau entreprise et la sécurité.
