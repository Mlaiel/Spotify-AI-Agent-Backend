# 🚀 Spotify AI Agent - Module API Core

## Aperçu

Le **Module API Core** est la couche fondamentale du backend Spotify AI Agent, fournissant des composants d'infrastructure de niveau entreprise pour le développement d'API, la gestion des requêtes/réponses, la configuration et le contexte applicatif.

## 🏗️ Architecture

```
app/api/core/
├── __init__.py           # Exports et initialisation du module
├── config.py             # Gestion et validation de la configuration
├── context.py            # Contexte de requête et injection de dépendances
├── exceptions.py         # Hiérarchie d'exceptions personnalisées
├── factory.py            # Patterns factory pour la création de composants
├── response.py           # Standardisation et formatage des réponses
└── README.fr.md          # Cette documentation
```

## 🔧 Composants Clés

### Gestion de Configuration (`config.py`)
- **APIConfig** : Configuration API centrale avec validation
- **DatabaseConfig** : Paramètres de connexion base de données
- **SecurityConfig** : Politiques de sécurité et authentification
- **MonitoringConfig** : Configuration observabilité et métriques
- Configurations spécifiques par environnement (dev, staging, prod)

### Contexte de Requête (`context.py`)
- **RequestContext** : Gestion de contexte thread-safe
- **DependencyInjector** : Injection de dépendances de services
- **ContextualLogger** : Logging contextuel intelligent
- Suivi des requêtes et IDs de corrélation

### Gestion d'Exceptions (`exceptions.py`)
- **APIException** : Exception de base pour toutes les erreurs API
- **ValidationError** : Échecs de validation d'entrée
- **AuthenticationError** : Échecs d'authentification
- **BusinessLogicError** : Erreurs spécifiques au domaine métier
- Réponses d'erreur structurées avec support i18n

### Patterns Factory (`factory.py`)
- **ComponentFactory** : Création générique de composants
- **ServiceFactory** : Gestion d'instances de services
- **MiddlewareFactory** : Construction de chaînes de middleware
- **DatabaseFactory** : Pooling de connexions base de données

### Standardisation Réponses (`response.py`)
- **APIResponse** : Format de réponse standardisé
- **PaginatedResponse** : Réponses de données paginées
- **ErrorResponse** : Formatage de réponses d'erreur
- **SuccessResponse** : Helpers de réponses de succès
- Headers de compression et cache des réponses

## 🚀 Démarrage Rapide

### Utilisation de Base

```python
from app.api.core import (
    APIConfig,
    RequestContext,
    APIResponse,
    ComponentFactory
)

# Initialiser la configuration
config = APIConfig.from_environment()

# Créer un contexte de requête
with RequestContext() as ctx:
    ctx.set_user_id("user_123")
    ctx.set_correlation_id("req_456")
    
    # Utiliser la factory pour créer des composants
    service = ComponentFactory.create_service("user_service")
    
    # Créer une réponse standardisée
    response = APIResponse.success(
        data={"message": "Bonjour le Monde"},
        meta={"version": "1.0.0"}
    )
```

### Gestion de Configuration

```python
from app.api.core.config import APIConfig, get_config

# Obtenir la configuration actuelle
config = get_config()

# Accéder aux paramètres spécifiques
database_url = config.database.url
redis_url = config.cache.redis_url
log_level = config.logging.level

# Valider la configuration
config.validate()
```

### Gestion d'Exceptions

```python
from app.api.core.exceptions import ValidationError, APIException

@app.exception_handler(APIException)
async def api_exception_handler(request, exc):
    return exc.to_response()

# Lever une erreur de validation
if not user_id:
    raise ValidationError(
        message="L'ID utilisateur est requis",
        field="user_id",
        code="MISSING_USER_ID"
    )
```

## 🔒 Fonctionnalités de Sécurité

- **Validation d'Entrée** : Validation complète des requêtes
- **Authentification** : Authentification JWT et clé API
- **Autorisation** : Contrôle d'accès basé sur les rôles (RBAC)
- **Limitation de Débit** : Limites par utilisateur et par endpoint
- **CORS** : Politiques de partage de ressources cross-origin
- **Headers de Sécurité** : Headers de sécurité conformes OWASP

## 📊 Monitoring & Observabilité

- **Métriques** : Métriques compatibles Prometheus
- **Traçage** : Traçage distribué OpenTelemetry
- **Logging** : Logging JSON structuré avec IDs de corrélation
- **Vérifications de Santé** : Monitoring santé application et dépendances
- **Performance** : Timing et profilage requête/réponse

## 🧪 Tests

```bash
# Exécuter les tests du module core
pytest tests_backend/app/api/core/ -v

# Exécuter avec couverture
pytest tests_backend/app/api/core/ --cov=app.api.core --cov-report=html

# Exécuter les tests de performance
pytest tests_backend/app/api/core/ -m performance
```

## 📈 Performance

- **Temps de Réponse** : < 10ms pour l'accès configuration
- **Utilisation Mémoire** : Optimisé pour faible empreinte mémoire
- **Débit** : Support 10 000+ requêtes par seconde
- **Cache** : Cache intelligent configuration et réponses

## 🔧 Configuration

### Variables d'Environnement

```env
# Configuration API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_VERSION=1.0.0

# Base de données
DATABASE_URL=postgresql://user:pass@localhost:5432/spotify_ai
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Sécurité
JWT_SECRET_KEY=votre-clé-secrète
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
LOG_LEVEL=INFO
```

## 🌐 Internationalisation

Le module core supporte plusieurs langues :
- **Français** (par défaut)
- **Anglais** (English)
- **Allemand** (Deutsch)
- **Espagnol** (español)

## 🤝 Contribution

1. Suivre le style et les patterns de code établis
2. Ajouter des tests complets pour les nouvelles fonctionnalités
3. Mettre à jour la documentation pour les changements d'API
4. S'assurer que toutes les vérifications de sécurité passent
5. Maintenir la compatibilité ascendante

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](../../../LICENSE) pour les détails.

## 👥 Auteurs

- **Fahed Mlaiel** - Développeur Principal & Architecte Entreprise
- **Équipe Spotify AI Agent** - Équipe de Développement Core

---

**Infrastructure API Niveau Entreprise** | Construit avec ❤️ pour la scalabilité et la performance
