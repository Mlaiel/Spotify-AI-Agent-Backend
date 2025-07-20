# 🎵 Spotify AI Agent - Package Utilitaires Enterprise

## 🏆 Vue d'ensemble

Le **Package Utilitaires Enterprise** est une suite complète d'utilitaires prêts pour la production, conçus pour le backend Spotify AI Agent. Construit selon des standards de niveau enterprise, ce package fournit des outils essentiels pour la transformation de données, la sécurité, le monitoring des performances, et bien plus encore.

## 🎯 Fonctionnalités Clés

### 🔄 **Transformation & Validation de Données**
- Validation avancée de structures de données avec support de schémas
- Assainissement sécurisé des entrées avec protection XSS
- Fusion profonde avec stratégies configurables
- Sérialisation JSON pour types de données complexes
- Utilitaires de manipulation et filtrage de dictionnaires

### 📝 **Traitement de Chaînes & Analytique Textuelle**
- Slugification intelligente multilingue
- Conversion de casse (camel, snake, pascal)
- Extraction de motifs (emails, URLs, numéros de téléphone)
- Hachage sécurisé et génération aléatoire
- Masquage de données sensibles et statistiques textuelles

### ⏰ **Gestion DateTime**
- Analyse automatique multi-formats
- Gestion des fuseaux horaires avec zoneinfo
- Humanisation des dates ("il y a 2 heures")
- Calendrier métier avec jours fériés
- Validation de plages de dates et horaires d'ouverture

### 🔐 **Sécurité Cryptographique**
- Chiffrement AES-256 (modes GCM/CBC)
- Chiffrement asymétrique RSA-2048
- Hachage sécurisé de mots de passe (Argon2, bcrypt, scrypt)
- Signatures HMAC et numériques
- Génération de tokens cryptographiquement sécurisés

### 📁 **Gestion de Fichiers**
- Upload sécurisé de fichiers avec validation MIME
- Compression/décompression (gzip, bz2, zip, tar)
- Extraction de métadonnées audio/image avec EXIF
- Streaming de gros fichiers avec chunking
- Nettoyage automatique et gestion d'espace

### ⚡ **Monitoring des Performances**
- Collecte de métriques en temps réel
- Profilage détaillé avec intégration cProfile
- Cache haute performance avec TTL
- Détection et analyse des goulots d'étranglement
- Limitation de débit et optimisation mémoire

### 🌐 **Utilitaires Réseau**
- Client HTTP asynchrone de niveau enterprise
- Vérifications de santé et monitoring automatiques
- Validation avancée URL/domaine/IP
- Résolution DNS et validation de certificats SSL
- Monitoring de connectivité en temps réel

### ✅ **Framework de Validation**
- Validation d'email avec vérification de délivrabilité
- Validation de numéros de téléphone internationaux avec info opérateur
- Notation de force de mot de passe avec recommandations sécurité
- Validation de métadonnées spécifiques métier
- Validation de fichiers audio/image avec sécurité

### 🎨 **Export Multi-Format & Templates**
- Export vers JSON, XML, CSV, YAML, Markdown
- Système de templates Jinja2 dynamiques
- Formatage de devises, pourcentages et durées
- Génération de tableaux texte et création de rapports
- Embellissement de code et présentation de données

## 🚀 Démarrage Rapide

### Installation

```python
# Importer le package utils complet
from backend.app.api.utils import *
```

### Exemples d'Utilisation de Base

#### Transformation de Données
```python
# Valider et assainir les entrées utilisateur
validated_data = validate_data_structure(user_input, schema)
sanitized = sanitize_input(validated_data)

# Fusion profonde de configurations
merged_config = deep_merge(default_config, user_config)

# Aplatir des dictionnaires imbriqués
flat_data = flatten_dict(nested_data)
```

#### Opérations Cryptographiques
```python
# Chiffrement sécurisé
encryptor = SecureEncryption()
encrypted_data = encryptor.encrypt_json(sensitive_data)
decrypted_data = encryptor.decrypt_json(encrypted_data)

# Hachage de mot de passe
password_hash = hash_password(user_password, 'argon2')
is_valid = verify_password(user_password, password_hash, 'argon2')

# Génération de tokens
api_key = generate_api_key('spotify', 32)
session_id = generate_session_id()
```

#### Monitoring des Performances
```python
# Monitorer les performances de fonction
@monitor_performance()
@memoize(maxsize=256, ttl=3600)
async def process_audio_file(file_path: str):
    return await heavy_audio_processing(file_path)

# Benchmark de fonctions
@benchmark(iterations=1000)
def data_processing_function(data):
    return transform_data(data)
```

#### Opérations Réseau
```python
# Client HTTP enterprise
async with EnterpriseHttpClient() as client:
    # Vérification de santé
    health = await check_http_health('https://api.spotify.com/health')
    
    # Appels API avec retry automatique
    response = await client.get_json('https://api.spotify.com/v1/tracks')
    
    # POST avec données JSON
    result = await client.post_json(api_url, payload_data)
```

#### Gestion de Fichiers
```python
# Upload sécurisé de fichiers
upload_manager = FileUploadManager('/uploads')
file_info = upload_manager.save_uploaded_file(file_data, 'audio.mp3')

# Obtenir les métadonnées de fichier
metadata = get_file_metadata('/path/to/audio.mp3')
print(f"Durée: {metadata.get('duration')} secondes")

# Compresser des fichiers
compressed_path = compress_file('/path/to/large_file.txt', compression='gzip')
```

#### Validation
```python
# Validation d'email avec délivrabilité
email_result = validate_email('user@example.com', check_deliverability=True)

# Validation de numéro de téléphone
phone_result = validate_phone('+33123456789', region='FR')

# Validation de force de mot de passe
password_result = validate_user_password('MonMotDePasse123!')
print(f"Force: {password_result['strength']}")

# Validation de métadonnées audio
metadata_result = validate_audio_metadata({
    'title': 'Titre de Chanson',
    'artist': 'Nom Artiste',
    'duration': 240.5
})
```

## 📚 Documentation des Modules

### Modules Principaux

| Module | Description | Fonctions Clés |
|--------|-------------|-----------------|
| `data_transform` | Transformation et validation de données | `transform_data`, `validate_data_structure`, `deep_merge` |
| `string_utils` | Traitement de chaînes et analytique textuelle | `slugify`, `extract_emails`, `mask_sensitive_data` |
| `datetime_utils` | Gestion de date et heure | `format_datetime`, `humanize_datetime`, `convert_timezone` |
| `crypto_utils` | Opérations cryptographiques | `SecureEncryption`, `hash_password`, `generate_secure_token` |
| `file_utils` | Gestion et traitement de fichiers | `FileUploadManager`, `get_file_metadata`, `compress_file` |
| `performance_utils` | Monitoring et optimisation des performances | `monitor_performance`, `PerformanceMonitor`, `memoize` |
| `network_utils` | Communication réseau et validation | `EnterpriseHttpClient`, `check_http_health`, `validate_url` |
| `validators` | Framework de validation de données | `validate_email`, `validate_audio_metadata`, `DataValidator` |
| `formatters` | Export multi-format et templates | `format_json`, `TemplateFormatter`, `MultiFormatExporter` |

## 🛡️ Fonctionnalités de Sécurité

- **Protection XSS**: Assainissement intégré avec `bleach`
- **Génération Aléatoire Sécurisée**: Tokens et clés cryptographiquement sécurisés
- **Comparaison Temps Constant**: Protection contre les attaques temporelles
- **Validation d'Entrée**: Validation complète pour toutes les entrées utilisateur
- **Sécurité Fichier**: Validation de type MIME et gestion d'upload sécurisé

## 🚀 Fonctionnalités de Performance

- **Cache Intelligent**: Cache LRU avec support TTL
- **Limitation de Débit**: Limitation de débit distribuée pour protection API
- **Monitoring Mémoire**: Suivi d'utilisation mémoire en temps réel
- **Intégration Profilage**: Profilage de performance intégré
- **Support Asynchrone**: Support natif async/await partout

## 🔧 Configuration

### Variables d'Environnement

```bash
# Paramètres de cache
CACHE_DEFAULT_TTL=3600
CACHE_MAX_SIZE=1000

# Paramètres d'upload de fichiers
MAX_FILE_SIZE_MB=100
UPLOAD_DIRECTORY=/tmp/uploads

# Paramètres de sécurité
ENCRYPTION_KEY_LENGTH=32
TOKEN_EXPIRY_HOURS=24

# Paramètres de performance
RATE_LIMIT_REQUESTS_PER_MINUTE=100
MONITORING_ENABLED=true
```

### Configuration Personnalisée

```python
from backend.app.api.utils import NetworkConfig, PerformanceMonitor

# Configurer le client réseau
network_config = NetworkConfig(
    timeout=30.0,
    max_retries=3,
    verify_ssl=True
)

# Configurer le monitoring des performances
perf_monitor = PerformanceMonitor(max_history=2000)
```

## 🧪 Tests

Le package d'utilitaires inclut une couverture de tests complète :

```bash
# Exécuter tous les tests
pytest tests/

# Exécuter des tests de module spécifiques
pytest tests/test_crypto_utils.py
pytest tests/test_validators.py

# Exécuter avec couverture
pytest --cov=backend.app.api.utils tests/
```

## 📊 Monitoring & Métriques

### Métriques de Performance

```python
from backend.app.api.utils import performance_monitor

# Obtenir les statistiques de fonction
stats = performance_monitor.get_stats('nom_fonction')
print(f"Temps d'exécution moyen: {stats['avg']:.3f}s")
print(f"95e percentile: {stats['p95']:.3f}s")

# Monitoring système
system_monitor = SystemMonitor()
system_monitor.start_monitoring()
current_metrics = system_monitor.get_current_metrics()
```

### Vérifications de Santé

```python
# Monitorer la santé des endpoints
connectivity_monitor = ConnectivityMonitor()
connectivity_monitor.add_endpoint('https://api.spotify.com')
await connectivity_monitor.start_monitoring()

# Obtenir le statut global
status = connectivity_monitor.get_overall_status()
print(f"Santé globale: {status['overall_health']:.1f}%")
```

## 🤝 Contribution

### Directives de Développement

1. **Qualité du Code**: Suivre PEP 8 et utiliser les annotations de type
2. **Sécurité**: Toutes les entrées doivent être validées et assainies
3. **Performance**: Inclure monitoring et optimisation
4. **Documentation**: Docstrings complètes requises
5. **Tests**: Couverture de test minimum 90%

### Ajouter de Nouveaux Utilitaires

1. Créer un nouveau module dans la catégorie appropriée
2. Suivre les motifs et conventions existants
3. Ajouter des tests complets
4. Mettre à jour les exports `__init__.py`
5. Documenter dans le README

## 📝 Licence

Licence MIT - voir le fichier LICENSE pour les détails.

## 👨‍💻 Équipe Enterprise

**Développé par l'Équipe Enterprise Spotify AI Agent**

- **Développeur Principal & Architecte IA**: Conception système avancée et intégration ML
- **Développeur Backend Senior**: Utilitaires principaux et conception API
- **Ingénieur Machine Learning**: Utilitaires spécifiques ML et traitement de données
- **Ingénieur Base de Données & Données**: Transformation de données et utilitaires de stockage
- **Spécialiste Sécurité Backend**: Utilitaires cryptographiques et fonctionnalités sécurité
- **Architecte Microservices**: Utilitaires réseau et systèmes distribués

---

**Attribution spéciale: Fahed Mlaiel** - Architecture Enterprise et Excellence Technique

## 🔗 Liens

- [Documentation API](./docs/api.md)
- [Guide Performance](./docs/performance.md)
- [Guide Sécurité](./docs/security.md)
- [Guide Migration](./docs/migration.md)

---

*Construit avec ❤️ pour les applications de niveau enterprise*
