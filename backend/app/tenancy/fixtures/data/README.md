# Multi-Tenant Test Data System
## Générateur de Données de Test et Fixtures

Ce système fournit une génération complète de données de test pour les scénarios multi-tenant, incluant des datasets réalistes pour les tests de performance, de conformité et de charge.

## 📁 Structure des Fichiers

```
data/
├── multi_tenant_data_generator.py      # Générateur principal de données
├── generate_test_data.py               # Scripts de génération de test
├── sample_free_tier.json               # Exemple de données tier gratuit
├── sample_enterprise_tier.json         # Exemple de données tier enterprise
└── README.md                           # Cette documentation
```

## 🎯 Types de Données Générées

### Entités Principales

#### 1. **Artistes** (`artists`)
```json
{
  "artist_id": "tenant_001_artist_000001",
  "tenant_id": "tenant_001", 
  "name": "Alex Rivers",
  "biography": "An indie pop artist from California...",
  "genres": ["Pop", "Indie", "Alternative"],
  "country": "US",
  "monthly_listeners": 15420,
  "is_verified": false,
  "social_media": {
    "spotify_url": "https://open.spotify.com/artist/...",
    "website": "https://alexriversmusic.com"
  }
}
```

#### 2. **Albums** (`albums`)
```json
{
  "album_id": "tenant_001_album_000001",
  "tenant_id": "tenant_001",
  "artist_id": "tenant_001_artist_000001",
  "title": "Dreamscapes",
  "album_type": "album",
  "release_date": "2023-06-15",
  "total_tracks": 12,
  "duration_ms": 2940000,
  "popularity": 65,
  "markets": ["US", "CA", "GB", "AU", "NZ"]
}
```

#### 3. **Tracks** (`tracks`)
```json
{
  "track_id": "tenant_001_track_000001",
  "tenant_id": "tenant_001",
  "name": "Midnight Drive",
  "artist_name": "Alex Rivers",
  "duration_ms": 245000,
  "popularity": 72,
  "audio_features": {
    "danceability": 0.65,
    "energy": 0.58,
    "valence": 0.71,
    "tempo": 120.5
  },
  "mood": "Dreamy"
}
```

#### 4. **Utilisateurs** (`users`)
```json
{
  "user_id": "tenant_001_user_000001",
  "tenant_id": "tenant_001",
  "display_name": "Music Lover",
  "email": "user1@example.com",
  "country": "US",
  "subscription_type": "premium",
  "preferences": {
    "preferred_genres": ["Pop", "Indie", "Rock"],
    "discovery_mode": "balanced"
  },
  "activity_metrics": {
    "total_listening_time_ms": 1800000,
    "tracks_played": 95,
    "last_active": "2024-12-18T20:15:00Z"
  }
}
```

### Données Comportementales

#### 5. **Historique d'Écoute** (`listening_history`)
```json
{
  "event_id": "tenant_001_event_000000001",
  "tenant_id": "tenant_001",
  "user_id": "tenant_001_user_000001",
  "track_id": "tenant_001_track_000001",
  "played_at": "2024-12-18T20:15:30Z",
  "duration_ms": 245000,
  "completion_percentage": 100.0,
  "context": {
    "type": "playlist",
    "uri": "spotify:playlist:..."
  },
  "device": {
    "type": "smartphone",
    "volume_percent": 75
  }
}
```

#### 6. **Événements Analytics** (`analytics_events`)
```json
{
  "event_id": "tenant_001_analytics_000000001",
  "tenant_id": "tenant_001",
  "user_id": "tenant_001_user_000001",
  "event_type": "app_open",
  "timestamp": "2024-12-19T09:30:00Z",
  "session_id": "ses_12345678-90ab-cdef-1234-567890abcdef",
  "properties": {
    "user_subscription": "premium",
    "user_country": "US"
  }
}
```

### Données ML/IA

#### 7. **Features ML** (`ml_features`)
```json
{
  "user_features": [
    {
      "user_id": "tenant_001_user_000001",
      "activity_score": 0.92,
      "diversity_score": 0.85,
      "preferred_genres_vector": [0, 0, 0, 1, 0, 1, 0, ...],
      "time_of_day_preference": {
        "morning": 0.7,
        "afternoon": 0.9,
        "evening": 0.8,
        "night": 0.4
      }
    }
  ],
  "track_features": [...],
  "interaction_features": [...]
}
```

## 🏗️ Générateur de Données

### Utilisation du `MultiTenantDataGenerator`

```python
from multi_tenant_data_generator import MultiTenantDataGenerator, TenantTier

# Initialisation
generator = MultiTenantDataGenerator(
    output_path="/app/tenancy/fixtures/data"
)

# Création d'un profil tenant
profile = generator.create_tenant_profile(
    tenant_id="demo_tenant_001",
    tenant_name="Demo Tenant",
    tier=TenantTier.STANDARD,
    scale_factor=0.5  # 50% des données max du tier
)

# Génération des données
tenant_data = await generator.generate_tenant_data(profile)

# Sauvegarde
await generator.save_tenant_data(tenant_data, format="json")
```

### Configuration par Tier

| Tier | Users | Tracks | Albums | Artists | Events | Scale |
|------|-------|--------|--------|---------|--------|-------|
| **Free** | 10 | 100 | 20 | 30 | 1K | 0.3x |
| **Standard** | 1K | 10K | 2K | 3K | 100K | 0.7x |
| **Premium** | 10K | 100K | 20K | 30K | 1M | 0.9x |
| **Enterprise** | 100K | 1M | 200K | 300K | 10M | 1.0x |

## 🚀 Scripts de Génération

### Scripts Disponibles

#### 1. **Données de Démonstration**
```bash
python generate_test_data.py demo
```
Génère des datasets de démonstration pour tous les tiers :
- `demo_free_001` - Petit dataset pour démos free
- `demo_standard_001` - Dataset moyen pour démos standard  
- `demo_premium_001` - Grand dataset pour démos premium
- `demo_enterprise_001` - Dataset enterprise complet

#### 2. **Tests de Charge**
```bash
python generate_test_data.py load-test
```
Génère des données pour tests de performance :
- `load_test_light` - 100 utilisateurs concurrent
- `load_test_medium` - 500 utilisateurs concurrent
- `load_test_heavy` - 2000 utilisateurs concurrent

#### 3. **Tests de Conformité**
```bash
python generate_test_data.py compliance
```
Génère des données pour tests de conformité :
- `compliance_gdpr_001` - Données GDPR avec PII réel
- `compliance_anonymized_001` - Données entièrement anonymisées

#### 4. **Tests de Migration**
```bash
python generate_test_data.py migration
```
Génère des données pour tests de migration :
- `migration_source` - Tenant source (Standard → Premium)
- `migration_target` - Tenant cible avec structure Premium

#### 5. **Benchmarks de Performance**
```bash
python generate_test_data.py benchmark
```
Génère des datasets pour benchmarking :
- Petits, moyens et grands datasets
- Métriques de performance automatiques
- Rapport de benchmark complet

#### 6. **Scénario Personnalisé**
```bash
python generate_test_data.py custom \
  --tenant-id "my_custom_tenant" \
  --tier "premium" \
  --scale-factor 0.3 \
  --data-types users tracks listening_history
```

#### 7. **Tous les Scénarios**
```bash
python generate_test_data.py all
```

## 🔒 Conformité et Sécurité

### GDPR et Anonymisation

#### Données Conformes GDPR
```python
profile = generator.create_tenant_profile(
    tenant_id="gdpr_tenant",
    tenant_name="GDPR Compliant Tenant",
    tier=TenantTier.PREMIUM,
    gdpr_compliant=True,
    pii_anonymization=False  # PII réel mais traçable
)
```

#### Données Anonymisées
```python
profile = generator.create_tenant_profile(
    tenant_id="anon_tenant",
    tenant_name="Anonymized Tenant",
    tier=TenantTier.ENTERPRISE,
    gdpr_compliant=True,
    pii_anonymization=True  # PII complètement anonymisé
)
```

#### Exemples d'Anonymisation
```json
// Données normales
{
  "display_name": "John Doe",
  "email": "john.doe@example.com",
  "birth_date": "1990-05-15"
}

// Données anonymisées
{
  "display_name": "User 1",
  "email": "user_1@tenant_001.anonymized",
  "birth_date": null
}
```

### Audit et Traçabilité

Chaque génération de données inclut :
- **Metadata de génération** : timestamp, version, paramètres
- **Rapport de conformité** : niveau GDPR, anonymisation appliquée
- **Trail d'audit** : qui, quand, quoi, pourquoi
- **Checksums** : intégrité des données

## 📊 Caractéristiques Réalistes

### Distribution Géographique
- Support multi-pays avec codes ISO
- Distribution réaliste par région
- Fuseaux horaires et langues locales

### Patterns d'Activité
- Activité variable selon l'heure et le jour
- Saisonnalité dans les écoutes
- Patterns d'usage par appareil

### Données Musicales
- 23 genres musicaux couverts
- 12 humeurs/moods différents
- Audio features réalistes (Spotify API)
- Distribution naturelle de popularité

### Comportements Utilisateur
- Taux de completion variables par track
- Contexts d'écoute diversifiés (playlist, album, radio)
- Préférences et découverte musicale

## 🔧 Configuration Avancée

### Variables d'Environnement
```bash
export TEST_DATA_OUTPUT_PATH="/custom/path/data"
export TEST_DATA_SCALE_FACTOR="0.1"
export TEST_DATA_GDPR_MODE="true"
export TEST_DATA_ANONYMIZE_PII="false"
```

### Configuration Personnalisée
```python
# Profil personnalisé
custom_profile = TenantDataProfile(
    tenant_id="custom_001",
    tenant_name="Custom Tenant",
    tier=TenantTier.PREMIUM,
    num_users=5000,
    num_tracks=50000,
    user_activity_level=0.8,
    music_diversity=0.9,
    geographic_spread=["US", "GB", "FR"],
    gdpr_compliant=True,
    pii_anonymization=True,
    generate_load_data=True,
    concurrent_users=500
)
```

### Types de Données Sélectives
```python
# Générer seulement certains types
data_types = [
    DataType.USERS,
    DataType.TRACKS,
    DataType.LISTENING_HISTORY
]

tenant_data = await generator.generate_tenant_data(
    profile, 
    data_types=data_types
)
```

## 📈 Performance et Optimisation

### Métriques de Génération

#### Temps de Génération Typique
- **Free Tier** (5 users, 50 tracks) : ~2 secondes
- **Standard Tier** (1K users, 10K tracks) : ~30 secondes
- **Premium Tier** (10K users, 100K tracks) : ~5 minutes
- **Enterprise Tier** (100K users, 1M tracks) : ~45 minutes

#### Taille des Données
- **Free** : ~2.5MB JSON
- **Standard** : ~250MB JSON
- **Premium** : ~2.5GB JSON
- **Enterprise** : ~15GB JSON

### Optimisations
```python
# Génération en parallèle
import asyncio

async def generate_multiple_tenants():
    tasks = []
    for tenant_config in tenant_configs:
        task = generator.generate_tenant_data(tenant_config)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## 🛠️ Cas d'Usage

### 1. **Tests d'Intégration**
```bash
# Génération rapide pour tests unitaires
python generate_test_data.py custom \
  --tenant-id "test_integration" \
  --tier "standard" \
  --scale-factor 0.01 \
  --data-types users tracks
```

### 2. **Tests de Performance**
```bash
# Dataset de charge pour stress tests
python generate_test_data.py load-test
```

### 3. **Validation de Migration**
```bash
# Données source et cible pour migration
python generate_test_data.py migration
```

### 4. **Entraînement ML**
```bash
# Dataset avec features ML complètes
python generate_test_data.py custom \
  --tenant-id "ml_training" \
  --tier "enterprise" \
  --scale-factor 1.0 \
  --data-types users tracks listening_history ml_features
```

### 5. **Audit de Conformité**
```bash
# Données anonymisées pour audit
python generate_test_data.py compliance
```

## 🔍 Validation et Qualité

### Validation des Données
- **Cohérence référentielle** : Toutes les FK valides
- **Contraintes métier** : Durées, dates, ranges valides
- **Distribution réaliste** : Pas de données parfaitement uniformes
- **Isolation tenant** : Aucune fuite de données entre tenants

### Métriques de Qualité
```json
{
  "data_quality_metrics": {
    "referential_integrity": "100%",
    "null_values": "< 5%",
    "duplicate_keys": "0%",
    "realistic_distribution": "95%",
    "tenant_isolation": "100%"
  }
}
```

## 📚 Exemples Complets

### Générateur Simple
```python
import asyncio
from multi_tenant_data_generator import generate_test_tenant_data, TenantTier

async def main():
    # Génération rapide
    tenant_data = await generate_test_tenant_data(
        tenant_id="quick_test",
        tier=TenantTier.STANDARD,
        scale_factor=0.1
    )
    
    print(f"Generated {len(tenant_data['data']['users'])} users")
    print(f"Generated {len(tenant_data['data']['tracks'])} tracks")

asyncio.run(main())
```

### Générateur de Charge
```python
import asyncio
from multi_tenant_data_generator import generate_load_test_data

async def main():
    # Données pour test de charge
    load_data = await generate_load_test_data(
        tenant_id="load_test_001",
        num_concurrent_users=1000
    )
    
    print(f"Load test data ready for {load_data['profile']['concurrent_users']} users")

asyncio.run(main())
```

### Générateur Enterprise
```python
import asyncio
from multi_tenant_data_generator import MultiTenantDataGenerator, TenantTier, DataType

async def main():
    generator = MultiTenantDataGenerator()
    
    # Profil enterprise complet
    profile = generator.create_tenant_profile(
        tenant_id="enterprise_full",
        tenant_name="Enterprise Full Dataset",
        tier=TenantTier.ENTERPRISE,
        scale_factor=1.0,
        gdpr_compliant=True,
        pii_anonymization=True,
        generate_load_data=True,
        concurrent_users=5000
    )
    
    # Génération complète
    tenant_data = await generator.generate_tenant_data(profile)
    
    # Sauvegarde multi-format
    await generator.save_tenant_data(tenant_data, format="json")
    await generator.save_tenant_data(tenant_data, format="csv")
    
    print("Enterprise dataset generated successfully!")

asyncio.run(main())
```

## 🎯 Bonnes Pratiques

### 1. **Gestion de la Mémoire**
- Utiliser `scale_factor` pour limiter la taille des datasets
- Générer par chunks pour très gros volumes
- Nettoyer les données temporaires

### 2. **Sécurité des Données**
- Toujours utiliser l'anonymisation en production
- Chiffrer les données sensibles à la génération
- Auditer l'accès aux données de test

### 3. **Performance**
- Générer en parallèle quand possible
- Utiliser le cache pour les données référentielles
- Monitorer l'usage mémoire et CPU

### 4. **Maintenance**
- Versioning des schemas de données
- Documentation des changements de format
- Tests de regression sur la génération

---

*Cette documentation est maintenue automatiquement par le système de génération de données de test. Pour des questions spécifiques, contactez l'équipe DevOps.*
