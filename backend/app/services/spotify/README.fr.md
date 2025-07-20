# Module Services Spotify (FR)

## Présentation
Ce module regroupe toute la logique métier avancée et les intégrations industrielles pour l’exploitation des données Spotify, l’analytique et l’automatisation IA.

### Fonctionnalités principales
- Intégration sécurisée et robuste à l’API Spotify (OAuth2, rate limiting, retry, cache)
- Insights artistes avancés (audience, tendances, clustering ML, scoring)
- Gestion intelligente des playlists (reco IA, analytics, ML)
- Monitoring streaming temps réel (détection anomalies, webhooks, stockage sécurisé)
- Analyse profonde des morceaux (features audio, ML, détection plagiat)
- Profilage utilisateur (segmentation, RGPD, anonymisation, synchronisation multi-sources)
- Sécurité totale (gestion tokens, validation, audit, logs, rate limiting)

### Structure
- `spotify_api_service.py` : Intégration API sécurisée, gestion tokens, erreurs
- `artist_insights_service.py` : Analytics audience, clustering ML, tendances
- `playlist_service.py` : Création playlist, reco IA, analytics
- `streaming_service.py` : Monitoring temps réel, anomalies, webhooks
- `track_analysis_service.py` : Extraction features audio, ML, optimisation
- `user_data_service.py` : Profilage, segmentation, RGPD, synchronisation

### Exemple d’utilisation
```python
from .spotify_api_service import SpotifyAPIService
from .artist_insights_service import ArtistInsightsService

api = SpotifyAPIService(client_id, client_secret)
service = ArtistInsightsService(api)
insights = service.get_artist_audience_insights(artist_id)
```

### Sécurité & conformité
- Tous les flux sont sécurisés (OAuth2, validation, logs)
- Données utilisateurs RGPD
- Rate limiting et monitoring inclus

### Extensibilité
- Services modulaires, testables, prêts production
- Hooks ML (TensorFlow, PyTorch, Hugging Face)
- Intégration microservices facilitée

---
Pour la documentation détaillée, voir les docstrings dans chaque fichier service.

