# Documentation des mappings Elasticsearch (FR)

**Créé par l’équipe : Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

---

## Objectif
Ce dossier contient tous les mappings d’index Elasticsearch pour le backend. Ils sont optimisés pour la recherche avancée, l’analytics, l’IA, la sécurité, l’audit et la conformité.

## Bonnes pratiques
- Tous les mappings sont versionnés, relus et gérés en CI/CD.
- Types de champs stricts, analyzers et suggesters pour la performance et la qualité.
- Mappings avec audit, consentement, RGPD, version, tags, lifecycle pour la conformité.
- Analyzer et suggest pour l’autocomplétion et le multilingue.
- Politiques de cycle de vie pour la gestion et la rétention des données.
- Utiliser geo_point et geo_shape pour les requêtes géospatiales.
- Utiliser les analyzers synonymes pour la recherche sémantique.
- Utiliser les objets imbriqués pour l’audit et l’analytics avancés.
- Utiliser les analyzers multilingues pour le contenu international.

## Fichiers
- `artists_mapping.json` – Profils artistes, genres, followers, popularité, audit, consentement, RGPD, suggest
- `playlists_mapping.json` – Playlists, tracks, statut collaboratif, analytics, audit, tags, suggest
- `tracks_mapping.json` – Métadonnées, audio features, recommandations, audit, tags, suggest
- `users_mapping.json` – Profils utilisateurs, rôles, préférences, activité, audit, consentement, RGPD
- `events_mapping.json` – Événements avec géolocalisation, synonymes, participants imbriqués, audit
- `venues_mapping.json` – Lieux avec geo_point, geo_shape (area), suggest, audit
- `multilingual_content_mapping.json` – Contenu multilingue avec analyzers DE/FR/EN, tokenizer custom, synonymes, audit avancé
- `advanced_content_mapping.json` – Contenu avancé : multilingue, synonymes, geo, geo-shape, AI, audit, consentement, GDPR, related content imbriqué, version, lifecycle

---

## Utilisation
Les mappings sont appliqués automatiquement via les scripts de déploiement ou CI/CD. Pour un setup manuel :
```bash
curl -X PUT http://localhost:9200/venues -H 'Content-Type: application/json' -d @venues_mapping.json
```

## Exemple : Requête multilingue
```json
{
  "query": {
    "multi_match": {
      "query": "artiste",
      "fields": [
        "title.de",
        "title.fr",
        "title.en",
        "description.de",
        "description.fr",
        "description.en"
      ],
      "type": "best_fields"
    }
  }
}
```

## Exemple : Synonymes & Analyzer multilingue
```json
{
  "query": {
    "match": {
      "description": {
        "query": "musique",
        "analyzer": "multilingual_analyzer"
      }
    }
  }
}
```

## Exemple : Audit Trail (nested)
```json
{
  "query": {
    "nested": {
      "path": "audit",
      "query": {
        "bool": {
          "must": [
            {"match": {"audit.user_id": "admin-123"}},
            {"range": {"audit.timestamp": {"gte": "now-7d/d"}}}
          ]
        }
      }
    }
  }
}
```

## Sécurité & Conformité
- Tous les mappings incluent audit, consentement, RGPD, version
- Politiques de cycle de vie et rétention appliquées
- Toute modification nécessite une revue métier et sécurité

## Contact
Pour toute modification ou question, contactez le Core Team via Slack #spotify-ai-agent ou GitHub.

