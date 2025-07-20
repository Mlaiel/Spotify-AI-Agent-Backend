# Elasticsearch Mappings Dokumentation (DE)

**Erstellt von: Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Zweck
Dieses Verzeichnis enthält alle Elasticsearch-Index-Mappings für das Backend. Die Mappings sind für fortgeschrittene Suche, Analytics, KI, Security, Audit und Compliance optimiert.

## Best Practices
- Alle Mappings sind versioniert, peer-reviewed und CI/CD-gemanaged.
- Strikte Feldtypen, Analyzer und Suggester für Performance und Datenqualität.
- Mappings mit Audit, Consent, DSGVO, Version, Tags, Lifecycle für Compliance.
- Analyzer und Suggest-Felder für Autocomplete und Mehrsprachigkeit.
- Lifecycle-Policies für Index-Management und Datenaufbewahrung.
- Nutzung von geo_point und geo_shape für Geo-Queries.
- Synonym-Analyzer für semantische Suche.
- Nested-Objekte für komplexe Analytics und Audit.
- Multilinguale Analyzer für internationalen Content.

## Dateien
- `artists_mapping.json` – Künstlerprofile, Genres, Follower, Popularität, Audit, Consent, DSGVO, Suggest
- `playlists_mapping.json` – Playlists, Tracks, Kollaborationsstatus, Analytics, Audit, Tags, Suggest
- `tracks_mapping.json` – Track-Metadaten, Audio-Features, Empfehlungen, Audit, Tags, Suggest
- `users_mapping.json` – Userprofile, Rollen, Präferenzen, Aktivität, Audit, Consent, DSGVO
- `events_mapping.json` – Events mit Geo-Location, Synonym-Analyzer, Nested Participants, Audit
- `venues_mapping.json` – Venues mit geo_point, geo_shape (area), Suggest, Audit
- `multilingual_content_mapping.json` – Multilinguale Inhalte mit DE/FR/EN-Analyzer, Custom Tokenizer, Synonyme, erweitertem Audit
- `advanced_content_mapping.json` – Advanced Content: Multilingual, Synonym, Geo, Geo-Shape, AI, Audit, Consent, GDPR, Related Content (nested), Version, Lifecycle

---

## Nutzung
Mappings werden automatisch per Deployment-Skripten oder CI/CD angewendet. Für manuelles Setup:
```bash
curl -X PUT http://localhost:9200/venues -H 'Content-Type: application/json' -d @venues_mapping.json
```

## Beispiel: Multilinguale Query
```json
{
  "query": {
    "multi_match": {
      "query": "künstler",
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

## Beispiel: Synonym & Multilingual Analyzer
```json
{
  "query": {
    "match": {
      "description": {
        "query": "musik",
        "analyzer": "multilingual_analyzer"
      }
    }
  }
}
```

## Beispiel: Audit Trail Query
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

## Sicherheit & Compliance
- Alle Mappings enthalten Audit, Consent, DSGVO, Version
- Index-Lifecycle- und Aufbewahrungsrichtlinien enforced
- Änderungen erfordern Business- und Security-Review

## Kontakt
Für Änderungen oder Fragen: Core Team via Slack #spotify-ai-agent oder GitHub.

