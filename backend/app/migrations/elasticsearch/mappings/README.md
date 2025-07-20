# Elasticsearch Mappings Documentation (EN)

**Created by: Spotify AI Agent Core Team**
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Purpose
This directory contains all Elasticsearch index mappings for the Spotify AI Agent backend. These mappings are optimized for advanced search, analytics, AI-driven features, security, audit, and compliance.

## Best Practices
- All mappings are versioned, peer-reviewed, and CI/CD-managed.
- Use strict field types, analyzers, and suggesters for performance and data quality.
- Mappings include audit, consent, GDPR, versioning, tags, and lifecycle for compliance.
- Analyzer and suggest fields for autocomplete and multilingual support.
- Lifecycle policies for index management and data retention.
- Use geo_point and geo_shape for location-based queries.
- Use synonym analyzers for semantic search.
- Use nested objects for complex analytics and audit trails.
- Use language-specific analyzers for multilingual content.

## Files
- `artists_mapping.json` – Artist profiles, genres, followers, popularity, audit, consent, GDPR, suggest
- `playlists_mapping.json` – Playlists, tracks, collaborative status, analytics, audit, tags, suggest
- `tracks_mapping.json` – Track metadata, audio features, recommendations, audit, tags, suggest
- `users_mapping.json` – User profiles, roles, preferences, activity, audit, consent, GDPR
- `events_mapping.json` – Events with geo-location, synonym analyzer, nested participants, audit
- `venues_mapping.json` – Venues with geo_point, geo_shape (area), suggest, audit
- `multilingual_content_mapping.json` – Multilingual content with DE/FR/EN analyzers, custom tokenizer, synonyms, advanced audit

---

## Usage
Mappings are applied automatically via deployment scripts or CI/CD pipelines. For manual setup:
```bash
curl -X PUT http://localhost:9200/multilingual_content -H 'Content-Type: application/json' -d @multilingual_content_mapping.json
```

## Example: Multilingual Query (German, French, English)
```json
{
  "query": {
    "multi_match": {
      "query": "Künstler",
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

## Example: Synonym & Multilingual Analyzer
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

## Example: Audit Trail Query
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

## Security & Compliance
- All mappings include audit, consent, GDPR, and versioning fields
- Index lifecycle and retention policies are enforced
- Changes require business and security review

## Contact
For changes or questions, contact the Core Team via Slack #spotify-ai-agent or GitHub.

