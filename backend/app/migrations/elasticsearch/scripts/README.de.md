# Elasticsearch Migrationsskripte Dokumentation (DE)

**Erstellt von: Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Zweck
Dieses Verzeichnis enthält alle Skripte zur Verwaltung von Elasticsearch-Indizes, Mappings und Datenmigrationen. Die Skripte sind für Automatisierung, CI/CD, Business-Logik, Multilingual, Geo, Nested, Audit, Security und Compliance ausgelegt.

## Skripte
- `create_indexes.py` – Erstellt die Kern-Indizes mit produktiven Mappings
- `create_all_indexes.py` – Erstellt alle Indizes (inkl. Multilingual, Geo, Nested, Advanced)
- `mapping_updates.py` – Wendet Mapping-Änderungen an und reindext ggf.
- `reindex_data.py` – Reindext Daten für Zero-Downtime-Migrationen
- `validate_mappings.py` – Validiert alle Mappings und Index-Health (Multilingual, Geo, Nested, Advanced)

## Best Practices
- Alle Skripte sind idempotent, produktionssicher und CI/CD-ready
- Nutzung von Umgebungsvariablen für Zugangsdaten und Endpunkte
- Unterstützung für Multilingual, Geo, Nested, Audit, Lifecycle, Compliance
- Health-Check und Validierung für alle Indizes enthalten

## Beispiel
```bash
python create_all_indexes.py
python validate_mappings.py
python mapping_updates.py --index=tracks --mapping=tracks_mapping.json
python reindex_data.py --source=old_index --dest=new_index
```

## Kontakt
Für Änderungen oder Fragen: Core Team via Slack #spotify-ai-agent oder GitHub.

