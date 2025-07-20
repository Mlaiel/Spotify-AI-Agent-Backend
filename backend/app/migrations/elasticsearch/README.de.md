# Elasticsearch Migration Dokumentation (DE)

**Erstellt von: Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Zweck
Dieses Modul enthält alle Mappings, Skripte und Automatisierungen zur Verwaltung von Elasticsearch-Indizes, Daten und Migrationen für das Backend. Es unterstützt fortgeschrittene Suche, Analytics, KI, Multilingual, Geo, Nested, Audit, Compliance, Lifecycle und Health-Check.

## Struktur
- `mappings/` : Alle Index-Mappings (artists, users, playlists, tracks, events, venues, multilingual, advanced, geo, nested)
- `scripts/` : Automatisierungsskripte für Index-Erstellung, Mapping-Updates, Reindexing, Validierung, Health-Check, Bulk Import/Export

## Best Practices
- Alle Mappings und Skripte sind versioniert, peer-reviewed und CI/CD-ready
- Nutzung von Umgebungsvariablen für Zugangsdaten und Endpunkte
- Alle Änderungen werden im Staging getestet, bevor sie in Produktion gehen
- Multilingual, Geo, Nested, Audit, Lifecycle, Compliance, Health-Check und Rollback sind Standard
- Für klassisches Setup: `create_indexes.py`, für Enterprise/AI/Analytics: `create_all_indexes.py`

## Beispiel
```bash
cd scripts
python create_indexes.py --env=production
python create_all_indexes.py
python validate_mappings.py
python mapping_updates.py --index=tracks --mapping=tracks_mapping.json
python reindex_data.py --source=old_index --dest=new_index
```

## Governance & Security
- Änderungen erfordern Business- und Security-Review
- Audit, Consent, DSGVO, Version, Lifecycle, Multilingual, Geo, Nested, Health-Check enforced

## Kontakt
Für Änderungen oder Fragen: Core Team via Slack #spotify-ai-agent oder GitHub.

