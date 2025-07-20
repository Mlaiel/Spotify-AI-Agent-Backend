# Elasticsearch Migration Documentation (EN)

**Created by: Spotify AI Agent Core Team**
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Purpose
This module contains all mappings, scripts, and automation for managing Elasticsearch indexes, data, and migrations for the Spotify AI Agent backend. It supports advanced search, analytics, AI, multilingual, geo, nested, audit, compliance, lifecycle, and health-check features.

## Structure
- `mappings/` : All index mappings (artists, users, playlists, tracks, events, venues, multilingual, advanced, geo, nested)
- `scripts/` : Automation scripts for index creation, mapping updates, reindexing, validation, health-check, bulk import/export

## Best Practices
- All mappings and scripts are versioned, peer-reviewed, and CI/CD-ready
- Use environment variables for credentials and endpoints
- All changes are tested in staging before production
- Multilingual, geo, nested, audit, lifecycle, compliance, health-check, and rollback are standard
- Use `create_indexes.py` for core setup, `create_all_indexes.py` for full enterprise/AI/analytics setup

## Usage Example
```bash
cd scripts
python create_indexes.py --env=production
python create_all_indexes.py
python validate_mappings.py
python mapping_updates.py --index=tracks --mapping=tracks_mapping.json
python reindex_data.py --source=old_index --dest=new_index
```

## Governance & Security
- All changes require business and security review
- Audit, consent, GDPR/DSGVO, versioning, lifecycle, multilingual, geo, nested, health-check are enforced

## Contact
For changes or questions, contact the Core Team via Slack #spotify-ai-agent or GitHub.

