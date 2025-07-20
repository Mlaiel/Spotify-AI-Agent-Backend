# Elasticsearch Migration Scripts Documentation (EN)

**Created by: Spotify AI Agent Core Team**
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Purpose
This directory contains all scripts for managing Elasticsearch indexes, mappings, and data migrations. Scripts are designed for automation, CI/CD, business logic, multilingual, geo, nested, audit, security, and compliance.

## Scripts
- `create_indexes.py` – Create core indexes with production mappings
- `create_all_indexes.py` – Create all indexes (incl. multilingual, geo, nested, advanced)
- `mapping_updates.py` – Apply mapping changes and handle reindexing if needed
- `reindex_data.py` – Reindex data for zero-downtime migrations and upgrades
- `validate_mappings.py` – Validate all mappings and index health (multilingual, geo, nested, advanced)

## Best Practices
- All scripts are idempotent, safe for production, and CI/CD-ready
- Use environment variables for credentials and endpoints
- Scripts support multilingual, geo, nested, audit, lifecycle, and compliance features
- Health-check and validation included for all indexes

## Usage Example
```bash
python create_all_indexes.py
python validate_mappings.py
python mapping_updates.py --index=tracks --mapping=tracks_mapping.json
python reindex_data.py --source=old_index --dest=new_index
```

## Contact
For changes or questions, contact the Core Team via Slack #spotify-ai-agent or GitHub.

