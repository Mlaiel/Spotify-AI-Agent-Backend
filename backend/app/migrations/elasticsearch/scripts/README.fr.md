# Documentation des scripts de migration Elasticsearch (FR)

**Créé par l’équipe : Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

---

## Objectif
Ce dossier contient tous les scripts pour gérer les indexes, mappings et migrations de données Elasticsearch. Les scripts sont conçus pour l’automatisation, le CI/CD, la logique métier, le multilingue, la géolocalisation, les objets imbriqués, l’audit, la sécurité et la conformité.

## Scripts
- `create_indexes.py` – Crée les indexes principaux avec les mappings de production
- `create_all_indexes.py` – Crée tous les indexes (multilingue, geo, nested, avancé)
- `mapping_updates.py` – Applique les changements de mapping et gère le reindex si besoin
- `reindex_data.py` – Reindexe les données pour des migrations sans interruption
- `validate_mappings.py` – Valide tous les mappings et la santé des indexes (multilingue, geo, nested, avancé)

## Bonnes pratiques
- Tous les scripts sont idempotents, sûrs pour la production et CI/CD-ready
- Utilisation de variables d’environnement pour les credentials et endpoints
- Support du multilingue, geo, nested, audit, lifecycle, conformité
- Health-check et validation inclus pour tous les indexes

## Exemple
```bash
python create_all_indexes.py
python validate_mappings.py
python mapping_updates.py --index=tracks --mapping=tracks_mapping.json
python reindex_data.py --source=old_index --dest=new_index
```

## Contact
Pour toute modification ou question, contactez le Core Team via Slack #spotify-ai-agent ou GitHub.

