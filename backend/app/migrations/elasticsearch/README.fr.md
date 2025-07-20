# Documentation des migrations Elasticsearch (FR)

**Créé par l’équipe : Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

---

## Objectif
Ce module contient tous les mappings, scripts et automatisations pour gérer les indexes, données et migrations Elasticsearch du backend. Il supporte la recherche avancée, l’analytics, l’IA, le multilingue, la géolocalisation, les objets imbriqués, l’audit, la conformité, le cycle de vie et le health-check.

## Structure
- `mappings/` : Tous les mappings d’index (artists, users, playlists, tracks, events, venues, multilingue, avancé, geo, nested)
- `scripts/` : Scripts d’automatisation pour création d’index, update de mapping, reindex, validation, health-check, bulk import/export

## Bonnes pratiques
- Tous les mappings et scripts sont versionnés, relus et CI/CD-ready
- Utilisation de variables d’environnement pour les credentials et endpoints
- Tous les changements sont testés sur staging avant la production
- Multilingue, geo, nested, audit, lifecycle, conformité, health-check, rollback sont standards
- Utiliser `create_indexes.py` pour le setup de base, `create_all_indexes.py` pour un setup complet enterprise/IA/analytics

## Exemple
```bash
cd scripts
python create_indexes.py --env=production
python create_all_indexes.py
python validate_mappings.py
python mapping_updates.py --index=tracks --mapping=tracks_mapping.json
python reindex_data.py --source=old_index --dest=new_index
```

## Gouvernance & Sécurité
- Toute modification nécessite une revue métier et sécurité
- Audit, consentement, RGPD, version, lifecycle, multilingue, geo, nested, health-check sont appliqués

## Contact
Pour toute modification ou question, contactez le Core Team via Slack #spotify-ai-agent ou GitHub.

